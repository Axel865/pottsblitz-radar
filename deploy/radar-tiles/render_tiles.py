#!/usr/bin/env python3
"""
PottsBlitz — DWD radar tile renderer.

Downloads the current DWD "WN" radar composite (opendata.dwd.de, CC BY 4.0,
updated every 5 minutes, includes a 2h-back history via separate archived
timestamps and a 2h-forward nowcast baked into each file), decodes the
ODIM_H5/RADOLAN reflectivity grid, and renders it as standard XYZ map tiles
(256x256 PNG, Web Mercator) at a few fixed zoom levels — the same scheme
OpenStreetMap/CARTO tiles use, so a normal Leaflet/MapLibre tileLayer can
show it with native sharpness at every supported zoom, instead of stretching
one fixed-resolution image.

Output layout (relative to --out):
    f<NN>/<zoom>/<x>/<y>.png
  NN = 00..48, same meaning as the frames array below (00 = -120 min,
  24 = now, 48 = +120 min). Frame folders are overwritten in place every
  run — stable paths, no cleanup step needed.
  A tile is only written if it has at least one non-transparent pixel
  (most of the grid is clear sky), which keeps real-world tile counts far
  below the theoretical max for the configured area/zoom levels.

Run standalone:  python3 render_tiles.py --out ./tiles
Used by: .github/workflows/radar-tiles.yml (scheduled, uploads --out via FTP)
"""
import argparse
import io
import math
import os
import re
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np
import requests
from PIL import Image
from pyproj import Transformer

TILE = 256
HEADERS = {"User-Agent": "PottsBlitz-radar-tiles/1.0 (+https://emscherwetter.de)"}
BASE_URL = "https://opendata.dwd.de/weather/radar/composite/wn/"

# Zoom levels to render, each with its own geographic box — wider area at low
# zoom (cheap, few tiles), a tighter crop at the highest zoom (many tiles per
# area, so keep the area modest). Boxes deliberately overlap: Leaflet falls
# back to the nearest *lower* zoom's tiles (scaled) while panning just
# outside a tier's box, which still looks fine since that's the same
# resolution a plain single-image overlay would have given everywhere.
ZOOM_TIERS = [
    {"zoom": 6, "min_lon": -1.0, "max_lon": 19.0, "min_lat": 44.5, "max_lat": 56.0},
    {"zoom": 8, "min_lon": 3.0, "max_lon": 16.0, "min_lat": 46.5, "max_lat": 55.5},
    {"zoom": 10, "min_lon": 5.5, "max_lon": 8.8, "min_lat": 50.3, "max_lat": 52.7},
    # 12/13 go all the way to street level, so the area has to shrink a lot
    # more per step (tile count grows ~4x per zoom level) — Ruhrgebiet-sized,
    # not all of Germany. See README for why this can't cover the whole country.
    {"zoom": 12, "min_lon": 6.6, "max_lon": 7.6, "min_lat": 51.15, "max_lat": 51.85},
    {"zoom": 13, "min_lon": 6.9, "max_lon": 7.3, "min_lat": 51.35, "max_lat": 51.65},
]

# Europe mode (--europe): OPERA composite outside the DWD coverage, blended at
# its edge (see europe.py). Coarse tiers for all of Europe; "k" = how many
# source pixels (1 km) are averaged into one for that zoom level.
EUROPE_BOX = {"min_lon": -25.0, "max_lon": 42.0, "min_lat": 33.0, "max_lat": 70.0}
EUROPE_TIERS = [
    dict(EUROPE_BOX, zoom=4, europe=True, k=6),
    dict(EUROPE_BOX, zoom=5, europe=True, k=3),
    dict(EUROPE_BOX, zoom=6, europe=True, k=1),
    # z8 keeps its Germany-sized box but gets OPERA where the DWD coverage ends
    {"zoom": 8, "min_lon": 3.0, "max_lon": 16.0, "min_lat": 46.5, "max_lat": 55.5, "europe": True, "k": 1},
]


def active_tiers(europe):
    if not europe:
        return ZOOM_TIERS
    by_zoom = {t["zoom"]: t for t in ZOOM_TIERS}
    for t in EUROPE_TIERS:
        by_zoom[t["zoom"]] = t
    return [by_zoom[z] for z in sorted(by_zoom)]


FRAME_COUNT = 49  # 24 past (-120..-5 min, 5-min steps) + now + 24 future (+5..+120)
NOW_INDEX = 24

STOPS = [
    (5, (100, 200, 255)), (15, (0, 150, 255)), (25, (0, 200, 0)),
    (35, (255, 255, 0)), (45, (255, 120, 0)), (55, (255, 0, 0)), (65, (200, 0, 200)),
]


def color_for(v):
    if v < STOPS[0][0]:
        return STOPS[0][1]
    for i in range(len(STOPS) - 1):
        v0, c0 = STOPS[i]
        v1, c1 = STOPS[i + 1]
        if v0 <= v <= v1:
            t = (v - v0) / (v1 - v0)
            return tuple(int(c0[k] + (c1[k] - c0[k]) * t) for k in range(3))
    return STOPS[-1][1]


LUT = np.zeros((256, 3), dtype=np.float32)
for _i in range(256):
    LUT[_i] = color_for(_i / 256 * 70)

# Snow: white -> violet with intensity (same dBZ axis as rain). The lightest
# step is a pale lavender rather than pure white so light snow still reads on
# the light OSM base map.
SNOW_STOPS = [
    (12, (222, 204, 255)), (20, (196, 164, 255)), (28, (165, 115, 245)),
    (36, (135, 70, 228)), (45, (108, 35, 195)), (55, (85, 10, 155)),
]


def snow_color_for(v):
    if v <= SNOW_STOPS[0][0]:
        return SNOW_STOPS[0][1]
    for i in range(len(SNOW_STOPS) - 1):
        v0, c0 = SNOW_STOPS[i]
        v1, c1 = SNOW_STOPS[i + 1]
        if v0 <= v <= v1:
            t = (v - v0) / (v1 - v0)
            return tuple(c0[k] + (c1[k] - c0[k]) * t for k in range(3))
    return SNOW_STOPS[-1][1]


SNOW_LUT = np.zeros((256, 3), dtype=np.float32)
for _i in range(256):
    SNOW_LUT[_i] = snow_color_for(_i / 256 * 70)


def latlon_to_tilef(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_merc(xt, yt, zoom):
    n = 2.0 ** zoom
    lon = xt / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * yt / n)))
    lat = math.degrees(lat_rad)
    R = 6378137.0
    mx = math.radians(lon) * R
    my = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R
    return mx, my


def fetch_tar(session, timestamp, attempts=4, retry_delay=3.0):
    """timestamp: 'YYYYMMDD_HHMM' or 'LATEST' sentinel for the live double-underscore alias.

    Transient connection resets/timeouts against opendata.dwd.de are common
    enough (seen in practice, not hypothetical) that a single failed fetch
    must not turn into a missing animation frame — retry a few times with a
    short delay before giving up on this particular timestamp."""
    if timestamp == "LATEST":
        url = BASE_URL + "composite_wn__LATEST.tar"
    else:
        url = BASE_URL + f"composite_wn_{timestamp}.tar"
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return tarfile.open(fileobj=io.BytesIO(r.content))
        except Exception as e:
            last_err = e
            if attempt < attempts:
                print(f"    fetch {timestamp} attempt {attempt}/{attempts} failed ({e}), retrying...")
                time.sleep(retry_delay)
    raise last_err


def list_dir(session):
    r = session.get(BASE_URL, timeout=20)
    r.raise_for_status()
    return re.findall(r"composite_wn_(\d{8}_\d{4})\.tar", r.text)


def decode_member(tar, member_name):
    f = h5py.File(io.BytesIO(tar.extractfile(member_name).read()), "r")
    data = f["dataset1/data1/data"][:]
    attrs = f["dataset1/data1/what"].attrs
    gain = float(attrs["gain"])
    offset = float(attrs["offset"])
    nodata = float(attrs["nodata"])
    undetect = float(attrs["undetect"])
    where = f["where"].attrs
    projdef = where["projdef"].decode() if isinstance(where["projdef"], bytes) else str(where["projdef"])
    xsize, ysize = int(where["xsize"]), int(where["ysize"])
    xscale, yscale = float(where["xscale"]), float(where["yscale"])
    ul_lon, ul_lat = float(where["UL_lon"]), float(where["UL_lat"])
    t_ll_to_dwd = Transformer.from_crs("EPSG:4326", projdef, always_xy=True)
    ul_x, ul_y = t_ll_to_dwd.transform(ul_lon, ul_lat)

    dbz = data.astype(np.float32) * np.float32(gain) + np.float32(offset)
    valid = ((data != nodata) & (data != undetect)).astype(np.float32)
    dbz_filled = np.where(valid > 0, dbz, np.float32(-30.0)).astype(np.float32)
    return {
        "dbz": dbz_filled, "valid": valid, "cover": (data != nodata).astype(np.float32), "projdef": projdef,
        "xsize": xsize, "ysize": ysize, "xscale": xscale, "yscale": yscale,
        "ul_x": ul_x, "ul_y": ul_y,
    }


_TRANSFORMER_CACHE = {}


def get_transformer(projdef):
    # Building a pyproj Transformer has real fixed overhead — DWD's projdef
    # never changes between frames/tiles, so build each one exactly once
    # instead of once per tile (that was the original bottleneck: thousands
    # of Transformer.from_crs() calls per run).
    if projdef not in _TRANSFORMER_CACHE:
        _TRANSFORMER_CACHE[projdef] = Transformer.from_crs("EPSG:3857", projdef, always_xy=True)
    return _TRANSFORMER_CACHE[projdef]


_GEOMETRY_CACHE = {}


def tier_geometry(frame, zoom, xt0, yt0, xt1, yt1):
    """Pixel -> DWD-grid lookup for one zoom tier. This is by far the most
    expensive part of rendering (one pyproj transform per output pixel, ~26M
    pixels per frame across all tiers) and it is identical for every frame —
    the DWD grid never moves — so compute it once per run instead of 49
    times. That was what made a run take ~11 minutes on GitHub's runners and
    left "Jetzt" in the app 15-25 minutes behind DWD."""
    key = (frame["projdef"], frame["ul_x"], frame["ul_y"], frame["xscale"], frame["yscale"],
           frame["xsize"], frame["ysize"], zoom, xt0, yt0, xt1, yt1)
    if key in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[key]
    n = 2.0 ** zoom
    mpp = 2 * math.pi * 6378137.0 / n / TILE
    merc_x0, merc_y0 = tile_to_merc(xt0, yt0, zoom)
    w = (xt1 - xt0 + 1) * TILE
    h = (yt1 - yt0 + 1) * TILE

    t_merc_to_dwd = get_transformer(frame["projdef"])
    cols = np.arange(w)
    rows = np.arange(h)
    oc, orow = np.meshgrid(cols, rows)
    merc_x = merc_x0 + oc * mpp
    merc_y = merc_y0 - orow * mpp
    dwd_x, dwd_y = t_merc_to_dwd.transform(merc_x.ravel(), merc_y.ravel())
    dwd_x = dwd_x.reshape(oc.shape)
    dwd_y = dwd_y.reshape(orow.shape)

    col_f = (dwd_x - frame["ul_x"]) / frame["xscale"]
    row_f = (frame["ul_y"] - dwd_y) / frame["yscale"]
    xsize, ysize = frame["xsize"], frame["ysize"]
    inside = ((col_f >= 0) & (col_f < xsize - 1) & (row_f >= 0) & (row_f < ysize - 1)).astype(np.float32)

    col0 = np.clip(np.floor(col_f).astype(np.int32), 0, xsize - 2)
    row0 = np.clip(np.floor(row_f).astype(np.int32), 0, ysize - 2)
    fc = np.clip(col_f - col0, 0, 1).astype(np.float32)
    fr = np.clip(row_f - row0, 0, 1).astype(np.float32)
    # Flat indices into the (ysize, xsize) grid for the 4 neighbours — one
    # np.take per neighbour later is much cheaper than 2-D fancy indexing.
    i00 = row0 * xsize + col0
    # Web Mercator is separable: longitude depends only on the column,
    # latitude only on the row — used to look up the snow grid (ICON-D2 is a
    # plain lat/lon grid) without another per-pixel projection.
    R = 6378137.0
    lon_cols = np.degrees((merc_x0 + cols * mpp) / R)
    lat_rows = np.degrees(2 * np.arctan(np.exp((merc_y0 - rows * mpp) / R)) - np.pi / 2)
    geo = {"w": w, "h": h, "i00": i00, "i10": i00 + 1, "i01": i00 + xsize, "i11": i00 + xsize + 1,
           "fc": fc, "fr": fr, "inside": inside, "lon_cols": lon_cols, "lat_rows": lat_rows}
    _GEOMETRY_CACHE[key] = geo
    return geo


def render_tier_mosaic(frame, zoom, xt0, yt0, xt1, yt1, snow=None):
    """Render an entire zoom tier's tile range in one vectorized pass, as a
    single big RGBA array covering (xt1-xt0+1)*256 x (yt1-yt0+1)*256 pixels.
    Individual tiles are sliced out of this afterward."""
    geo = tier_geometry(frame, zoom, xt0, yt0, xt1, yt1)
    w, h = geo["w"], geo["h"]
    fc, fr = geo["fc"], geo["fr"]

    def bilerp(grid):
        flat = grid.ravel()
        v00, v10 = flat.take(geo["i00"]), flat.take(geo["i10"])
        v01, v11 = flat.take(geo["i01"]), flat.take(geo["i11"])
        return (v00 * (1 - fc) + v10 * fc) * (1 - fr) + (v01 * (1 - fc) + v11 * fc) * fr

    dbz = bilerp(frame["dbz"])
    valid = bilerp(frame["valid"]) * geo["inside"]
    snow_frac = snow.sample(geo["lat_rows"], geo["lon_cols"]) if snow is not None else None
    return colorize(dbz, valid, h, w, snow_frac)


# Below ~12 dBZ is less than 0.2 mm/h (Marshall-Palmer, Z = 200 R^1.6) — the
# pale-blue haze that is mostly drizzle-at-best and radar clutter (ground
# echoes, wind farms, fog). Was 7 dBZ (0.1 mm/h); raised on request. Not
# drawn at all; fade in over the next few dBZ so the edge of a rain area
# stays soft instead of a hard cut.
MIN_DBZ = 12.0
FADE_DBZ = 4.0


def colorize(dbz, valid, h, w, snow_frac=None):
    t = np.clip((dbz - MIN_DBZ) / FADE_DBZ, 0, 1)
    smooth_alpha = t * t * (3 - 2 * t)
    alpha = (smooth_alpha * valid * 180).astype(np.uint8)

    idx = np.clip((dbz / 70 * 256), 0, 255).astype(np.int32)
    rgb = LUT[idx].astype(np.uint8)
    if snow_frac is not None:
        # Rain -> snow blended per pixel, so sleet (snow fraction between 0
        # and 1) lands visibly between the two palettes instead of a hard
        # boundary along a model grid line. Only on pixels that are actually
        # drawn and have any snow — usually a small share of the mosaic.
        m = (alpha > 0) & (snow_frac > 0)
        if m.any():
            sf = snow_frac[m][:, None]
            im = idx[m]
            rgb[m] = (LUT[im] * (1 - sf) + SNOW_LUT[im] * sf).astype(np.uint8)
            # Light colours need a bit more opacity to stand out on the
            # light base map.
            alpha[m] = np.minimum(alpha[m].astype(np.float32) * (1 + 0.25 * sf[:, 0]), 255).astype(np.uint8)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., 0:3] = rgb
    out[..., 3] = alpha
    return out


def tile_range_for_box(zoom, min_lon, max_lon, min_lat, max_lat):
    x0f, y0f = latlon_to_tilef(max_lat, min_lon, zoom)
    x1f, y1f = latlon_to_tilef(min_lat, max_lon, zoom)
    xt0, xt1 = int(math.floor(x0f)), int(math.floor(x1f))
    yt0, yt1 = int(math.floor(y0f)), int(math.floor(y1f))
    return xt0, yt0, xt1, yt1


def render_frame_tiles(frame, out_dir, frame_label, snow=None, opera=None, europe=False):
    n_written = 0
    for tier in active_tiers(europe):
        zoom = tier["zoom"]
        xt0, yt0, xt1, yt1 = tile_range_for_box(
            zoom, tier["min_lon"], tier["max_lon"], tier["min_lat"], tier["max_lat"])
        if tier.get("europe"):
            import europe as eu
            mosaic = eu.blended_mosaic(sys.modules[__name__], frame, opera, zoom, xt0, yt0, xt1, yt1,
                                       tier["k"], snow)
        else:
            mosaic = render_tier_mosaic(frame, zoom, xt0, yt0, xt1, yt1, snow)
        for xi, xt in enumerate(range(xt0, xt1 + 1)):
            for yi, yt in enumerate(range(yt0, yt1 + 1)):
                chunk = mosaic[yi * TILE:(yi + 1) * TILE, xi * TILE:(xi + 1) * TILE]
                if chunk[..., 3].max() == 0:
                    continue
                tile_dir = os.path.join(out_dir, frame_label, str(zoom), str(xt))
                os.makedirs(tile_dir, exist_ok=True)
                img = Image.fromarray(chunk, "RGBA")
                # Quantizing to a small palette (the color ramp only has 7
                # stops anyway) cuts file size roughly 4x vs. full RGBA with
                # no visible loss — important since this gets re-uploaded
                # every cycle.
                img.quantize(colors=64, method=Image.FASTOCTREE).save(
                    os.path.join(tile_dir, f"{yt}.png"), "PNG", optimize=True)
                n_written += 1
    return n_written


ICON_BASE = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/"


def fetch_icon_field(session, run_dt, step, var):
    """One ICON-D2 single-level field on DWD's regular 0.02 deg lat/lon grid
    (rows south -> north). Returns (values[Nj, Ni] float32 with NaN for
    missing, grid dict)."""
    import bz2
    import eccodes
    run = run_dt.strftime("%Y%m%d%H")
    url = (f"{ICON_BASE}{run_dt:%H}/{var}/"
           f"icon-d2_germany_regular-lat-lon_single-level_{run}_{step:03d}_2d_{var}.grib2.bz2")
    r = session.get(url, timeout=30)
    r.raise_for_status()
    h = eccodes.codes_new_from_message(bz2.decompress(r.content))
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dlat = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
        dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        if not eccodes.codes_get(h, "jScansPositively"):
            raise ValueError("unexpected ICON-D2 scan order")
        missing = eccodes.codes_get(h, "missingValue")
        vals = eccodes.codes_get_values(h).astype(np.float32).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    vals[vals == missing] = np.nan
    if lon0 > 180:
        lon0 -= 360
    return vals, {"lat0": lat0, "lon0": lon0, "dlat": dlat, "dlon": dlon, "ni": ni, "nj": nj}


def wet_bulb_c(t_c, td_c):
    """Wet-bulb temperature (deg C) from temperature and dew point (Stull 2011).
    Wet-bulb rather than plain air temperature because falling snow cools the
    air by evaporating into it — dry air at +2 deg C still gives snow."""
    rh = 100.0 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c))
    rh = np.clip(rh, 1.0, 100.0)
    return (t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t_c + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


# Snow fraction from wet-bulb temperature: full snow at or below SNOW_TW_FULL,
# rain at or above SNOW_TW_NONE, sleet in between (common forecasting rule of
# thumb: ~50% chance of snow around Tw = 1 deg C).
SNOW_TW_FULL = 0.5
SNOW_TW_NONE = 1.5


class SnowModel:
    """Hourly snow-fraction grids (0 = rain, 1 = snow) for the time span the
    animation covers, from the newest ICON-D2 run that has all needed steps."""

    def __init__(self, session, t_from, t_to):
        now = datetime.utcnow()
        latest_run = now.replace(minute=0, second=0, microsecond=0)
        latest_run -= timedelta(hours=latest_run.hour % 3)
        hours = []
        h = t_from.replace(minute=0, second=0, microsecond=0)
        while h <= t_to + timedelta(hours=1):
            hours.append(h)
            h += timedelta(hours=1)
        last_err = None
        for back in range(0, 5):  # runs every 3 h, usually ~1.5-2 h until online
            run = latest_run - timedelta(hours=3 * back)
            try:
                self._load(session, run, hours)
                print(f"  snow detection: ICON-D2 run {run:%Y-%m-%d %H} UTC, {len(hours)} hours")
                return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"no usable ICON-D2 run ({last_err})")

    def _load(self, session, run, hours):
        grids = {}
        grid = None
        for hour in hours:
            step = max(0, int((hour - run).total_seconds() // 3600))
            t, grid = fetch_icon_field(session, run, step, "t_2m")
            td, _ = fetch_icon_field(session, run, step, "td_2m")
            tw = wet_bulb_c(t - 273.15, td - 273.15)
            frac = np.clip((SNOW_TW_NONE - tw) / (SNOW_TW_NONE - SNOW_TW_FULL), 0, 1)
            grids[hour] = np.nan_to_num(frac, nan=0.0).astype(np.float32)
        self.grids, self.grid = grids, grid
        self.hours = sorted(grids)

    def at(self, frame_dt):
        """Snow-fraction field for one frame time, linearly blended between
        the two surrounding model hours."""
        h0 = frame_dt.replace(minute=0, second=0, microsecond=0)
        h0 = min(max(h0, self.hours[0]), self.hours[-1])
        h1 = min(h0 + timedelta(hours=1), self.hours[-1])
        w1 = min(max((frame_dt - h0).total_seconds() / 3600.0, 0.0), 1.0) if h1 != h0 else 0.0
        return SnowField(self.grids[h0] * (1 - w1) + self.grids[h1] * w1, self.grid)


class SnowField:
    def __init__(self, values, grid):
        self.values, self.g = values, grid

    def edge_weight(self, lat_rows, lon_cols, feather_deg=0.3):
        """1 well inside this model's area, falling to 0 over `feather_deg`
        at its edge (and 0 outside) — to hand over to a wider model smoothly."""
        g = self.g
        lat_min, lat_max = g["lat0"], g["lat0"] + g["dlat"] * (g["nj"] - 1)
        lon_min, lon_max = g["lon0"], g["lon0"] + g["dlon"] * (g["ni"] - 1)
        wj = np.clip(np.minimum(lat_rows - lat_min, lat_max - lat_rows) / feather_deg, 0, 1)
        wi = np.clip(np.minimum(lon_cols - lon_min, lon_max - lon_cols) / feather_deg, 0, 1)
        return (wj[:, None] * wi[None, :]).astype(np.float32)

    def sample(self, lat_rows, lon_cols):
        """Bilinear lookup for a mosaic given its per-row latitudes and
        per-column longitudes -> (len(lat_rows), len(lon_cols)) array."""
        g = self.g
        fj = (lat_rows - g["lat0"]) / g["dlat"]
        fi = (lon_cols - g["lon0"]) / g["dlon"]
        inside_j = (fj >= 0) & (fj <= g["nj"] - 1)
        inside_i = (fi >= 0) & (fi <= g["ni"] - 1)
        j0 = np.clip(np.floor(fj).astype(np.int32), 0, g["nj"] - 2)
        i0 = np.clip(np.floor(fi).astype(np.int32), 0, g["ni"] - 2)
        wj = np.clip(fj - j0, 0, 1).astype(np.float32)[:, None]
        wi = np.clip(fi - i0, 0, 1).astype(np.float32)[None, :]
        rows = self.values[j0] * (1 - wj) + self.values[j0 + 1] * wj  # (h, Ni)
        out = rows[:, i0] * (1 - wi) + rows[:, i0 + 1] * wi
        out *= inside_j[:, None] & inside_i[None, :]
        return out



ICON_EU_BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/"


def fetch_icon_eu_field(session, run_dt, step, var):
    """One ICON-EU single-level field on DWD's regular lat/lon grid
    (see fetch_icon_field for the ICON-D2 twin)."""
    import bz2
    import eccodes
    run = run_dt.strftime("%Y%m%d%H")
    url = (f"{ICON_EU_BASE}{run_dt:%H}/{var}/"
           f"icon-eu_europe_regular-lat-lon_single-level_{run}_{step:03d}_{var.upper()}.grib2.bz2")
    r = session.get(url, timeout=60)
    r.raise_for_status()
    h = eccodes.codes_new_from_message(bz2.decompress(r.content))
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dlat = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
        dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        if not eccodes.codes_get(h, "jScansPositively"):
            raise ValueError("unexpected ICON-EU scan order")
        missing = eccodes.codes_get(h, "missingValue")
        vals = eccodes.codes_get_values(h).astype(np.float32).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    vals[vals == missing] = np.nan
    if lon0 > 180:
        lon0 -= 360
    return vals, {"lat0": lat0, "lon0": lon0, "dlat": dlat, "dlon": dlon, "ni": ni, "nj": nj}


class SnowModelEU(SnowModel):
    """Same as SnowModel, but from ICON-EU (all of Europe, ~6 km) instead of
    ICON-D2 (central Europe only, ~2 km)."""

    def __init__(self, session, t_from, t_to):  # noqa: D107 — same signature, other source
        now = datetime.utcnow()
        latest_run = now.replace(minute=0, second=0, microsecond=0)
        latest_run -= timedelta(hours=latest_run.hour % 3)
        hours = []
        h = t_from.replace(minute=0, second=0, microsecond=0)
        while h <= t_to + timedelta(hours=1):
            hours.append(h)
            h += timedelta(hours=1)
        last_err = None
        for back in range(0, 5):
            run = latest_run - timedelta(hours=3 * back)
            try:
                self._load(session, run, hours)
                print(f"  snow detection Europe: ICON-EU run {run:%Y-%m-%d %H} UTC, {len(hours)} hours")
                return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"no usable ICON-EU run ({last_err})")

    def _load(self, session, run, hours):
        grids, grid = {}, None
        for hour in hours:
            step = max(0, int((hour - run).total_seconds() // 3600))
            t, grid = fetch_icon_eu_field(session, run, step, "t_2m")
            td, _ = fetch_icon_eu_field(session, run, step, "td_2m")
            tw = wet_bulb_c(t - 273.15, td - 273.15)
            frac = np.clip((SNOW_TW_NONE - tw) / (SNOW_TW_NONE - SNOW_TW_FULL), 0, 1)
            grids[hour] = np.nan_to_num(frac, nan=0.0).astype(np.float32)
        self.grids, self.grid = grids, grid
        self.hours = sorted(grids)


class _CombinedSnowField:
    """ICON-D2 where it exists (feathered at its edge), ICON-EU elsewhere."""

    def __init__(self, fine, wide):
        self.fine, self.wide = fine, wide

    def sample(self, lat_rows, lon_cols):
        if self.fine is None:
            return self.wide.sample(lat_rows, lon_cols)
        a = self.fine.sample(lat_rows, lon_cols)
        if self.wide is None:
            return a
        w = self.fine.edge_weight(lat_rows, lon_cols)
        return a * w + self.wide.sample(lat_rows, lon_cols) * (1 - w)


class CombinedSnowModel:
    def __init__(self, fine, wide):
        self.fine, self.wide = fine, wide

    def at(self, frame_dt):
        return _CombinedSnowField(self.fine.at(frame_dt) if self.fine else None,
                                  self.wide.at(frame_dt) if self.wide else None)


_LATEST_TAR = None
_OUT_DIR = None
_SNOW = None
_OPERA = None      # europe.OperaArchive when running with --europe
_FORECAST = None   # (Flow, pooled newest OPERA frame, its time) for the frames after "now"
_WIND = None       # (u, v, grid) ICON-EU 700 hPa wind, also used for the movement arrow outside the DWD area
_NOW_DT = None
_THREAD = threading.local()


def render_job(job):
    """One animation frame. Runs in a worker thread (or the main thread for
    the first one); returns (label, tiles_written, log line)."""
    kind, src, label, desc, frame_dt = job
    try:
        if kind == "latest":
            frame = decode_member(_LATEST_TAR, src)
        else:
            session = getattr(_THREAD, "session", None)
            if session is None:
                session = _THREAD.session = requests.Session()
                session.headers.update(HEADERS)
            tar = fetch_tar(session, src)
            member = f"composite_wn_{src}_000-hd5"
            if member not in tar.getnames():
                return label, 0, f"  {label}: member missing in {src}.tar, skipping"
            frame = decode_member(tar, member)
        snow = _SNOW.at(frame_dt) if _SNOW is not None else None
        opera, note = None, ""
        if _OPERA is not None and frame_dt <= _NOW_DT:
            t = _OPERA.nearest(frame_dt)
            if t is not None:
                opera = _OPERA.fetch(t)
                note = f", OPERA {t:%H:%M}"
            else:
                note = ", no OPERA frame"
        elif _OPERA is not None and _FORECAST is not None:
            # Beyond "now": outside the DWD coverage the newest OPERA frame
            # is moved with the model wind (the DWD nowcast covers its own area).
            flow, base, base_dt = _FORECAST
            opera = flow.advect(base, (frame_dt - base_dt).total_seconds() / 60.0)
            note = f", OPERA {base_dt:%H:%M} advected"
        n = render_frame_tiles(frame, _OUT_DIR, label, snow, opera, europe=_OPERA is not None)
        return label, n, f"  {label} ({desc}{note}): {n} tiles"
    except Exception as e:
        return label, 0, f"  {label} ({desc}): failed ({e}), skipping"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for the tile tree")
    ap.add_argument("--europe", action="store_true",
                    help="also render Europe-wide tiles using the OPERA composite (see europe.py)")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    print("Fetching current frame set (LATEST, includes forecast steps)...")
    latest_tar = fetch_tar(session, "LATEST")
    latest_members = sorted(latest_tar.getnames())
    if not latest_members:
        print("ERROR: LATEST tar is empty", file=sys.stderr)
        sys.exit(1)
    # filename: composite_wn_YYYYMMDD_HHMM_MMM-hd5  (MMM = forecast minute offset)
    m0 = re.match(r"composite_wn_(\d{8})_(\d{4})_\d+-hd5", latest_members[0])
    now_dt = datetime.strptime(m0.group(1) + m0.group(2), "%Y%m%d%H%M")
    print(f"  now = {now_dt.isoformat()}, {len(latest_members)} forecast steps")

    # Render "now" first in this process: that builds the per-tier geometry
    # cache (see tier_geometry) once, which the worker threads below then share
    # instead of racing to compute it several times in parallel.
    future_sorted = sorted(latest_members, key=lambda n: int(re.search(r"_(\d+)-hd5", n).group(1)))
    jobs = []
    for name in future_sorted:
        minute = int(re.search(r"_(\d+)-hd5", name).group(1))
        if minute % 5 != 0 or minute > 120:
            continue
        jobs.append(("latest", name, f"f{NOW_INDEX + minute // 5:02d}", f"+{minute} min",
                     now_dt + timedelta(minutes=minute)))

    # Past: archived timestamps now-120 .. now-5, each file's own "000" member
    # is that past moment's real observation.
    available = set(list_dir(session))
    for steps_back in range(24, 0, -1):
        ts_dt = now_dt - timedelta(minutes=5 * steps_back)
        ts = ts_dt.strftime("%Y%m%d_%H%M")
        label = f"f{NOW_INDEX - steps_back:02d}"
        if ts not in available:
            print(f"  {label} (-{5*steps_back} min): source {ts} not available, skipping")
            continue
        jobs.append(("past", ts, label, f"-{5*steps_back} min", ts_dt))

    # Rain vs. snow: the radar only measures intensity, so precipitation
    # type comes from DWD's ICON-D2 model (2 m temperature + dew point). Any
    # failure here just means everything is drawn as rain, never a broken run.
    global _LATEST_TAR, _OUT_DIR, _SNOW, _OPERA, _NOW_DT, _FORECAST, _WIND
    t_span = (now_dt - timedelta(minutes=120), now_dt + timedelta(minutes=120))
    snow_fine = snow_wide = None
    try:
        snow_fine = SnowModel(session, *t_span)
    except Exception as e:
        print(f"  snow detection unavailable ({e}) — drawing everything as rain")
    if args.europe:
        try:
            snow_wide = SnowModelEU(session, *t_span)
        except Exception as e:
            print(f"  snow detection for Europe unavailable ({e})")
    _SNOW = CombinedSnowModel(snow_fine, snow_wide) if (snow_fine or snow_wide) else None
    _NOW_DT = now_dt
    if args.europe:
        try:
            import europe as eu
            _OPERA = eu.OperaArchive(session, now_dt - timedelta(minutes=125), now_dt)
            newest = max(t for t in _OPERA.times if t <= now_dt)
            print(f"  OPERA: {len(_OPERA.times)} composites, newest {newest:%H:%M} (DWD now {now_dt:%H:%M})")
            try:
                flow, base, run, wind = eu.build_forecaster(session, _OPERA, newest, now_dt)
                _FORECAST = (flow, base, newest)
                _WIND = wind
                print(f"  forecast outside the DWD area: OPERA {newest:%H:%M} moved with ICON-EU {run:%H} UTC 700 hPa wind")
            except Exception as e:
                print(f"  no wind for the forecast ({e}) — frames after now show Germany only")
        except Exception as e:
            print(f"  OPERA unavailable ({e}) — Germany only")
            _OPERA = None
    _LATEST_TAR, _OUT_DIR = latest_tar, args.out
    first, rest = jobs[0], jobs[1:]
    total_tiles = 0
    label, n, msg = render_job(first)
    total_tiles += n
    print(msg)

    # The frames are independent, so render several at once (GitHub's runners
    # have 4 cores). Threads rather than processes: numpy, PIL and the
    # network I/O release the GIL for the heavy parts, the geometry cache is
    # shared for free, and fork()-based pools hang on macOS when testing
    # locally. Together with the geometry cache this brought a run from ~11
    # minutes down to a couple.
    workers = max(2, min((os.cpu_count() or 2) + 2, 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for label, n, msg in pool.map(render_job, rest):
            total_tiles += n
            print(msg, flush=True)

    import json
    os.makedirs(args.out, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    # Precipitation areas + their movement for the home-screen tile's arrow
    # (see nowcast.py). Optional: a failure here only means no arrow.
    try:
        from nowcast import build_nowcast
        import europe as eu
        by_minute = {int(re.search(r"_(\d+)-hd5", n).group(1)): n for n in latest_members}
        if _FORECAST is not None and _WIND is not None:
            nc = eu.build_nowcast_europe(
                decode_member(latest_tar, by_minute[0]), decode_member(latest_tar, by_minute[30]),
                decode_member(latest_tar, by_minute[60]) if 60 in by_minute else None,
                _FORECAST[1], _WIND, now_dt, generated_at)
        else:
            nc = build_nowcast(decode_member(latest_tar, by_minute[0]),
                               decode_member(latest_tar, by_minute[30]), now_dt, generated_at,
                               decode_member(latest_tar, by_minute[60]) if 60 in by_minute else None)
        if nc:
            with open(os.path.join(args.out, "nowcast.json"), "w") as fh:
                json.dump(nc, fh, separators=(",", ":"))
            print(f"  nowcast: {len(nc['areas'])} precipitation areas")
        else:
            print("  nowcast: no precipitation areas")
    except Exception as e:
        print(f"  nowcast unavailable ({e}) — no movement arrow")

    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w") as fh:
        json.dump({
            "generated_at": generated_at,
            "now": now_dt.isoformat(),
            "frame_count": FRAME_COUNT,
            "now_index": NOW_INDEX,
            "step_minutes": 5,
            "zooms": [t["zoom"] for t in active_tiers(_OPERA is not None)],
            "europe": _OPERA is not None,
        }, fh)

    print(f"Done. {total_tiles} tiles written to {args.out}")


if __name__ == "__main__":
    main()
