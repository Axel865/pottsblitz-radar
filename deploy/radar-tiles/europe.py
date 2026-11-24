"""
PottsBlitz — Europe-wide radar: OPERA composite blended with the DWD composite.

The DWD composite only covers Germany and its surroundings. For the rest of
Europe EUMETNET's OPERA composite ("CIRRUS" maximum-reflectivity composite,
dBZ, 1 km grid, LAEA projection, a new file every 5 minutes, CC BY 4.0) is
used. Both are free and need no key: OPERA's 24 h cache is a public S3 bucket.

Inside the DWD coverage the DWD data are used unchanged. Towards its edge
they are blended into OPERA over ~30 km (in linear reflectivity Z, so a
shower that crosses the seam doesn't get halved), outside it OPERA is used.
"""
import io
import re
from datetime import datetime, timedelta

import h5py
import numpy as np
from pyproj import Transformer

# Anonymous public buckets: the primary one and the ECMWF backup the
# project switches to during provider outages (it was the only one with data
# when this was written). Tried in this order.
OPERA_ENDPOINTS = [
    "https://s3.waw3-1.cloudferro.com/openradar-24h",
    "https://object-store.os-api.cci1.ecmwf.int/openradar-24h-backup",
]
OPERA_MIN_DBZ = 14.0     # OPERA shows more weak echoes (clutter) than the DWD one
OPERA_MAX_AGE_MIN = 15   # use an older OPERA frame if the newest is still missing
SEAM_KM = 30             # blend width inside the DWD coverage


def opera_key(dt):
    return f"{dt:%Y/%m/%d}/OPERA/COMP/OPERA@{dt:%Y%m%dT%H%M}@0@DBZH.h5"


def decode_opera(content):
    f = h5py.File(io.BytesIO(content), "r")
    data = f["dataset1/data1/data"][:]
    w = f["dataset1/data1/what"].attrs
    wh = f["where"].attrs
    projdef = wh["projdef"].decode() if isinstance(wh["projdef"], bytes) else str(wh["projdef"])
    to_proj = Transformer.from_crs("EPSG:4326", projdef, always_xy=True)
    ul_x, ul_y = to_proj.transform(float(wh["UL_lon"]), float(wh["UL_lat"]))
    nodata, undetect = float(w["nodata"]), float(w["undetect"])
    cover = data != nodata
    dbz = data.astype(np.float32) * np.float32(w["gain"]) + np.float32(w["offset"])
    valid = cover & (data != undetect) & (dbz >= OPERA_MIN_DBZ)
    dbz = np.where(valid, dbz, np.float32(-30.0)).astype(np.float32)
    return {
        "dbz": dbz, "valid": valid.astype(np.float32), "cover": cover.astype(np.float32),
        "projdef": projdef, "xsize": int(wh["xsize"]), "ysize": int(wh["ysize"]),
        "xscale": float(wh["xscale"]), "yscale": float(wh["yscale"]), "ul_x": ul_x, "ul_y": ul_y,
    }


class OperaArchive:
    """Knows which OPERA composite times exist and fetches/decodes them."""

    def __init__(self, session, t_from, t_to):
        self.session = session
        self.times = set()
        self.base = {}  # dt -> endpoint that has it
        hours = []
        h = t_from.replace(minute=0, second=0, microsecond=0)
        while h <= t_to:
            hours.append(h)
            h += timedelta(hours=1)
        for ep in OPERA_ENDPOINTS:
            for hour in hours:
                prefix = f"{hour:%Y/%m/%d}/OPERA/COMP/OPERA@{hour:%Y%m%dT%H}"
                try:
                    r = session.get(ep, params={"prefix": prefix, "max-keys": 200}, timeout=30)
                    r.raise_for_status()
                except Exception:
                    continue
                for m in re.finditer(r"<Key>[^<]*OPERA@(\d{8}T\d{4})@0@DBZH\.h5</Key>", r.text):
                    dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M")
                    self.base.setdefault(dt, ep)
        self.times = set(self.base)
        if not self.times:
            raise RuntimeError("no OPERA composite found on any endpoint")

    def nearest(self, dt):
        """Newest available OPERA time <= dt (within OPERA_MAX_AGE_MIN), else None."""
        for back in range(0, OPERA_MAX_AGE_MIN + 1, 5):
            cand = dt - timedelta(minutes=back)
            if cand in self.times:
                return cand
        return None

    def fetch(self, dt):
        last = None
        eps = [self.base[dt]] + [e for e in OPERA_ENDPOINTS if e != self.base[dt]]
        for ep in eps:
            try:
                r = self.session.get(f"{ep}/{opera_key(dt)}", timeout=60)
                r.raise_for_status()
                return decode_opera(r.content)
            except Exception as e:  # try the other endpoint
                last = e
        raise RuntimeError(f"OPERA {dt:%Y-%m-%d %H:%M} unavailable ({last})")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def pool(frame, k, weight_radius_km=None):
    """Area-average a frame down by an integer factor k (low zoom levels
    would otherwise point-sample the 1 km grid and speckle). Reflectivity is
    averaged in linear Z, so isolated showers dilute like they physically
    would; the grid scale grows by k, the upper-left corner stays."""
    if k > 1:
        ys, xs = (frame["ysize"] // k) * k, (frame["xsize"] // k) * k

        def blocks(a):
            return a[:ys, :xs].reshape(ys // k, k, xs // k, k)

        z = (10.0 ** (blocks(frame["dbz"]) / 10.0)).mean(axis=(1, 3))
        out = dict(frame)
        out["dbz"] = (10.0 * np.log10(np.maximum(z, 1e-3))).astype(np.float32)
        out["valid"] = blocks(frame["valid"]).max(axis=(1, 3))
        if "cover" in frame:
            out["cover"] = blocks(frame["cover"]).mean(axis=(1, 3))
        out["xsize"], out["ysize"] = xs // k, ys // k
        out["xscale"], out["yscale"] = frame["xscale"] * k, frame["yscale"] * k
        frame = out
    if weight_radius_km is not None and "cover" in frame:
        from nowcast import _box_fraction
        r_px = max(1, int(round(weight_radius_km * 1000.0 / frame["xscale"])))
        cov = frame["cover"] > 0.5
        frac = _box_fraction(cov, 2 * r_px + 1)
        frame = dict(frame)
        frame["weight"] = (np.clip((frac - 0.7) / 0.3, 0.0, 1.0) * cov).astype(np.float32)
    return frame


def _lerp4(geo, grid):
    flat = grid.ravel()
    fc, fr = geo["fc"], geo["fr"]
    v00, v10 = flat.take(geo["i00"]), flat.take(geo["i10"])
    v01, v11 = flat.take(geo["i01"]), flat.take(geo["i11"])
    return (v00 * (1 - fc) + v10 * fc) * (1 - fr) + (v01 * (1 - fc) + v11 * fc) * fr


def blended_mosaic(render, dwd, opera, zoom, xt0, yt0, xt1, yt1, k, snow):
    """RGBA mosaic for one zoom tier: DWD inside its coverage, OPERA outside,
    blended at the seam. `render` is the render_tiles module (geometry cache,
    colouring). `opera` may be None (then DWD only)."""
    d = pool(dwd, k, SEAM_KM)
    geo = render.tier_geometry(d, zoom, xt0, yt0, xt1, yt1)
    w, h = geo["w"], geo["h"]
    inside_d = geo["inside"]
    dbz_d = _lerp4(geo, d["dbz"])
    valid_d = _lerp4(geo, d["valid"]) * inside_d
    if opera is None:
        return render.colorize(dbz_d, valid_d, h, w, _snow(snow, geo))
    # `k` is the target resolution in km; an advected OPERA frame already sits
    # on a coarser (2 km) grid.
    o = pool(opera, max(1, int(round(k * 1000.0 / opera["xscale"]))))
    geo_o = render.tier_geometry(o, zoom, xt0, yt0, xt1, yt1)
    dbz_o = _lerp4(geo_o, o["dbz"])
    valid_o = _lerp4(geo_o, o["valid"]) * geo_o["inside"]
    wd = _lerp4(geo, d["weight"]) * inside_d  # 1 deep inside the DWD coverage, 0 outside
    mix = (wd > 0.0) & (wd < 0.999)
    out_dbz = np.where(wd >= 0.999, dbz_d, dbz_o)
    out_valid = np.where(wd >= 0.999, valid_d, valid_o)
    if mix.any():
        z = wd[mix] * 10.0 ** (dbz_d[mix] / 10.0) + (1 - wd[mix]) * 10.0 ** (dbz_o[mix] / 10.0)
        out_dbz[mix] = 10.0 * np.log10(np.maximum(z, 1e-3))
        out_valid[mix] = wd[mix] * valid_d[mix] + (1 - wd[mix]) * valid_o[mix]
    return render.colorize(out_dbz, out_valid, h, w, _snow(snow, geo))


def _snow(snow, geo):
    return snow.sample(geo["lat_rows"], geo["lon_cols"]) if snow is not None else None


# ---------------------------------------------------------------------------
# Forecast outside the DWD domain: move the newest OPERA frame with the wind
# ---------------------------------------------------------------------------
# The DWD nowcast only exists inside its own coverage. Outside it, the newest
# OPERA frame is advected with the 700 hPa wind of DWD's ICON-EU model.
#
# Why the wind and not a motion measured from the last OPERA frames (as
# nowcast.py does for DWD data): the OPERA composite carries lots of static
# echoes plus persistent frontal bands, and every variant tried (phase
# correlation, normalised cross-correlation, several window sizes, masking of
# static pixels) got pinned to ~zero motion or noise and did not beat simply
# freezing the picture in a hindcast against real OPERA frames. The model wind
# did, by a small margin up to +60 min, and it never produces noise vectors.
FLOW_POOL = 2          # advected frames use a 2 km grid
FLOW_NODE_PX = 32      # wind sampled every 32 px (= 64 km), interpolated between
WIND_LEVEL = 700       # hPa
WIND_FACTOR = 0.8      # rain systems move a bit slower than the 700 hPa wind
ICON_EU_BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/"


def _icon_eu_field(session, run, step, var, level):
    import bz2
    import eccodes
    url = (f"{ICON_EU_BASE}{run:%H}/{var.lower()}/icon-eu_europe_regular-lat-lon_pressure-level_"
           f"{run:%Y%m%d%H}_{step:03d}_{level}_{var}.grib2.bz2")
    r = session.get(url, timeout=60)
    r.raise_for_status()
    h = eccodes.codes_new_from_message(bz2.decompress(r.content))
    try:
        g = {k: eccodes.codes_get(h, k) for k in (
            "Ni", "Nj", "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
            "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees", "jScansPositively")}
        if not g["jScansPositively"]:
            raise ValueError("unexpected ICON-EU scan order")
        vals = eccodes.codes_get_values(h).astype(np.float32).reshape(g["Nj"], g["Ni"])
    finally:
        eccodes.codes_release(h)
    lon0 = g["longitudeOfFirstGridPointInDegrees"]
    if lon0 > 180:
        lon0 -= 360
    return vals, {"lat0": g["latitudeOfFirstGridPointInDegrees"], "lon0": lon0,
                  "dlat": g["jDirectionIncrementInDegrees"], "dlon": g["iDirectionIncrementInDegrees"],
                  "ni": g["Ni"], "nj": g["Nj"]}


def icon_eu_wind(session, now_dt):
    """(u, v, grid, run_dt) 700 hPa wind [m/s] of the newest ICON-EU run
    that is already online, at the model hour nearest to now."""
    latest = now_dt.replace(minute=0, second=0, microsecond=0)
    latest -= timedelta(hours=latest.hour % 3)
    last = None
    for back in range(0, 5):
        run = latest - timedelta(hours=3 * back)
        step = int(round((now_dt - run).total_seconds() / 3600.0))
        try:
            u, g = _icon_eu_field(session, run, step, "U", WIND_LEVEL)
            v, _ = _icon_eu_field(session, run, step, "V", WIND_LEVEL)
            return u, v, g, run
        except Exception as e:
            last = e
    raise RuntimeError(f"no usable ICON-EU run ({last})")


def wind_nodes(frame, u, v, g):
    """Wind sampled at a coarse node grid of the (pooled) OPERA frame and
    converted to pixel motion per minute (rows down, cols right)."""
    h, w = frame["dbz"].shape
    node = FLOW_NODE_PX
    ny, nx = h // node, w // node
    inv = Transformer.from_crs(frame["projdef"], "EPSG:4326", always_xy=True)
    fwd = Transformer.from_crs("EPSG:4326", frame["projdef"], always_xy=True)
    rr, cc = np.meshgrid((np.arange(ny) + 0.5) * node, (np.arange(nx) + 0.5) * node, indexing="ij")
    x = frame["ul_x"] + cc * frame["xscale"]
    y = frame["ul_y"] - rr * frame["yscale"]
    lon, lat = (np.asarray(a) for a in inv.transform(x, y))
    # Unit vectors east / north in grid axes (the LAEA grid is rotated against
    # true north depending on longitude).
    e = 0.01
    xe, ye = fwd.transform(lon + e, lat)
    xn, yn = fwd.transform(lon, lat + e)
    ex, ey = np.asarray(xe) - x, np.asarray(ye) - y
    n1 = np.hypot(ex, ey)
    ex, ey = ex / n1, ey / n1
    nx_, ny_ = np.asarray(xn) - x, np.asarray(yn) - y
    n2 = np.hypot(nx_, ny_)
    nx_, ny_ = nx_ / n2, ny_ / n2
    fj = np.clip(np.round((lat - g["lat0"]) / g["dlat"]).astype(int), 0, g["nj"] - 1)
    fi = np.clip(np.round((lon - g["lon0"]) / g["dlon"]).astype(int), 0, g["ni"] - 1)
    uu, vv = u[fj, fi] * WIND_FACTOR, v[fj, fi] * WIND_FACTOR
    dx = uu * ex + vv * nx_    # m/s along the grid's x axis (right)
    dy = uu * ey + vv * ny_    # m/s along the grid's y axis (up)
    return (-dy * 60.0 / frame["yscale"]).astype(np.float32), (dx * 60.0 / frame["xscale"]).astype(np.float32)


class Flow:
    """Motion field upsampled to the frame grid, with a bilinear advection."""

    def __init__(self, nodes, shape):
        vr, vc = nodes
        h, w = shape
        S = FLOW_NODE_PX
        ny, nx = vr.shape
        gy = (np.arange(h) - S / 2.0) / S
        gx = (np.arange(w) - S / 2.0) / S
        r0 = np.clip(np.floor(gy).astype(int), 0, ny - 2)
        c0 = np.clip(np.floor(gx).astype(int), 0, nx - 2)
        fr = np.clip(gy - r0, 0, 1).astype(np.float32)[:, None]
        fc = np.clip(gx - c0, 0, 1).astype(np.float32)[None, :]

        def up(a):
            top = a[r0][:, c0] * (1 - fc) + a[r0][:, c0 + 1] * fc
            bot = a[r0 + 1][:, c0] * (1 - fc) + a[r0 + 1][:, c0 + 1] * fc
            return (top * (1 - fr) + bot * fr).astype(np.float32)

        self.vr, self.vc = up(vr), up(vc)
        self.h, self.w = h, w
        self.rr = np.arange(h, dtype=np.float32)[:, None]
        self.cc = np.arange(w, dtype=np.float32)[None, :]

    def advect(self, frame, minutes):
        """The frame as it will look `minutes` later (semi-Lagrangian,
        single backward step with the constant flow)."""
        sr = self.rr - self.vr * minutes
        sc = self.cc - self.vc * minutes
        inside = (sr >= 0) & (sr <= self.h - 1) & (sc >= 0) & (sc <= self.w - 1)
        sr = np.clip(sr, 0, self.h - 1.001)
        sc = np.clip(sc, 0, self.w - 1.001)
        r0, c0 = sr.astype(np.int32), sc.astype(np.int32)
        fr, fc = sr - r0, sc - c0
        idx = r0 * self.w + c0

        def warp(grid):
            g = grid.ravel()
            v00, v10 = g.take(idx), g.take(idx + 1)
            v01, v11 = g.take(idx + self.w), g.take(idx + self.w + 1)
            return (v00 * (1 - fc) + v10 * fc) * (1 - fr) + (v01 * (1 - fc) + v11 * fc) * fr

        out = {k: v for k, v in frame.items() if k not in ("cover", "weight")}
        out["dbz"] = np.where(inside, warp(frame["dbz"]), np.float32(-30.0)).astype(np.float32)
        out["valid"] = (warp(frame["valid"]) * inside).astype(np.float32)
        return out


def build_forecaster(session, archive, newest, now_dt):
    """(Flow, pooled newest OPERA frame, ICON-EU run) for forecasting outside
    the DWD coverage, or raises if the wind is unavailable."""
    frame = pool(archive.fetch(newest), FLOW_POOL)
    u, v, g, run = icon_eu_wind(session, now_dt)
    return Flow(wind_nodes(frame, u, v, g), frame["dbz"].shape), frame, run
