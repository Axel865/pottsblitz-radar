"""
PottsBlitz — precipitation areas + movement direction for the home-screen
radar tile ("nowcast.json").

The app shows a small white arrow on the precipitation area that is nearest
to the user's location, pointing the way that area is moving. The browser
cannot work this out itself (it only gets map-tile PNGs, and DWD's grid is in
a polar-stereographic projection), so it is computed here from the same DWD
composite frames the tiles are rendered from:

  * the "now" frame is turned into a coarse lat/lon raster of precipitation
    areas (connected blobs >= MIN_CELLS cells, each with an id), and
  * for every area the movement is measured by phase correlation between the
    current frame and DWD's own +30 min nowcast frame (the pattern shift in
    km, converted to a compass bearing and a speed). The nowcast is what the
    animation shows, so the arrow always agrees with it. (Measuring from the
    past 30 minutes instead was tried and fails: radar range edges and
    ground clutter are static and dominate the correlation.)

Output (JSON, small):
  grid   : lat0/lon0 (south-west corner), dlat/dlon, nlat/nlon (row 0 = south)
  rle    : [id, count, id, count, ...] over the raster, row-major (id 0 = dry)
  areas  : {"<id>": {"b": bearing 0-359 (direction it moves TO), "v": km/h}}
"""
import math

import numpy as np
from pyproj import Transformer

# Raster covering the whole DWD composite (Germany and surroundings).
LAT0, LAT1, DLAT = 45.5, 56.0, 0.04
LON0, LON1, DLON = 1.5, 19.0, 0.05

MIN_DBZ = 12.0        # same threshold the tiles use (~0.2 mm/h)
BLUR_PX = 5           # box size (km) for the "is this cell wet" fraction
WET_FRACTION = 0.35   # share of wet 1 km pixels around a cell to count it as wet
MIN_CELLS = 15        # ignore specks smaller than ~15 cells (~190 km2)
MAX_AREAS = 250       # ids are stored as small ints
HORIZON_MINUTES = 30  # frame pair used for the motion measurement: now -> +30 min
SLOW_KMH = 8          # slower areas are measured again over +60 min: twice the
                      # displacement, so twice the precision for slow movers
MIN_PEAK = 0.10       # phase-correlation peak needed to trust a measurement
MAX_SHIFT_PX = 45     # 45 km per 30 min = 90 km/h upper bound


def _prep(frame):
    """Wet-intensity field in dBZ-above-threshold, 0 where dry/invalid."""
    f = np.clip(frame["dbz"] - (MIN_DBZ - 5.0), 0, 40).astype(np.float32)
    f[frame["valid"] <= 0] = 0
    return f


def _box_fraction(mask, k):
    """Fraction of True pixels in a k x k box around every pixel."""
    m = mask.astype(np.float32)
    c = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = m.shape
    r = k // 2
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    s = c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return s / area


def _label(wet):
    """8-connected component labelling of a small boolean raster."""
    h, w = wet.shape
    labels = np.zeros((h, w), dtype=np.int32)
    flat_wet = wet.ravel()
    flat_lab = labels.ravel()
    n = 0
    sizes = [0]
    for start in np.flatnonzero(flat_wet):
        if flat_lab[start]:
            continue
        n += 1
        flat_lab[start] = n
        stack = [start]
        size = 0
        while stack:
            p = stack.pop()
            size += 1
            y, x = divmod(p, w)
            for dy in (-1, 0, 1):
                yy = y + dy
                if yy < 0 or yy >= h:
                    continue
                for dx in (-1, 0, 1):
                    xx = x + dx
                    if xx < 0 or xx >= w:
                        continue
                    q = yy * w + xx
                    if flat_wet[q] and not flat_lab[q]:
                        flat_lab[q] = n
                        stack.append(q)
        sizes.append(size)
    return labels, sizes


def _phase_shift(a, b):
    """Shift (drow, dcol) of pattern b relative to a, plus the peak strength.
    Positive = b is a moved down / right."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fa = np.fft.rfft2(a * win)
    fb = np.fft.rfft2(b * win)
    cross = np.conj(fa) * fb
    cross /= np.maximum(np.abs(cross), 1e-9)
    r = np.fft.irfft2(cross, s=(h, w))
    # Only plausible shifts: search a (2*MAX+1)^2 neighbourhood of zero shift.
    m = MAX_SHIFT_PX
    rr = np.roll(np.roll(r, m, axis=0), m, axis=1)[: 2 * m + 1, : 2 * m + 1]
    iy, ix = np.unravel_index(np.argmax(rr), rr.shape)
    peak = float(rr[iy, ix])

    def sub(vm, v0, vp):
        d = vm - 2 * v0 + vp
        return 0.0 if abs(d) < 1e-9 else 0.5 * (vm - vp) / d

    dy = dx = 0.0
    if 0 < iy < 2 * m:
        dy = sub(rr[iy - 1, ix], rr[iy, ix], rr[iy + 1, ix])
    if 0 < ix < 2 * m:
        dx = sub(rr[iy, ix - 1], rr[iy, ix], rr[iy, ix + 1])
    return (iy - m + dy, ix - m + dx), peak


def _measure(fa, fb, crow, ccol, half):
    """Movement of the pattern around (crow, ccol), window +-half px."""
    h, w = fa.shape
    r0, r1 = max(0, int(crow) - half), min(h, int(crow) + half)
    c0, c1 = max(0, int(ccol) - half), min(w, int(ccol) + half)
    if r1 - r0 < 48 or c1 - c0 < 48:
        return None
    a, b = fa[r0:r1, c0:c1], fb[r0:r1, c0:c1]
    if (a > 0).mean() < 0.03 or (b > 0).mean() < 0.03:
        return None
    (drow, dcol), peak = _phase_shift(a, b)
    if peak < MIN_PEAK:
        return None
    return drow, dcol, peak, (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _bearing_speed(frame, drow, dcol, crow, ccol, minutes):
    """Pixel shift at a grid position -> compass bearing (deg, direction of
    travel) and speed (km/h), via real lat/lon (DWD's grid is rotated against
    true north by a few degrees)."""
    inv = Transformer.from_crs(frame["projdef"], "EPSG:4326", always_xy=True)
    x0 = frame["ul_x"] + ccol * frame["xscale"]
    y0 = frame["ul_y"] - crow * frame["yscale"]
    x1 = x0 + dcol * frame["xscale"]
    y1 = y0 - drow * frame["yscale"]
    lon0, lat0 = inv.transform(x0, y0)
    lon1, lat1 = inv.transform(x1, y1)
    p0, p1 = math.radians(lat0), math.radians(lat1)
    dl = math.radians(lon1 - lon0)
    yb = math.sin(dl) * math.cos(p1)
    xb = math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dl)
    bearing = (math.degrees(math.atan2(yb, xb)) + 360.0) % 360.0
    dist_km = math.hypot(dcol * frame["xscale"], drow * frame["yscale"]) / 1000.0
    return bearing, dist_km * 60.0 / minutes


def build_nowcast(frame_now, frame_future, now_dt, generated_at, frame_future60=None):
    """Returns the nowcast.json dict, or None if there is no usable
    precipitation."""
    lats = LAT0 + DLAT * (np.arange(int(round((LAT1 - LAT0) / DLAT))) + 0.5)
    lons = LON0 + DLON * (np.arange(int(round((LON1 - LON0) / DLON))) + 0.5)
    nlat, nlon = len(lats), len(lons)

    fwd = Transformer.from_crs("EPSG:4326", frame_now["projdef"], always_xy=True)
    lon_g, lat_g = np.meshgrid(lons, lats)
    gx, gy = fwd.transform(lon_g.ravel(), lat_g.ravel())
    col = ((np.asarray(gx) - frame_now["ul_x"]) / frame_now["xscale"]).reshape(nlat, nlon)
    row = ((frame_now["ul_y"] - np.asarray(gy)) / frame_now["yscale"]).reshape(nlat, nlon)
    ysize, xsize = frame_now["dbz"].shape
    ci, ri = np.round(col).astype(int), np.round(row).astype(int)
    inside = (ci >= 0) & (ci < xsize) & (ri >= 0) & (ri < ysize)
    ci, ri = np.clip(ci, 0, xsize - 1), np.clip(ri, 0, ysize - 1)

    wet_px = (frame_now["dbz"] >= MIN_DBZ) & (frame_now["valid"] > 0)
    frac = _box_fraction(wet_px, BLUR_PX)
    wet = (frac[ri, ci] >= WET_FRACTION) & inside
    if not wet.any():
        return None

    labels, sizes = _label(wet)
    order = [i for i in np.argsort(sizes)[::-1] if i > 0 and sizes[i] >= MIN_CELLS][:MAX_AREAS]
    if not order:
        return None
    remap = np.zeros(len(sizes), dtype=np.int32)
    for new_id, old in enumerate(order, start=1):
        remap[old] = new_id
    ids = remap[labels]

    fa, fb = _prep(frame_now), _prep(frame_future)
    fc = _prep(frame_future60) if frame_future60 is not None else None
    minutes = HORIZON_MINUTES
    areas, found = {}, []
    for new_id in range(1, len(order) + 1):
        sel = ids == new_id
        crow, ccol = row[sel].mean(), col[sel].mean()
        res = None
        for half in (80, 160):
            res = _measure(fa, fb, crow, ccol, half)
            if res:
                half_used_ = half
                break
        if res:
            drow, dcol, peak, wr, wc = res
            b, v = _bearing_speed(frame_now, drow, dcol, wr, wc, minutes)
            if v < SLOW_KMH and fc is not None:
                res60 = _measure(fa, fc, crow, ccol, half_used_)
                if res60:
                    drow, dcol, peak, wr, wc = res60
                    b, v = _bearing_speed(frame_now, drow, dcol, wr, wc, 2 * minutes)
            areas[str(new_id)] = {"b": int(round(b)) % 360, "v": round(v, 1)}
            found.append((b, v))
    # Areas where no reliable measurement was possible get the average motion
    # of the others (weather systems in one region move alike).
    if found:
        sx = sum(math.sin(math.radians(b)) for b, _ in found)
        sy = sum(math.cos(math.radians(b)) for b, _ in found)
        fb_b = int(round(math.degrees(math.atan2(sx, sy)) + 360.0)) % 360
        fb_v = round(sum(v for _, v in found) / len(found), 1)
        for new_id in range(1, len(order) + 1):
            areas.setdefault(str(new_id), {"b": fb_b, "v": fb_v, "est": 1})
    else:
        return None

    flat = ids.ravel()
    change = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], change))
    counts = np.diff(np.concatenate((starts, [flat.size])))
    rle = []
    for s, c in zip(starts, counts):
        rle.extend([int(flat[s]), int(c)])

    return {
        "generated_at": generated_at,
        "now": now_dt.isoformat(),
        "grid": {"lat0": LAT0, "lon0": LON0, "dlat": DLAT, "dlon": DLON, "nlat": nlat, "nlon": nlon},
        "rle": rle,
        "areas": areas,
    }
