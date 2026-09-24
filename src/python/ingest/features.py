"""
SPRINT.md Sec 1.4 -- ~110 cheap features per patch, pure numpy, no I/O.

Input: dict of band_name -> 2D uint16 numpy array (raw pixel values, native
resolution). Bands at 20m/60m are upsampled to 120x120 nearest-neighbour here.

Output: dict of feature_name -> float32 value, plus the frozen f_000..f_109
ordering used by the Scala job.
"""
import numpy as np
from scipy.ndimage import zoom, uniform_filter, sobel

BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
TARGET_SIZE = 120
EPS = 1e-6


def _safe_div(num, den):
    den = np.where(np.abs(den) < EPS, EPS, den)
    return num / den


def _upsample(band: np.ndarray) -> np.ndarray:
    if band.shape == (TARGET_SIZE, TARGET_SIZE):
        return band
    factor = TARGET_SIZE / band.shape[0]
    return zoom(band, factor, order=0)  # nearest-neighbour


def load_bands(band_arrays: dict) -> dict:
    """band_arrays: {band_name: raw 2D array at native resolution} -> all at 120x120 float32."""
    return {b: _upsample(band_arrays[b]).astype(np.float32) for b in BAND_ORDER}


def per_band_stats(bands: dict) -> dict:
    """(a) 12 bands x 6 stats = 72 features."""
    feats = {}
    for b in BAND_ORDER:
        arr = bands[b]
        feats[f"{b}_mean"] = float(np.mean(arr))
        feats[f"{b}_std"] = float(np.std(arr))
        p10, p50, p90 = np.percentile(arr, [10, 50, 90])
        feats[f"{b}_p10"] = float(p10)
        feats[f"{b}_p50"] = float(p50)
        feats[f"{b}_p90"] = float(p90)
        feats[f"{b}_iqr"] = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    return feats


def spectral_indices(bands: dict) -> dict:
    """(b) 13 indices x mean/std = 26 features."""
    B01, B02, B03, B04 = bands["B01"], bands["B02"], bands["B03"], bands["B04"]
    B05, B06, B07, B08 = bands["B05"], bands["B06"], bands["B07"], bands["B08"]
    B11, B12 = bands["B11"], bands["B12"]

    indices = {
        "NDVI": _safe_div(B08 - B04, B08 + B04),
        "EVI": 2.5 * _safe_div(B08 - B04, B08 + 6 * B04 - 7.5 * B02 + 1),
        "SAVI": 1.5 * _safe_div(B08 - B04, B08 + B04 + 0.5),
        "NDWI": _safe_div(B03 - B08, B03 + B08),
        "MNDWI": _safe_div(B03 - B11, B03 + B11),
        "NDMI": _safe_div(B08 - B11, B08 + B11),
        "NDBI": _safe_div(B11 - B08, B11 + B08),
        "BSI": _safe_div((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02)),
        "NBR": _safe_div(B08 - B12, B08 + B12),
        "NDRE1": _safe_div(B08 - B05, B08 + B05),
        "NDRE2": _safe_div(B08 - B06, B08 + B06),
        "NDRE3": _safe_div(B08 - B07, B08 + B07),
        "CIre": _safe_div(B07, np.where(np.abs(B05) < EPS, EPS, B05)) - 1,
    }

    feats = {}
    for name, arr in indices.items():
        feats[f"{name}_mean"] = float(np.mean(arr))
        feats[f"{name}_std"] = float(np.std(arr))
    return feats


def texture_features(bands: dict) -> dict:
    """(c) cheap texture -- 12 features: std/p90 of 3x3 var, 7x7 var, sobel mag, for B08 and B04."""
    feats = {}
    for band_name in ("B08", "B04"):
        arr = bands[band_name]
        mean3 = uniform_filter(arr, size=3)
        mean_sq3 = uniform_filter(arr * arr, size=3)
        var3 = np.clip(mean_sq3 - mean3 * mean3, 0, None)

        mean7 = uniform_filter(arr, size=7)
        mean_sq7 = uniform_filter(arr * arr, size=7)
        var7 = np.clip(mean_sq7 - mean7 * mean7, 0, None)

        gx = sobel(arr, axis=0)
        gy = sobel(arr, axis=1)
        grad_mag = np.hypot(gx, gy)

        for stat_name, stat_arr in (("var3x3", var3), ("var7x7", var7), ("sobel", grad_mag)):
            feats[f"{band_name}_{stat_name}_std"] = float(np.std(stat_arr))
            feats[f"{band_name}_{stat_name}_p90"] = float(np.percentile(stat_arr, 90))
    return feats


def extract_features(band_arrays: dict) -> dict:
    """band_arrays: {band_name: raw 2D uint16 array at native resolution}. Returns f_000..f_109."""
    bands = load_bands(band_arrays)
    all_feats = {}
    all_feats.update(per_band_stats(bands))
    all_feats.update(spectral_indices(bands))
    all_feats.update(texture_features(bands))

    for v in all_feats.values():
        if not np.isfinite(v):
            raise ValueError("non-finite feature produced -- check band alignment / division guards")

    names_in_order = sorted(all_feats.keys())
    return {f"f_{i:03d}": np.float32(all_feats[name]) for i, name in enumerate(names_in_order)}, names_in_order


def feature_dictionary() -> list:
    """Returns [(f_i, human_name), ...] in the frozen order, for docs/feature_dictionary.md."""
    native = _native_sizes()
    rng = np.random.default_rng(0)
    fake_bands = {b: rng.integers(500, 4000, size=(sz, sz), dtype=np.uint16) for b, sz in native.items()}
    _, names_in_order = extract_features(fake_bands)
    return [(f"f_{i:03d}", name) for i, name in enumerate(names_in_order)]


def _native_sizes() -> dict:
    sizes_10m = {"B02", "B03", "B04", "B08"}
    sizes_60m = {"B01", "B09"}
    return {b: (120 if b in sizes_10m else 20 if b in sizes_60m else 60) for b in BAND_ORDER}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    native = _native_sizes()
    fake_bands = {b: rng.integers(500, 4000, size=(sz, sz), dtype=np.uint16) for b, sz in native.items()}
    feats, names = extract_features(fake_bands)
    print(f"Produced {len(feats)} features (expected 110): f_000..f_{len(feats)-1:03d}")
    assert len(feats) == 110, f"expected 110 features, got {len(feats)}"
    for k in list(feats.keys())[:5]:
        print(k, "->", names[int(k.split('_')[1])], "=", feats[k])
    print("OK")
