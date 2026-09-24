"""
Parse metadata out of a BigEarthNet-S2 patch_id, confirmed against the archive
in docs/archive_layout.md:

    S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
    0   1      2                3     4    5      6  7

    2 = acquisition datetime (YYYYMMDDTHHMMSS)
    5 = MGRS tile code (54 unique values across the archive -- this is "tile"
        for the leave-tile-out split, NOT the per-acquisition folder name)
    6,7 = patch grid coordinates (gx, gy) within that tile acquisition
"""
import re

_PATCH_RE = re.compile(
    r"^(?P<sensor>S2[AB])_(?P<level>MSIL2A)_(?P<datetime>\d{8}T\d{6})_"
    r"(?P<baseline>N\d{4})_(?P<orbit>R\d{3})_(?P<tile>T\w{5})_"
    r"(?P<gx>\d+)_(?P<gy>\d+)$"
)


def parse_patch_id(patch_id: str) -> dict:
    m = _PATCH_RE.match(patch_id)
    if not m:
        raise ValueError(f"patch_id does not match expected BigEarthNet-S2 format: {patch_id!r}")
    dt = m.group("datetime")
    return {
        "tile": m.group("tile"),
        "gx": int(m.group("gx")),
        "gy": int(m.group("gy")),
        "year": int(dt[0:4]),
        "month": int(dt[4:6]),
    }


def tile_folder_name(patch_id: str) -> str:
    """The per-acquisition directory name a patch lives under in the archive
    (patch_id with the trailing _gx_gy stripped)."""
    parts = patch_id.split("_")
    return "_".join(parts[:-2])


if __name__ == "__main__":
    example = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
    print(parse_patch_id(example))
    assert tile_folder_name(example) == "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP"
    print("OK")
