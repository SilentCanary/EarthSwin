"""The real-data path (GEE-format tiles -> chips) exercised on tiles fabricated from the synthetic region."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")

from rasterio import features  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from pmgst.data import chips as C  # noqa: E402
from pmgst.data.formats import IDX_ELEV, load_chip, write_index  # noqa: E402
from pmgst.data.synth import make_region  # noqa: E402
from pmgst.data.terrain import area_resample  # noqa: E402

X0, Y1 = 700_000, 3_560_000


def _write(path, arr, res, dtype, x0=X0, y1=Y1):
    arr = np.asarray(arr)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0], dtype=dtype,
                       crs="EPSG:32643", transform=from_origin(x0, y1, res, res)) as dst:
        dst.write(arr.astype(dtype))


@pytest.fixture(scope="module")
def tile(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    R = make_region(5, 8, 8, 64, n_dates=15)                    # 512x512 px tile = core 4x4 chips + 2-chip buffer
    flag = lambda v: v.astype(np.int16)
    def s2(i):
        a = np.concatenate([(R["s2"][i] * 10000).astype(np.int16), R["s2_valid"][i].astype(np.int16)], 0)
        return a
    _write(raw / f"s2pre_x{X0 + 1280}_y{Y1 - 1280}.tif", s2(0), 10, "int16")
    _write(raw / f"s2post_x{X0 + 1280}_y{Y1 - 1280}.tif", s2(1), 10, "int16")
    s1 = np.concatenate([(R["s1"][:, 0] * 100).round(), (R["s1"][:, 1] * 100).round()], 0)
    _write(raw / f"s1_x{X0 + 1280}_y{Y1 - 1280}.tif", s1, 10, "int16")
    dem40 = area_resample(R["dem"][0:1], 128)                    # 40 m DEM
    _write(raw / f"dem_x{X0 + 1280}_y{Y1 - 1280}.tif", dem40, 40, "float32")
    _write(raw / f"rain_x{X0 + 1280}_y{Y1 - 1280}.tif", np.tile(R["rain_daily"][:, None, None], (1, 1, 1)), 5120, "float32")
    static = np.stack([np.where(R["water"] > 0, 80, 10), R["lith"]])
    _write(raw / f"static_x{X0 + 1280}_y{Y1 - 1280}.tif", static, 10, "int16")
    tf = from_origin(X0, Y1, 10, 10)
    lbl = (R["label"] == 1).astype(np.uint8)
    polys = [rasterio.features.shape for rasterio.features.shape in []]
    from shapely.geometry import shape
    geoms = [shape(g) for g, v in features.shapes(lbl, mask=lbl > 0, transform=tf)]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:32643")
    return raw, gdf, R


def test_tile_to_chips(tile, tmp_path):
    raw, gdf, R = tile
    assert C.find_tiles(raw) == [(X0 + 1280, Y1 - 1280)]
    out = tmp_path / "chips"; out.mkdir()
    rows = C.build_tile(raw, X0 + 1280, Y1 - 1280, gdf, None, None, 64, 32, 3, 1280.0, 1.0, np.random.default_rng(0), out, "t")
    assert len(rows) == 16                                       # 4x4 core chips
    c = load_chip(out / f"{rows[0]['chip_id']}.npz")
    assert c["s2"].shape == (2, 6, 64, 64) and c["s1"].shape == (15, 2, 64, 64) and c["dem"].shape == (8, 64, 64)
    assert c["ctx_s2"].shape == (2, 2, 6, 32, 32) and c["ctx_dem"].shape == (2, 8, 32, 32) and c["label"].shape == (64, 64)
    assert c["s1_dt"].shape == (15,) and set(np.unique(c["label"])) <= {0, 1, 255}
    # S1 zeros must be invalid (audit C2): validity mask is 0 exactly where the export had no data
    assert (c["s1"][c["s1_valid"].astype(bool).repeat(2, 1) == 0] == 0).all()
    # elevation survives 40 m -> 10 m resampling (to within local relief) and chips sit where we expect
    r0 = rows[0]
    gy, gx = 0, 0
    ref = R["dem"][IDX_ELEV][128:192, 128:192].mean()
    assert abs(c["dem"][IDX_ELEV].mean() - ref) < 25
    assert abs(r0["x"] - (X0 + 128 * 10 + 320)) < 1 and abs(r0["y"] - (Y1 - 128 * 10 - 320)) < 1
    npos = sum(r["n_pos_px"] for r in rows)
    ref_pos = int((R["label"][128:384, 128:384] == 1).sum())
    assert 0.9 * ref_pos <= npos <= 1.05 * ref_pos              # rasterised polygons reproduce the label mask
    write_index(out.parent / "index.csv", rows)


def test_label_extent_masks_unmapped(tile, tmp_path):
    raw, gdf, R = tile
    from shapely.geometry import box
    ext = gpd.GeoDataFrame(geometry=[box(X0, Y1 - 5120, X0 + 2560, Y1)], crs="EPSG:32643")   # left half only
    out = tmp_path / "chips"; out.mkdir()
    rows = C.build_tile(raw, X0 + 1280, Y1 - 1280, gdf, ext, None, 64, 32, 3, 1280.0, 1.0, np.random.default_rng(0), out, "t")
    assert 0 < len(rows) < 16                                    # chips fully outside the inventory are dropped
    lab = np.stack([load_chip(out / f"{r['chip_id']}.npz")["label"] for r in rows])
    assert (lab == 255).any()
