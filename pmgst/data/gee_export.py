"""Google Earth Engine export: one set of GeoTIFFs per 10.24 km tile (+2.56 km context buffer) on a fixed UTM grid.

  pip install earthengine-api pyproj
  earthengine authenticate
  python -m pmgst.data.gee_export --project YOUR_GEE_PROJECT --bbox 76.5 31.4 77.9 32.4 --year 2023 --folder pmgst_2023 --dry-run
  python -m pmgst.data.gee_export --project YOUR_GEE_PROJECT --bbox 76.5 31.4 77.9 32.4 --year 2023 --folder pmgst_2023 --max-tiles 1
  python -m pmgst.data.gee_export --project YOUR_GEE_PROJECT --extent labels/extent.gpkg --folder pmgst_2023     # once you have labels
Size: ~220 MB per tile (so the Mandi+Kullu bbox = 168 tiles = ~37 GB; the labelled area alone is ~30-40 tiles). Use --extent.

Per tile  (name = <kind>_x<X0>_y<Y1>, X0/Y1 = top-left corner in EPSG:32643):
  s2pre / s2post  Int16 [7]  B2,B3,B4,B8,B11,B12 (reflectance*10000, 0 = masked) + SCL-derived clear flag (1/0)
  s1              Int16 [30] VV_00..VV_14, VH_00..VH_14: median dB*100 of 12-day bins from 1 May (0 = no acquisition)
  dem             Float32 [1] Copernicus GLO-30 elevation at 30 m
  rain            Float32 [60] CHIRPS daily mm for the 60 days before the post window ends (5.12 km grid)
  static          Int16 [2]  ESA WorldCover class, OpenLandMap soil-texture class
Dataset IDs follow docs/04 §2 - if Earth Engine reports a missing ID, check the catalogue (names change between versions).
This script is not exercised by the unit tests (needs an Earth Engine account); use --dry-run to see the plan first.
"""
from __future__ import annotations

import argparse
import math

TILE_CORE = 10_240          # 8 chips of 128 px at 10 m
BUFFER = 2_560              # context crops of the largest scale reach 2.56 km from a chip centre
RES = 10
BIN_DAYS, N_BINS = 12, 15
CRS = "EPSG:32643"
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


def tiles_for_bbox(bbox, crs=CRS):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = zip(*[tr.transform(lon, lat) for lon, lat in ((bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]))])
    x0, x1 = math.floor(min(xs) / TILE_CORE) * TILE_CORE, math.ceil(max(xs) / TILE_CORE) * TILE_CORE
    y0, y1 = math.floor(min(ys) / TILE_CORE) * TILE_CORE, math.ceil(max(ys) / TILE_CORE) * TILE_CORE
    return [(x, y + TILE_CORE) for x in range(x0, x1, TILE_CORE) for y in range(y0, y1, TILE_CORE)]   # (X0, Y1) of each core


def tiles_for_extent(extent_path, crs=CRS):
    """Tiles whose core touches the polygon(s) in extent_path (the area the label inventory covers).
    Returns (tiles, bbox_lonlat). Needs geopandas + shapely."""
    import geopandas as gpd
    from shapely.geometry import box
    g = gpd.read_file(extent_path)
    if g.crs is None:
        raise SystemExit(f"{extent_path} has no CRS")
    lon0, lat0, lon1, lat1 = g.to_crs("EPSG:4326").total_bounds
    union = g.to_crs(crs).unary_union
    tiles = [(x, y1) for (x, y1) in tiles_for_bbox((lon0, lat0, lon1, lat1), crs)
             if box(x, y1 - TILE_CORE, x + TILE_CORE, y1).intersects(union)]
    return tiles, (lon0, lat0, lon1, lat1)


def order_by_centre(tiles):
    """Centre-most tiles first, so --max-tiles N is a sensible pilot (not a corner of the bbox)."""
    cx = sum(x for x, _ in tiles) / len(tiles); cy = sum(y for _, y in tiles) / len(tiles)
    return sorted(tiles, key=lambda t: (t[0] - cx) ** 2 + (t[1] - cy) ** 2)


def build_images(ee, geom, year, pre=("05-01", "06-30"), post=("09-15", "10-31"), orbit="ASCENDING"):
    d = lambda md: f"{year}-{md}"

    def s2_composite(a, b):
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(a, b)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))

        def mask(img):
            scl = img.select("SCL")
            clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(0))
            return img.select(S2_BANDS).updateMask(clear)
        comp = col.map(mask).median()
        refl = comp.select(S2_BANDS).unmask(0)                                    # S2_SR_HARMONIZED is already reflectance*10000
        flag = comp.select("B8").mask().rename("clear")                           # 1 where a valid composite exists
        return refl.addBands(flag.unmask(0)).toInt16()

    s1c = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("orbitProperties_pass", orbit))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select(["VV", "VH"]))
    start = ee.Date(d("05-01"))
    vv, vh = [], []
    for i in range(N_BINS):
        a, b = start.advance(BIN_DAYS * i, "day"), start.advance(BIN_DAYS * (i + 1), "day")
        col = s1c.filterDate(a, b)
        med = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), ee.Image.constant([0, 0]).rename(["VV", "VH"])))
        vv.append(med.select("VV").multiply(100).unmask(0).toInt16().rename(f"VV_{i:02d}"))
        vh.append(med.select("VH").multiply(100).unmask(0).toInt16().rename(f"VH_{i:02d}"))
    s1 = ee.Image.cat(vv + vh)

    dem = (ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().resample("bilinear")).toFloat()

    end = ee.Date(d(post[1]))
    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(end.advance(-60, "day"), end).select("precipitation")
            .toBands().unmask(0).toFloat())

    static = (ee.Image("ESA/WorldCover/v200").select("Map").rename("worldcover")
              .addBands(ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0").rename("soil"))
              .unmask(0).toInt16())
    return dict(s2pre=s2_composite(d(pre[0]), d(pre[1])), s2post=s2_composite(d(post[0]), d(post[1])),
                s1=s1, dem=dem, rain=rain, static=static)


SPEC = {  # kind -> pixel size (m)
    "s2pre": RES, "s2post": RES, "s1": RES, "dem": 30, "rain": 5120, "static": RES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Earth Engine cloud project id")
    ap.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--extent", default=None, help="polygon file of the labelled area: export ONLY tiles that touch it (saves ~75%%)")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--folder", default="pmgst_2023", help="Google Drive folder for the exports")
    ap.add_argument("--orbit", default="ASCENDING", choices=["ASCENDING", "DESCENDING"])
    ap.add_argument("--max-tiles", type=int, default=0, help="debug: only the first N tiles")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.extent:
        tiles, _ = tiles_for_extent(a.extent)
    elif a.bbox:
        tiles = tiles_for_bbox(a.bbox)
    else:
        raise SystemExit("give --extent (preferred) or --bbox")
    if not tiles:
        raise SystemExit("no tiles selected")
    tiles = order_by_centre(tiles)
    if a.max_tiles:
        tiles = tiles[: a.max_tiles]
    side = TILE_CORE + 2 * BUFFER
    side_px = side // RES
    mb = (2 * 7 * 2 + 30 * 2 + 2 * 2) * side_px * side_px / 1e6                      # s2 pre+post, s1, static (rain, dem are small)
    print(f"{len(tiles)} tiles of {TILE_CORE / 1000:.2f} km (+{BUFFER / 1000:.2f} km buffer) -> {len(tiles) * len(SPEC)} export tasks, "
          f"about {len(tiles) * mb / 1000:.1f} GB in Drive")
    if a.dry_run:
        print("first tiles (X0, Y1):", tiles[:5]); return
    import ee
    ee.Initialize(project=a.project)
    for (x0, y1) in tiles:
        X0, Y1 = x0 - BUFFER, y1 + BUFFER
        geom = ee.Geometry.Rectangle([X0, Y1 - side, X0 + side, Y1], proj=CRS, geodesic=False)
        imgs = build_images(ee, geom, a.year, orbit=a.orbit)
        for kind, res in SPEC.items():
            name = f"{kind}_x{x0}_y{y1}"
            ee.batch.Export.image.toDrive(
                image=imgs[kind], description=name, folder=a.folder, fileNamePrefix=name, region=geom,
                crs=CRS, crsTransform=[res, 0, X0, 0, -res, Y1], fileFormat="GeoTIFF", maxPixels=int(1e10)).start()
    print("submitted. Monitor at https://code.earthengine.google.com/tasks ; download the Drive folder to data/raw/")


if __name__ == "__main__":
    main()
