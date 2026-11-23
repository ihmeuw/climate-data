"""Raking shapes with vertices a hair past ±180° must not wrap across the projected map.

CLIMATE-36 (16Nov2026): ``shapes_lsae_1578_a2.parquet`` has vertices at 180.0000000001°.
PROJ wraps those to the far edge of ESRI:54034, so a 0.1°-wide polygon in Fiji projects to a
band 40,075 km wide. Rasterized with ``MergeAlg.replace``, the band overwrote a neighbouring
unit's pixels (3,490 people moved from Bua to Cakaudrove) and the block filter picked up 33
blocks the shape does not touch.
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterra as rt
from affine import Affine
from shapely.geometry import box

from climate_data.aggregate.utils import (
    blocks_with_shapefile_intersections,
    build_location_masks,
)
from climate_data.data import PopulationModelData

BLOCK_CRS = "ESRI:54034"
PAST_180 = 180.0000000001  # as stored in shapes_lsae_1578_a2.parquet
LAT_MIN, LAT_MAX = -17.0, -16.0
NEIGHBOUR, THIN = 1, 2  # location ids
MAX_THIN_PIXELS = 150  # a 0.1° x 0.1° box holds about 11 x 11 one-km pixels

# side -> (geographic box of a block beside the line, a thin polygon reaching just past it)
SIDES = {
    "east": ((178.0, LAT_MIN, 179.99, LAT_MAX), (179.9, -16.6, PAST_180, -16.5)),
    "west": ((-179.99, LAT_MIN, -178.0, LAT_MAX), (-PAST_180, -16.6, -179.9, -16.5)),
}


def _shapes(
    rows: list[tuple[int, tuple[float, float, float, float]]],
) -> gpd.GeoDataFrame:
    ids = []
    geoms = []
    for loc_id, bounds in rows:
        ids.append(loc_id)
        geoms.append(box(*bounds))
    return gpd.GeoDataFrame({"location_id": ids, "geometry": geoms}, crs="EPSG:4326")


@pytest.mark.parametrize("side", list(SIDES))
def test_block_filter_ignores_vertices_past_180(
    monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    near_box, thin = SIDES[side]
    blocks = gpd.GeoDataFrame(
        {
            "block_key": ["B-near", "B-far"],
            "geometry": [box(*near_box), box(-10.0, LAT_MIN, -9.0, LAT_MAX)],
        },
        crs="EPSG:4326",
    ).to_crs(BLOCK_CRS)
    shapes = _shapes([(THIN, thin)])
    monkeypatch.setattr(
        PopulationModelData, "load_raking_shapes", lambda *a, **k: shapes
    )

    result = blocks_with_shapefile_intersections(
        "lsae_1578", PopulationModelData("unused-root"), blocks
    )

    assert result == {"B-near"}


def _template(block_box: tuple[float, float, float, float]) -> rt.RasterArray:
    """A 1 km ESRI:54034 raster covering the geographic box."""
    corners = gpd.GeoSeries([box(*block_box)], crs="EPSG:4326").to_crs(BLOCK_CRS)
    xmin, ymin, xmax, ymax = corners.total_bounds
    res = 1000.0
    n_cols = int((xmax - xmin) // res)
    n_rows = int((ymax - ymin) // res)
    return rt.RasterArray(
        np.ones((n_rows, n_cols), dtype=np.float32),
        transform=Affine(res, 0.0, xmin, 0.0, -res, ymax),
        crs=BLOCK_CRS,
        no_data_value=np.nan,
    )


def _pixel_at(template: rt.RasterArray, lon: float, lat: float) -> tuple[int, int]:
    pt = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs(
        BLOCK_CRS
    )
    col, row = ~template.transform * (pt.x.iloc[0], pt.y.iloc[0])
    return int(row), int(col)


@pytest.mark.parametrize("side", list(SIDES))
def test_masks_keep_the_neighbour_pixels(
    monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    block_box, thin = SIDES[side]
    lon_lo, _, lon_hi, _ = block_box
    mid = (lon_lo + lon_hi) / 2
    # NEIGHBOUR fills most of the block; THIN is the polygon at the line, listed second so
    # MergeAlg.replace would let a wrapped band overwrite NEIGHBOUR.
    neighbour = (min(lon_lo + 0.2, mid), -16.9, max(lon_hi - 0.5, mid), -16.1)
    template = _template(block_box)
    shapes = _shapes([(NEIGHBOUR, neighbour), (THIN, thin)])
    monkeypatch.setattr(PopulationModelData, "load_results", lambda *a, **k: template)
    monkeypatch.setattr(
        PopulationModelData, "load_raking_shapes", lambda *a, **k: shapes
    )

    _, bounds_map, location_mask = build_location_masks(
        "lsae_1578", "B-near", PopulationModelData("unused-root")
    )

    # A pixel inside NEIGHBOUR, in the latitude band the thin polygon spans.
    row, col = _pixel_at(template, mid, -16.55)
    assert location_mask[row, col] == NEIGHBOUR
    # The thin polygon keeps only pixels near the line.
    assert 0 < int((location_mask == THIN).sum()) < MAX_THIN_PIXELS
    # Its window is the polygon plus build_bounds_map's 10-pixel buffer, not the whole block.
    _, cols, _ = bounds_map[THIN]
    assert (cols.stop - cols.start) < template.shape[1] // 4
