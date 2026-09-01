import geopandas as gpd
import pytest
from shapely.geometry import box

from climate_data.aggregate.utils import blocks_with_shapefile_intersections
from climate_data.data import PopulationModelData


def _blocks() -> gpd.GeoDataFrame:
    """Two blocks: `B-in` near the origin, `B-out` far away."""
    return gpd.GeoDataFrame(
        {
            "block_key": ["B-in", "B-out"],
            "geometry": [box(0, 0, 1, 1), box(10, 10, 11, 11)],
        },
        crs="EPSG:4326",
    )


def test_keeps_only_intersecting_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    shapes = gpd.GeoDataFrame(
        {"location_id": [1], "geometry": [box(0.5, 0.5, 5.0, 5.0)]}, crs="EPSG:4326"
    )
    monkeypatch.setattr(
        PopulationModelData, "load_raking_shapes", lambda *a, **k: shapes
    )

    result = blocks_with_shapefile_intersections(
        "gbd_2021", PopulationModelData("unused-root"), _blocks()
    )

    assert result == {"B-in"}


def test_reprojects_mismatched_crs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same overlapping shape, but in Web Mercator: the helper must reproject the
    # shapes to the blocks' CRS before the intersects test.
    shapes = gpd.GeoDataFrame(
        {"location_id": [1], "geometry": [box(0.5, 0.5, 5.0, 5.0)]}, crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    monkeypatch.setattr(
        PopulationModelData, "load_raking_shapes", lambda *a, **k: shapes
    )

    result = blocks_with_shapefile_intersections(
        "gbd_2021", PopulationModelData("unused-root"), _blocks()
    )

    assert result == {"B-in"}


def test_raises_when_nothing_intersects(monkeypatch: pytest.MonkeyPatch) -> None:
    # A shape that no block touches: the helper must fail loudly rather than
    # silently returning an empty set (which would skip every block).
    shapes = gpd.GeoDataFrame(
        {"location_id": [1], "geometry": [box(100, 100, 101, 101)]}, crs="EPSG:4326"
    )
    monkeypatch.setattr(
        PopulationModelData, "load_raking_shapes", lambda *a, **k: shapes
    )

    with pytest.raises(ValueError, match="No blocks intersect"):
        blocks_with_shapefile_intersections(
            "gbd_2021", PopulationModelData("unused-root"), _blocks()
        )
