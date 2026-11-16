"""Tests for `climate_data.utils.to_raster` (CLIMATE-49).

The climate grids store cell-centre coordinates. A raster transform needs the
outer corner of the grid, half a cell beyond the first and last centres.
"""

import numpy as np
import pytest
import xarray as xr

from climate_data.utils import to_raster

LON_CENTRES = np.array([10.05, 10.15, 10.25, 10.35])
LAT_CENTRES = np.array([0.05, 0.15, 0.25])


def _grid(*, lat_descending: bool) -> xr.DataArray:
    """A 3 x 4 grid of 0.1 degree cells; every cell holds a distinct value."""
    lat = LAT_CENTRES[::-1] if lat_descending else LAT_CENTRES
    data = np.arange(lat.size * LON_CENTRES.size, dtype="float64").reshape(
        lat.size, LON_CENTRES.size
    )
    return xr.DataArray(
        data, coords={"lat": lat, "lon": LON_CENTRES}, dims=["lat", "lon"]
    )


@pytest.mark.parametrize("lat_descending", [True, False])
def test_pixel_centres_match_the_grid_centres(lat_descending: bool) -> None:
    # Each raster pixel's centre must be the coordinate of the value it holds.
    # The defect put the grid's first centre on the raster's corner, so every
    # pixel centre sat half a cell east and north of its value's coordinate.
    da = _grid(lat_descending=lat_descending)
    raster = to_raster(da, no_data_value=np.nan)
    values = raster.to_numpy()
    n_rows, n_cols = values.shape

    for row in range(n_rows):
        for col in range(n_cols):
            x, y = raster.transform * (col + 0.5, row + 0.5)
            expected = da.sel(lon=x, lat=y, method="nearest")
            assert float(expected["lon"]) == pytest.approx(x, abs=1e-9)
            assert float(expected["lat"]) == pytest.approx(y, abs=1e-9)
            assert values[row, col] == float(expected)


@pytest.mark.parametrize("lat_descending", [True, False])
def test_raster_bounds_are_the_outer_cell_edges(lat_descending: bool) -> None:
    raster = to_raster(_grid(lat_descending=lat_descending), no_data_value=np.nan)
    n_rows, n_cols = raster.to_numpy().shape

    xs = []
    ys = []
    for corner in [(0, 0), (n_cols, n_rows)]:
        x, y = raster.transform * corner
        xs.append(x)
        ys.append(y)

    assert min(xs) == pytest.approx(10.0)
    assert max(xs) == pytest.approx(10.4)
    assert min(ys) == pytest.approx(0.0)
    assert max(ys) == pytest.approx(0.3)
