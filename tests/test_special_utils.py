"""Tests for `special.utils.get_temperature_coordinates`: which climate cell a pixel is in.

Found 10Nov2026: the daily files store latitude descending, and the cell lookup built its bin
edges as if latitude ascended, so about 99% of population pixels were given the climate cell one
row (0.1 degree) north. It also used pixel corners instead of centres. A pixel must get the cell
that contains its centre, whichever way the climate latitude runs.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

from climate_data.special import utils

# A 0.1 degree climate grid: 4 rows, 5 columns.
LAT_DESCENDING = np.array([40.35, 40.25, 40.15, 40.05])
LONGITUDE = np.array([-120.45, -120.35, -120.25, -120.15, -120.05])


class _FakePopulation:
    """Stands in for a rasterra population raster in EPSG:4326 (so no reprojection).

    Pixels are 0.013 degree, so their centres fall at many offsets inside a 0.1 degree
    cell and never on a cell edge. Rows run north to south, as in a raster.
    """

    crs = "EPSG:4326"

    def __init__(self, x_min: float, y_max: float, width: int, height: int) -> None:
        self.res = 0.013
        self.x_min, self.y_max = x_min, y_max
        self.width, self.height = width, height

    def x_coordinates(self, *, center: bool = False) -> npt.NDArray[np.floating[Any]]:
        left = self.x_min + self.res * np.arange(self.width)
        return left + self.res / 2 if center else left

    def y_coordinates(self, *, center: bool = False) -> npt.NDArray[np.floating[Any]]:
        # Ascending, as rasterra returns them; corners are each row's north edge.
        bottom = self.y_max - self.res * self.height
        north = bottom + self.res * np.arange(1, self.height + 1)
        return north - self.res / 2 if center else north

    def centres(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Pixel centres in raster order (row 0 north), flattened."""
        xx, yy = np.meshgrid(
            self.x_coordinates(center=True), self.y_coordinates(center=True)[::-1]
        )
        return xx.flatten(), yy.flatten()


class _FakePopulationModel:
    def __init__(self, pop: _FakePopulation) -> None:
        self.pop = pop

    def load_results(self, *_: Any) -> _FakePopulation:
        return self.pop


def _climate(latitude: npt.NDArray[np.float64]) -> xr.Dataset:
    return xr.Dataset(
        {
            "value": (
                ("latitude", "longitude"),
                np.zeros((len(latitude), len(LONGITUDE))),
            )
        },
        coords={"latitude": latitude, "longitude": LONGITUDE},
    )


def _containing_cell(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    latitude: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """The flat index of the cell whose centre is within half a cell of each point."""
    cells = []
    for xi, yi in zip(x, y, strict=True):
        row = np.flatnonzero(np.abs(latitude - yi) < 0.05)  # noqa: PLR2004
        col = np.flatnonzero(np.abs(LONGITUDE - xi) < 0.05)  # noqa: PLR2004
        assert len(row) == 1
        assert len(col) == 1
        cells.append(int(col[0] + row[0] * len(LONGITUDE)))
    return np.array(cells)


@pytest.mark.parametrize("descending", [True, False])
def test_each_pixel_gets_the_cell_that_contains_its_centre(*, descending: bool) -> None:
    # 30 x 25 pixels, all inside the grid.
    pop = _FakePopulation(x_min=-120.4987, y_max=40.3987, width=30, height=25)
    latitude = LAT_DESCENDING if descending else LAT_DESCENDING[::-1].copy()

    cells = utils.get_temperature_coordinates(
        "B-0000X-0000Y",
        _FakePopulationModel(pop),  # type: ignore[arg-type]
        _climate(latitude),
    )

    x, y = pop.centres()
    np.testing.assert_array_equal(cells, _containing_cell(x, y, latitude))


def test_a_known_point_lands_in_its_cell() -> None:
    # One pixel centred at (40.2735 N, 120.3135 W): the cell centred at (40.25, -120.35).
    pop = _FakePopulation(x_min=-120.32, y_max=40.28, width=1, height=1)

    cells = utils.get_temperature_coordinates(
        "B-0000X-0000Y",
        _FakePopulationModel(pop),  # type: ignore[arg-type]
        _climate(LAT_DESCENDING),
    )

    row, col = 1, 1
    assert cells.tolist() == [col + row * len(LONGITUDE)]


@pytest.mark.parametrize("descending", [True, False])
def test_cell_index_for_either_order(*, descending: bool) -> None:
    centres = LAT_DESCENDING if descending else LAT_DESCENDING[::-1].copy()
    values = np.array([40.39, 40.31, 40.27, 40.21, 40.18, 40.01])

    got = utils._cell_index(values, centres)  # noqa: SLF001

    expected = []
    for value in values:
        expected.append(int(np.argmin(np.abs(centres - value))))
    np.testing.assert_array_equal(got, expected)
