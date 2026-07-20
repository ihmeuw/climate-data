import numpy as np

from climate_data.special.utils import compute_person_days

GOOD_PIXEL_POP = 10.0


def _minimal_kernel_inputs(population: np.ndarray):
    """Build the smallest valid ``compute_person_days`` call.

    Two high-res pixels, both belonging to location 0 and mapping to the single
    low-res cell, over a single day. Only the population array varies between
    scenarios, so the output cell accumulates exactly ``population.sum()``.
    """
    location_idx = np.array([0, 0], dtype=np.int64)
    temp_coords = np.array([0, 0], dtype=np.int64)
    temp_idx = np.array([[0]], dtype=np.int64)  # (days, low_res_pixel)
    tz_idx = np.array([0], dtype=np.int64)
    out = np.zeros((1, 1, 1), dtype=np.float64)
    return location_idx, temp_idx, tz_idx, population, temp_coords, out


def test_nan_population_poisons_output_cell() -> None:
    """Document the latent bug: a NaN pixel poisons every cell it touches.

    ``compute_person_days`` accumulates with ``out += pop``, so a single NaN
    population pixel turns the whole output cell into NaN (read back as 0, which
    silently zeroed small locations like American Samoa). This is the behavior the
    caller must guard against.
    """
    population = np.array([GOOD_PIXEL_POP, np.nan], dtype=np.float64)
    args = _minimal_kernel_inputs(population)
    compute_person_days(*args)
    out = args[-1]
    assert np.isnan(out[0, 0, 0])


def test_nan_filled_population_does_not_poison() -> None:
    """The fix: zero-filling nodata before the kernel keeps output finite.

    A pixel with no modeled population contributes no person-days, so the good
    pixel's count survives intact.
    """
    population = np.nan_to_num(
        np.array([GOOD_PIXEL_POP, np.nan], dtype=np.float64), nan=0.0
    )
    args = _minimal_kernel_inputs(population)
    compute_person_days(*args)
    out = args[-1]
    assert out[0, 0, 0] == GOOD_PIXEL_POP
