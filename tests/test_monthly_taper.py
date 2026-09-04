"""Tests for the `monthly-taper` anomaly scheme and the generalised Jensen de-bias.

The taper replaces the constant `eps = 1` with `e = max(0, eps_floor - R)`. Its whole
justification rests on three properties, so each is asserted directly rather than
inferred from an aggregate: level-neutrality everywhere, exactness above the floor, and
a de-bias factor that never drops below 1.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_data import constants as cdc
from climate_data.generate.scenario_daily import (
    compute_anomaly,
    jensen_debias_factor,
)

TAPER = cdc.ANOMALY_SCHEME_MONTHLY_TAPER
CONTINUITY_TOL = 1e-5
TEST_CAP = 20.0
WET_REFERENCE_YEAR = 2019
LAT = np.arange(2.0)
LON = np.arange(5.0)
# Wet through bone dry, straddling the default floor of 1.0 mm/day.
REFERENCE_RATES = np.array([[5.0, 2.0, 1.0, 0.3, 0.0]] * 2)


def _dataset(dates: pd.DatetimeIndex, values: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"value": (("date", "latitude", "longitude"), values)},
        coords={"date": dates, "latitude": LAT, "longitude": LON},
    )


def _flat(dates: pd.DatetimeIndex, rates: np.ndarray) -> xr.Dataset:
    return _dataset(dates, np.broadcast_to(rates, (len(dates), *rates.shape)).copy())


@pytest.fixture
def reference() -> xr.Dataset:
    return _flat(pd.date_range("2019-01-01", "2023-12-31", freq="D"), REFERENCE_RATES)


@pytest.fixture
def target_dates() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", "2024-12-31", freq="D")


def _taper(
    reference: xr.Dataset,
    target: xr.Dataset,
    *,
    debias_method: str = "none",
    dry_day_rule: str = "none",
    eps_floor: float = 1.0,
) -> xr.Dataset:
    return compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method=debias_method,
        dry_day_rule=dry_day_rule,
        anomaly_scheme=TAPER,
        eps_floor=eps_floor,
    )


def test_level_neutral_at_every_reference_rate(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """A cell the model reports unchanged keeps its ERA5 climatology exactly.

    This is the property a bare floor `T / max(R, f)` loses: it returns `R / f` here,
    cutting an unchanged arid cell by up to 90%.
    """
    target = _flat(target_dates, REFERENCE_RATES)
    anomaly = _taper(reference, target)["value"].to_numpy()
    np.testing.assert_allclose(anomaly, 1.0, rtol=0, atol=1e-12)


def test_exact_above_the_floor(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """Where R >= floor the taper is zero, so the model's ratio passes through exactly."""
    target = _flat(target_dates, REFERENCE_RATES * 1.1)
    anomaly = _taper(reference, target)["value"].to_numpy()
    # columns 0, 1, 2 have R = 5.0, 2.0, 1.0, all >= the floor of 1.0
    np.testing.assert_allclose(anomaly[:, :, :3], 1.1, rtol=0, atol=1e-12)


def test_damps_less_than_the_constant_eps_everywhere(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """The taper never damps more than the shipped constant-eps scheme.

    `R/f > R/(R+1)` below the floor and the taper is exact above it, so this holds
    pointwise -- the reason `monthly-taper` was chosen over a larger floor.
    """
    target = _flat(target_dates, REFERENCE_RATES * 1.1)
    tapered = _taper(reference, target)["value"].to_numpy()
    constant = compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY,
    )["value"].to_numpy()
    assert (tapered >= constant - 1e-12).all()
    assert (tapered[:, :, :3] > constant[:, :, :3]).all()


def test_matches_the_shipped_scheme_where_the_reference_is_zero(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """At R = 0 the taper reduces to (T + f)/f, which at f = 1 is the shipped form."""
    target = _flat(target_dates, REFERENCE_RATES * 1.1)
    tapered = _taper(reference, target)["value"].to_numpy()
    constant = compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY,
    )["value"].to_numpy()
    np.testing.assert_allclose(tapered[:, :, 4], constant[:, :, 4], rtol=0, atol=1e-12)


def test_continuous_at_the_changeover(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """No jump as R crosses the floor: T/R and (T + f - R)/f agree at R = f."""
    rates = np.array([[1.0 - 1e-6, 1.0, 1.0 + 1e-6]] * 2)
    lon = np.arange(3.0)
    ref = xr.Dataset(
        {
            "value": (
                ("date", "latitude", "longitude"),
                np.broadcast_to(rates, (1826, 2, 3)).copy(),
            )
        },
        coords={
            "date": pd.date_range("2019-01-01", "2023-12-31", freq="D"),
            "latitude": LAT,
            "longitude": lon,
        },
    )
    tgt = xr.Dataset(
        {
            "value": (
                ("date", "latitude", "longitude"),
                np.broadcast_to(rates * 1.2, (366, 2, 3)).copy(),
            )
        },
        coords={"date": target_dates, "latitude": LAT, "longitude": lon},
    )
    anomaly = _taper(ref, tgt)["value"].to_numpy()[0, 0, :]
    assert abs(anomaly[0] - anomaly[1]) < CONTINUITY_TOL
    assert abs(anomaly[2] - anomaly[1]) < CONTINUITY_TOL


def test_debias_factor_at_eps_one_matches_the_shipped_result(
    reference: xr.Dataset,
) -> None:
    """The generalised factor is a strict superset: e = 1 is the shipped special case."""
    monthly = reference.groupby("date.month").mean("date")
    rng = np.random.default_rng(0)
    noisy = reference * (0.5 + rng.random(reference["value"].shape))
    noisy_monthly = noisy.groupby("date.month").mean("date")
    generalised = jensen_debias_factor(noisy, noisy_monthly, "loo", 1.0)
    shipped = jensen_debias_factor(noisy, noisy_monthly, "loo")
    xr.testing.assert_identical(generalised, shipped)
    assert monthly is not None  # fixture sanity


def test_debias_factor_survives_a_one_wet_four_dry_window() -> None:
    """The undefined-fold case must yield 1.0, not a collapsed factor below 1.

    With e = 0 and four dry reference years, the fold that holds out the wet year has a
    zero denominator. Dropping it and averaging the survivors discards precisely the
    large fold that makes Jensen's bound hold, collapsing the factor to 0.2 -- which
    would make the de-bias inflate the anomaly instead of shrinking it.
    """
    dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    values = np.zeros((len(dates), 1, 1))
    values[dates.year == WET_REFERENCE_YEAR] = 1.0  # one wet year, four bone dry
    ref = xr.Dataset(
        {"value": (("date", "latitude", "longitude"), values)},
        coords={"date": dates, "latitude": [0.0], "longitude": [0.0]},
    )
    monthly = ref.groupby("date.month").mean("date")
    eps = (0.05 - monthly).clip(min=0.0)  # floor below the wet year -> e = 0 there
    factor = jensen_debias_factor(ref, monthly, "loo", eps)
    assert float(factor["value"].min()) >= 1.0


def test_taper_accepts_the_debias_axis(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """`monthly-taper` carries an eps, so the de-bias and dry-day rule apply to it."""
    target = _flat(target_dates, REFERENCE_RATES * 1.1)
    anomaly = _taper(reference, target, debias_method="loo", dry_day_rule="preserve")
    assert np.isfinite(anomaly["value"].to_numpy()).all()


def test_eps_free_schemes_still_reject_the_debias_axis(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """The ratio family has no eps, so asking for a correction to it is an error."""
    target = _flat(target_dates, REFERENCE_RATES * 1.1)
    with pytest.raises(ValueError, match="does not use"):
        compute_anomaly(
            reference,
            target,
            "multiplicative",
            debias_method="loo",
            dry_day_rule="none",
            anomaly_scheme=cdc.ANOMALY_SCHEME_MONTHLY_RATIO,
        )


# --------------------------------------------------------------------------
# Anomaly cap (--anomaly-cap). Separate axis from eps: it bounds the multiplier
# for ANY multiplicative scheme, and exists because a GCM whose reference window
# is near-zero in a cell can produce an anomaly in the hundreds.
# --------------------------------------------------------------------------


def _capped(
    reference: xr.Dataset,
    target: xr.Dataset,
    scheme: str,
    *,
    anomaly_cap: float | None,
) -> xr.DataArray:
    return compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=scheme,
        anomaly_cap=anomaly_cap,
    )["value"]


def test_cap_bounds_the_period_mean_not_each_day(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """The ceiling is on the multiplier the scheme anchors with, not on each day.

    A flat target cannot distinguish the two -- every day equals the mean -- so this
    uses a spiky one. Bounding each day instead would clip ordinary heavy rainfall:
    against an annual denominator, 20x sits inside the normal distribution of daily
    values, and on real extracts an elementwise clip altered a fifth of land pixels,
    82% of which had an annual anomaly at or below 2.
    """
    # the year's MEAN must clear the cap for this to test anything, so the spike has
    # to be big enough to carry it there on its own
    spike = np.ones(len(target_dates))
    spike[:5] = 4000.0  # a few enormous days, most of the year calm
    target = _flat(target_dates, REFERENCE_RATES) * xr.DataArray(
        spike, coords={"date": target_dates}, dims="date"
    )
    uncapped = _capped(
        reference, target, cdc.ANOMALY_SCHEME_YEARLY_DELTA, anomaly_cap=None
    )
    capped = _capped(
        reference, target, cdc.ANOMALY_SCHEME_YEARLY_DELTA, anomaly_cap=TEST_CAP
    )

    assert float(uncapped.mean("date").max()) > TEST_CAP, (
        "fixture must exceed the cap for this to test anything"
    )
    # the ANNUAL mean is brought to the ceiling ...
    assert float(capped.mean("date").max()) <= TEST_CAP + 1e-9
    # ... while individual days may still exceed it, which an elementwise clip forbade
    assert float(capped.max()) > TEST_CAP, (
        "a rescale preserves within-year shape; if no day exceeds the cap this has "
        "silently become an elementwise clip again"
    )
    # and the shape is preserved exactly: one factor for the whole cell
    ratio = (capped / uncapped).to_numpy()
    ratio = ratio[np.isfinite(ratio)]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9, atol=0)


def test_cap_leaves_a_cell_below_the_ceiling_untouched(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """A cell whose period mean is under the cap must be bit-identical to uncapped."""
    target = _flat(target_dates, REFERENCE_RATES * 2.0)
    uncapped = _capped(
        reference, target, cdc.ANOMALY_SCHEME_YEARLY_DELTA, anomaly_cap=None
    ).to_numpy()
    capped = _capped(
        reference, target, cdc.ANOMALY_SCHEME_YEARLY_DELTA, anomaly_cap=TEST_CAP
    ).to_numpy()
    assert np.nanmax(uncapped) < TEST_CAP, "fixture must sit below the cap"
    np.testing.assert_allclose(capped, uncapped, rtol=0, atol=1e-12)


def test_cap_applies_at_each_schemes_own_granularity(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """The cap is orthogonal to the eps axis, and it bounds where the scheme anchors.

    The monthly family anchors each calendar month to ERA5 separately, so the ceiling
    belongs on the per-month multiplier; the yearly family anchors the annual level, so
    it belongs on the period mean.

    A flat target cannot tell those apart -- every month's mean equals the year's -- so
    the assertion passes for either granularity and proves nothing. This uses a seasonal
    target instead, where one wet month sits far above the annual level, which is what
    makes the two bounds observably different.
    """
    seasonal = np.ones(len(target_dates))
    seasonal[target_dates.month == 1] = 60.0
    target = _flat(target_dates, REFERENCE_RATES * 40.0) * xr.DataArray(
        seasonal, coords={"date": target_dates}, dims="date"
    )
    for scheme in (cdc.ANOMALY_SCHEME_MONTHLY, TAPER, cdc.ANOMALY_SCHEME_YEARLY_DELTA):
        uncapped = _capped(reference, target, scheme, anomaly_cap=None)
        capped = _capped(reference, target, scheme, anomaly_cap=TEST_CAP)
        assert float(uncapped.mean("date").max()) > TEST_CAP, (
            f"fixture must exceed the cap for this to test anything ({scheme})"
        )
        by_month = capped.groupby("date.month").mean("date")
        if scheme in cdc.YEARLY_ANOMALY_SCHEMES:
            # the annual multiplier is brought to the ceiling ...
            assert float(capped.mean("date").max()) <= TEST_CAP + 1e-9, scheme
            # ... and only that one is, so the wet month stays well above it. The margin
            # matters: a per-month bound leaves the wet month at the ceiling to within
            # floating-point noise (20.000000000000004 against a cap of 20.0), which a
            # bare `> TEST_CAP` accepts. The fixture puts the true value near 10x.
            assert float(by_month.max()) > 2 * TEST_CAP, scheme
        else:
            # every calendar month is bounded, which implies the annual mean is too
            assert float(by_month.max()) <= TEST_CAP + 1e-9, scheme
            assert float(capped.mean("date").max()) <= TEST_CAP + 1e-9, scheme


def test_cap_is_off_by_default(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """Shipped behaviour is uncapped; the default must not silently clip."""
    target = _flat(target_dates, REFERENCE_RATES * 40.0)
    assert cdc.DEFAULT_ANOMALY_CAP is None
    a = _taper(reference, target)["value"].to_numpy()
    b = compute_anomaly(
        reference,
        target,
        "multiplicative",
        debias_method="none",
        dry_day_rule="none",
        anomaly_scheme=TAPER,
        eps_floor=1.0,
        anomaly_cap=None,
    )["value"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_cap_rejects_an_additive_anomaly(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    """A ceiling on a multiplier is meaningless for a difference in native units."""
    target = _flat(target_dates, REFERENCE_RATES)
    with pytest.raises(ValueError, match="additive anomaly"):
        compute_anomaly(
            reference,
            target,
            "additive",
            debias_method="none",
            dry_day_rule="none",
            anomaly_cap=TEST_CAP,
        )


def test_cap_rejects_a_non_positive_value(
    reference: xr.Dataset, target_dates: pd.DatetimeIndex
) -> None:
    target = _flat(target_dates, REFERENCE_RATES)
    with pytest.raises(ValueError, match="must be positive"):
        compute_anomaly(
            reference,
            target,
            "multiplicative",
            debias_method="none",
            dry_day_rule="none",
            anomaly_scheme=TAPER,
            anomaly_cap=0.0,
        )
