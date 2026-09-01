import itertools
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from climate_data import (
    cli_options as clio,
)
from climate_data import (
    constants as cdc,
)
from climate_data.data import ClimateData
from climate_data.generate import utils
from climate_data.jobmon_utils import run_parallel_maybe_dry_run

# Map from source variable to a unit conversion function
CONVERT_MAP = {
    "uas": utils.scale_wind_speed_height,
    "vas": utils.scale_wind_speed_height,
    "hurs": utils.identity,
    "tas": utils.kelvin_to_celsius,
    "tasmin": utils.kelvin_to_celsius,
    "tasmax": utils.kelvin_to_celsius,
    "pr": utils.precipitation_flux_to_rainfall,
}

# Map from target variable to:
#  - a list of source variables
#  - a transformation function
#  - a tuple of offset and scale factors for the output for serialization
#  - an anomaly type
TRANSFORM_MAP: dict[str, tuple[utils.Transform, str]] = {
    "mean_temperature": (
        utils.Transform(
            source_variables=["tas"],
            transform_funcs=[utils.identity],
            encoding_scale=0.01,
        ),
        "additive",
    ),
    "max_temperature": (
        utils.Transform(
            source_variables=["tasmax"],
            transform_funcs=[utils.identity],
            encoding_scale=0.01,
        ),
        "additive",
    ),
    "min_temperature": (
        utils.Transform(
            source_variables=["tasmin"],
            transform_funcs=[utils.identity],
            encoding_scale=0.01,
        ),
        "additive",
    ),
    "wind_speed": (
        utils.Transform(
            source_variables=["uas", "vas"],
            transform_funcs=[utils.vector_magnitude],
            encoding_scale=0.01,
        ),
        "multiplicative",
    ),
    "relative_humidity": (
        utils.Transform(
            source_variables=["hurs"],
            transform_funcs=[utils.identity],
            encoding_scale=0.01,
        ),
        "multiplicative",
    ),
    "total_precipitation": (
        utils.Transform(
            source_variables=["pr"],
            transform_funcs=[utils.identity],
            encoding_scale=0.1,
        ),
        "multiplicative",
    ),
}


# Slack on the de-bias factor's `>= 1` guarantee, for floating-point accumulation in the
# fold average. Tight enough that a real violation (0.2 for a collapsed fold) cannot hide.
_DEBIAS_FACTOR_TOLERANCE = 1e-9


ANOMALY_TYPES = {}
for _variable, (_transform, _anomaly_type) in TRANSFORM_MAP.items():
    ANOMALY_TYPES[_variable] = _anomaly_type


def variables_for_anomaly_scheme(
    target_variables: list[str],
    anomaly_scheme: str,
    anomaly_types: dict[str, str],
) -> list[str]:
    """Drop target variables the anomaly scheme cannot be applied to.

    The yearly schemes are defined for multiplicative variables only -- `compute_anomaly`
    raises for anything else -- while `--target-variable` defaults to ALL. Without this
    filter the default invocation of a yearly scheme submits additive jobs that are
    certain to fail, and they fail only after being scheduled and retried.

    Skipped variables are named rather than dropped quietly, and selecting nothing but
    additive variables is an error rather than an empty run.
    """
    if anomaly_scheme == cdc.ANOMALY_SCHEME_MONTHLY:
        return target_variables

    keep = []
    skip = []
    for variable in target_variables:
        if anomaly_types[variable] == "multiplicative":
            keep.append(variable)
        else:
            skip.append(variable)

    if skip:
        print(
            f"Anomaly scheme '{anomaly_scheme}' applies to multiplicative variables only;"
            f" skipping {len(skip)}: {', '.join(skip)}."
        )
    if not keep:
        msg = (
            f"No multiplicative variables selected, so anomaly scheme '{anomaly_scheme}'"
            f" has nothing to run. Selected: {', '.join(target_variables)}."
        )
        raise click.UsageError(msg)
    return keep


def load_and_shift_longitude(
    member_path: str | Path,
    time_slice: slice,
) -> xr.Dataset:
    ds = xr.open_dataset(member_path).sortby("time").sel(time=time_slice).compute()
    if ds.time.size == 0:
        msg = "No data in slice"
        raise KeyError(msg)
    ds = (
        ds.assign_coords(lon=(ds.lon + 180) % 360 - 180)
        .sortby("lon")
        .rename({"lat": "latitude", "lon": "longitude"})
    )
    return ds


def load_and_shift_longitude_and_correct_time(
    member_path: str | Path,
    year: str,
) -> xr.Dataset:
    """Put a member's year onto the real Gregorian calendar, day by day.

    The conversion is by DATE, never by value. `interp_calendar` used to resample onto the
    target axis by linear interpolation, so whenever the source calendar's year length
    differed from the target's -- a `noleap` member in a leap year -- every target day
    became a blend of two source days. That is harmless for a total but not for a
    threshold: a dry day beside a wet one picked up a share of the wet day's rain and
    crossed the 0.1 mm wet-day cut, inflating `precipitation_days` by ~13.5 d per noleap
    member in all 19 leap years 2024-2096. `align_on="date"` keeps each source day's value
    at its own date and leaves 29 February missing, and the `reindex` holds the output axis
    to exactly this year's days regardless of the source calendar. `interpolate_na` then
    fills the gap from the nearest real day. (CLIMATE-35)
    """
    time_slice = slice(f"{year}-01-01", f"{year}-12-31")
    time_range = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    ds = load_and_shift_longitude(member_path, time_slice)
    ds = (
        ds.assign_coords(time=ds.time.dt.floor("D"))
        .convert_calendar("standard", align_on="date")
        .reindex(time=time_range)
        .interpolate_na(dim="time", method="nearest", fill_value="extrapolate")
        .rename({"time": "date"})
    )
    return ds


def load_variable(
    member_path: str | Path,
    year: str | int,
    reference_period: slice = cdc.REFERENCE_PERIOD,
) -> xr.Dataset:
    if year == "reference":
        ds = load_and_shift_longitude(member_path, reference_period).rename(
            {"time": "date"}
        )
    else:
        try:
            ds = load_and_shift_longitude_and_correct_time(member_path, str(year))
        except KeyError as e:
            if int(year) == 2100:  # noqa: PLR2004
                # Some datasets stop in 2099.  Just reuse the last year, relabelled onto
                # 2100's own dates.  This used to add `date.size` days to every stamp,
                # using the axis COUNT as a calendar DURATION -- the two agree only while
                # 2099 is a complete 365-day run, and `assign_coords` validates nothing, so
                # a longer axis would have slid the year onto 2100-01-02..2101-01-01 and
                # `groupby("date.year")` would have filed a day under 2101 in silence.
                # Assigning the target range instead makes a mismatch raise.
                ds = load_and_shift_longitude_and_correct_time(member_path, "2099")
                ds = ds.assign_coords(
                    date=pd.date_range("2100-01-01", "2100-12-31"),
                )
            else:
                raise e

    variable = str(next(iter(ds)))
    conversion = CONVERT_MAP[variable]
    ds = conversion(utils.rename_val_column(ds))
    return ds


def _monthly_means_by_reference_year(reference: xr.Dataset) -> xr.Dataset:
    """Monthly means of the reference period, one slice per reference year.

    Dims ``(reference_year, month, latitude, longitude)``, on the GCM's own grid. The years
    come from the data rather than from ``cdc.REFERENCE_YEARS`` so that the ``(n-1)/n`` rescale
    below stays consistent with whatever window was actually loaded. Selection is by boolean
    mask rather than by date string so it works for the 360-day cftime calendars some GCMs use.
    """
    years = [int(y) for y in np.unique(reference["date"].dt.year.values)]
    by_year = []
    for year in years:
        one_year = reference.sel(date=reference["date"].dt.year == year)
        by_year.append(one_year.groupby("date.month").mean("date"))
    return xr.concat(by_year, dim="reference_year").assign_coords(
        reference_year=years,
    )


def jensen_debias_factor(
    reference: xr.Dataset,
    reference_monthly: xr.Dataset,
    debias_method: str,
    eps: xr.Dataset | float = 1.0,
) -> xr.Dataset:
    """The factor to divide a multiplicative anomaly by, per month and per GCM cell.

    The anomaly is ``(T + 1) / (R + 1)`` with ``R`` a monthly mean over only five reference
    years. ``1/(R + 1)`` is convex, so by Jensen's inequality the anomaly averages above 1 even
    when the target year is drawn from the same distribution as the reference period -- a level
    bias on every forecast year. This returns an estimate of that inflation.

    ``loo`` -- leave-one-out. For each held-out reference year, form the multiplier of that
    year against the mean of the *other* years and average over the folds. Between reference
    years there is no climate signal, so an unbiased estimator would return 1; the excess is
    the bias, measured from the data with no series expansion. The held-out denominator
    averages ``n-1`` years while the pipeline averages ``n``, and the bias goes as ``1/n``, so
    the excess is rescaled by ``(n-1)/n``.

    This is provably ``>= 1``: with ``u_y = T_y + 1`` and ``S = sum_y u_y`` the held-out
    denominator is ``(S - u_y)/(n-1)``, so each fold is ``(n-1)*u_y/(S - u_y)``, convex in
    ``u_y``, and Jensen gives ``mean_y >= f(S/n) = 1`` with equality iff every reference year
    is identical. So dividing by it can only shrink the anomaly, never inflate it -- which is
    what makes the effect on a threshold count such as ``precipitation_days`` sign-definite.

    ``analytic`` -- the second-order expansion ``1 + Var(Rbar)/(R + 1)^2``. Cheaper to reason
    about but it is a truncated series, and the neglected terms matter exactly where the
    correction is largest (near-zero ``R``, where ``eps`` dominates the denominator). Kept for
    comparison; ``loo`` is the estimator this was built for.

    ``reference_monthly`` is the pipeline's own denominator, passed in rather than recomputed so
    the analytic form squares precisely the value the anomaly divides by.
    """
    by_year = _monthly_means_by_reference_year(reference)
    n_years = by_year.sizes["reference_year"]

    if debias_method == "loo":
        mean_year = by_year.mean("reference_year")
        folds = []
        valid = []
        for i in range(n_years):
            held_out = by_year.isel(reference_year=i)
            others = (n_years * mean_year - held_out) / (n_years - 1)
            fold_denominator = others + eps
            folds.append(
                (held_out + eps) / fold_denominator.where(fold_denominator > 0)
            )
            valid.append((fold_denominator > 0).astype("int8"))
        raw = xr.concat(folds, dim="reference_year").mean("reference_year")
        factor = 1.0 + ((n_years - 1) / n_years) * (raw - 1.0)
        # Where any fold is undefined the estimate is undefined, so apply no correction.
        # Averaging the surviving folds is NOT a repair: the undefined one is precisely
        # the large fold that lifts the mean above 1, so dropping it collapses the factor
        # (0.2 for a one-wet-four-dry cell-month) and the de-bias INFLATES the anomaly --
        # the opposite of its purpose. Only reachable when eps can be zero, i.e. under the
        # taper; the constant eps = 1 floors every denominator at 1.
        factor = factor.where(sum(valid) == n_years, 1.0).fillna(1.0)
    elif debias_method == "analytic":
        variance = by_year.var("reference_year", ddof=1)
        factor = 1.0 + (variance / n_years) / (reference_monthly + eps) ** 2
    else:
        msg = f"Unknown debias method: {debias_method}"
        raise ValueError(msg)

    factor = factor.drop_vars("reference_year", errors="ignore")
    values = factor.to_dataarray()
    if not bool(np.isfinite(values).all()):
        msg = (
            f"Non-finite value in the {debias_method} de-bias factor. Interpolation would "
            "silently fill it from a neighbour rather than surface it, so refusing to proceed."
        )
        raise ValueError(msg)
    minimum = float(values.min())
    if minimum < 1.0 - _DEBIAS_FACTOR_TOLERANCE:
        # Both estimators are >= 1 by construction -- loo by Jensen on a convex fold,
        # analytic because it adds a non-negative term. A value below 1 means the
        # construction's precondition failed, and dividing by it would inflate rather
        # than shrink the anomaly. Refuse rather than ship an anti-correction.
        msg = (
            f"The {debias_method} de-bias factor reached {minimum!r}, below its "
            "guaranteed lower bound of 1. Dividing by it would inflate the anomaly."
        )
        raise ValueError(msg)
    return factor


def apply_dry_day_rule(
    anomaly: xr.Dataset,
    target: xr.Dataset,
    dry_day_rule: str,
) -> xr.Dataset:
    """Stop the ``eps`` offset from manufacturing rain on days the model reports as dry.

    The multiplicative anomaly ``(T + eps)/(R + eps)`` is strictly positive even when the
    model reports no rain at all, so a rainless model day still receives
    ``E_ref(month) * a(d) > 0`` of the ERA5 climatology. Worse, ``eps`` dominates the
    numerator for such a day, so *every* rainless day in a cell-month gets the identical
    anomaly ``1/(R_m + eps)`` -- a flat positive floor rather than a dry spell. Wherever that
    floor clears the 0.1 mm/day wet-day threshold the pipeline reports a wet day that neither
    ERA5 nor the GCM has, which is what makes ``precipitation_days`` step at the boundary.

    ``preserve`` zeroes the anomaly on those days and rescales the surviving days of the same
    cell-month so the month's *summed* anomaly is unchanged. Because that sum is preserved per
    GCM cell and ``interpolate_to_target_latlon`` is linear, the monthly -- and therefore the
    annual -- total on the target grid is untouched to floating point. Only the distribution
    across days moves. That is deliberate: this is a shape fix, exactly orthogonal to the
    Jensen de-bias, which is a level fix that leaves the shape alone. The two commute, because
    the de-bias scales a whole month uniformly and this rescale is scale-invariant.

    A cell-month the model reports dry on *every* day has nothing to renormalise onto. Those
    are left exactly as they are rather than zeroed. Zeroing them is the variant that was
    measured and rejected: it loses up to 1.2% of the population-weighted annual total across
    the 1-3% of cell-months that are all-dry. Keeping them is what makes this rule
    total-preserving, and it is the whole difference between the two.
    """
    if dry_day_rule == "none":
        return anomaly
    if dry_day_rule != "preserve":
        msg = f"Unknown dry-day rule: {dry_day_rule}"
        raise ValueError(msg)

    wet = target > cdc.DRY_DAY_THRESHOLD_MM
    kept = anomaly.where(wet, 0.0)

    month_total = anomaly.groupby("date.month").sum("date")
    kept_total = kept.groupby("date.month").sum("date")
    has_wet_day = kept_total > 0
    # 1.0 on an all-dry cell-month, so the restore below survives the rescale untouched.
    rescale = (month_total / kept_total.where(has_wet_day)).fillna(1.0)

    has_wet_day_daily = has_wet_day.sel(month=anomaly["date"].dt.month).drop_vars(
        "month"
    )
    kept = kept.where(has_wet_day_daily, anomaly)

    rescaled = kept.groupby("date.month") * rescale
    return rescaled.drop_vars("month")


def _report_zeroed(n_zeroed: int, unit: str) -> None:
    if n_zeroed:
        # Downstream bilinear regridding spreads a zeroed native cell into
        # its whole 0.1 degree neighbourhood, so make the extent visible.
        print(
            f"Zero-reference guard: {n_zeroed} {unit} have no rain in the "
            f"reference window and will forecast zero; regridding spreads "
            f"these zeros into neighbouring target pixels."
        )


def _yearly_anomaly(
    reference: xr.Dataset, target: xr.Dataset, anomaly_scheme: str
) -> xr.Dataset:
    """Yearly multiplier: one annual-mean denominator instead of twelve monthly ones.

    Because ratio-of-sums equals ratio-of-means, this is identical to raking each
    year's total to the reference level and distributing it over days by the GCM's
    own daily shape. No +1 stabiliser: annual-mean denominators sit far from zero,
    unlike dry-season monthly means, which is what CLIMATE-30 is about.
    """
    reference_mean = reference.mean("date")
    # A reference window with zero rain forecasts zero rain: mask the
    # denominator so a bone-dry cell yields 0 rather than inf/NaN, while a
    # missing (NaN) reference stays NaN.
    positive_mean = reference_mean.where(reference_mean > 0)
    n_zeroed = int(
        sum((reference_mean[v] == 0).sum().item() for v in reference_mean.data_vars)
    )
    _report_zeroed(n_zeroed, "cells")
    anomaly = (target / positive_mean).where(reference_mean > 0, reference_mean * 0.0)
    if anomaly_scheme == cdc.ANOMALY_SCHEME_YEARLY_DELTA:
        # The window-mean denominator is noisy, so E[T/ref] exceeds
        # T/E[ref] (Jensen). Dividing by 1 + Var(mean)/mean**2 removes
        # that mean bias without changing the CV.
        yearly_means = reference.groupby("date.year").mean("date")
        n_years = yearly_means.sizes["year"]
        if n_years < 2:  # noqa: PLR2004
            # With one reference year the variance is NaN and the fillna
            # below would silently turn the correction off.
            msg = (
                f"Anomaly scheme '{cdc.ANOMALY_SCHEME_YEARLY_DELTA}' needs "
                f"at least two reference years to estimate the window "
                f"variance; got {n_years}."
            )
            raise ValueError(msg)
        variance_of_mean = yearly_means.var("year", ddof=1) / n_years
        inflation = (variance_of_mean / positive_mean**2).fillna(0.0)
        anomaly = anomaly / (1.0 + inflation)
    return anomaly


def _monthly_ratio_anomaly(
    reference: xr.Dataset, target: xr.Dataset, anomaly_scheme: str
) -> xr.Dataset:
    """The yearly scheme applied per month, keeping the ERA5 monthly anchor.

    A pure per-month ratio of each day to its month's reference-window mean --
    no +1 stabiliser. A zero reference month forecasts zero; a missing (NaN)
    reference stays NaN. 'monthly-delta' additionally divides each month's
    ratio by its analytic Jensen inflation factor.
    """
    monthly_mean = reference.groupby("date.month").mean("date")
    positive_mean = monthly_mean.where(monthly_mean > 0)
    n_zeroed = int(
        sum((monthly_mean[v] == 0).sum().item() for v in monthly_mean.data_vars)
    )
    _report_zeroed(n_zeroed, "month-cells")
    # 1/mean where the month has rain; 0 where it is bone-dry; NaN where missing.
    factor = (1.0 / positive_mean).where(monthly_mean > 0, monthly_mean * 0.0)
    if anomaly_scheme != cdc.ANOMALY_SCHEME_MONTHLY_RATIO:
        factor = factor / _monthly_jensen_factor(
            reference, positive_mean, anomaly_scheme
        )
    anomaly = target.groupby("date.month") * factor
    return anomaly.drop_vars("month")


def _monthly_jensen_factor(
    reference: xr.Dataset, positive_mean: xr.Dataset, anomaly_scheme: str
) -> xr.Dataset:
    """Per-month analytic Jensen inflation factor, 1 + Var(window mean) / mean**2.

    Bounded below by 1 by construction, with no undefined cases. A leave-one-out
    estimator was evaluated and rejected here: it requires a strictly positive
    series, and this scheme divides by a bare monthly mean that can be zero.
    """
    per_month = reference.resample(date="1MS").mean()
    n_years = int(reference.groupby("date.year").mean("date").sizes["year"])
    if n_years < 2:  # noqa: PLR2004
        msg = (
            f"Anomaly scheme '{anomaly_scheme}' needs at least two reference "
            f"years to estimate the per-month variance; got {n_years}."
        )
        raise ValueError(msg)
    variance_of_mean = per_month.groupby("date.month").var("date", ddof=1) / n_years
    inflation = (variance_of_mean / positive_mean**2).fillna(0.0)
    return 1.0 + inflation


def _monthly_taper_anomaly(
    reference: xr.Dataset,
    reference_monthly: xr.Dataset,
    target: xr.Dataset,
    eps_floor: float,
    *,
    debias_method: str,
    dry_day_rule: str,
) -> xr.Dataset:
    """``(T + e) / (R + e)`` with ``e = max(0, eps_floor - R)`` -- a tapered stabiliser.

    The shipped ``monthly`` scheme uses a constant ``eps = 1``, which damps the model's
    fractional change everywhere by ``R/(R + 1)``: only 71% survives at the global-mean
    reference of 2.43 mm/day, and the projected trend comes out at 0.65 of the driving
    models' own. The taper is zero wherever ``R >= eps_floor``, so those cells pass the
    model's ratio through exactly, and grows only as ``R`` approaches zero.

    It keeps the property that makes the constant form safe and that a bare floor
    ``T / max(R, eps_floor)`` loses: ``e`` enters numerator *and* denominator, so a cell
    the model reports unchanged (``T = R``) still gets exactly 1 and keeps its ERA5
    climatology. A bare floor returns ``R / eps_floor`` there, cutting an unchanged arid
    cell by up to 90%.

    At ``R = 0`` the taper reduces to ``(T + eps_floor) / eps_floor``, which for the
    default floor of 1.0 is exactly what the shipped scheme already does.

    Computed as ``T/D + e/D`` rather than ``(T + e)/D`` because ``e`` is month-indexed
    while ``target`` is date-indexed; chaining two groupbys to add them would outer-broadcast
    the daily array. The two forms are algebraically identical.
    """
    eps = (eps_floor - reference_monthly).clip(min=0.0)
    denominator = reference_monthly + eps  # == max(R, eps_floor)
    if debias_method != "none":
        # Folded into the denominator rather than dividing the anomaly, for the same
        # reason as the constant-eps path: `anomaly / factor` outer-broadcasts.
        denominator = denominator * jensen_debias_factor(
            reference, reference_monthly, debias_method, eps
        )
    offset = (eps / denominator).sel(month=target["date"].dt.month).drop_vars("month")
    anomaly = target.groupby("date.month") / denominator + offset
    anomaly = apply_dry_day_rule(anomaly, target, dry_day_rule)
    return anomaly.drop_vars("month", errors="ignore")


def check_scheme_compatibility(
    anomaly_scheme: str,
    anomaly_type: str,
    debias_method: str,
    dry_day_rule: str,
) -> None:
    """Reject combinations of the two correction axes that cannot both apply.

    `debias_method` and `dry_day_rule` correct the `(T + eps) / (R + eps)` construction.
    Both `monthly` (constant eps) and `monthly-taper` (tapered eps) use it, so both accept
    them. The yearly and monthly-ratio families have no eps for those corrections to act
    on, so asking for either against them is a mistake rather than a no-op -- and silently
    ignoring the request would produce a file whose attrs claim a correction that was
    never applied.
    """
    has_eps = anomaly_scheme in cdc.EPS_BEARING_SCHEMES
    if not has_eps and (debias_method != "none" or dry_day_rule != "none"):
        msg = (
            f"debias_method={debias_method!r} and dry_day_rule={dry_day_rule!r} cannot "
            f"be combined with anomaly_scheme={anomaly_scheme!r}: they correct the eps "
            f"stabiliser, which this scheme does not use. Schemes that carry an eps: "
            f"{list(cdc.EPS_BEARING_SCHEMES)}."
        )
        raise ValueError(msg)
    if anomaly_type != "multiplicative":
        msg = f"Anomaly scheme '{anomaly_scheme}' only applies to multiplicative variables."
        raise ValueError(msg)


def compute_anomaly(
    reference: xr.Dataset,
    target: xr.Dataset,
    anomaly_type: str,
    *,
    debias_method: str,
    dry_day_rule: str,
    anomaly_scheme: str = cdc.ANOMALY_SCHEME_MONTHLY,
    eps_floor: float = cdc.DEFAULT_EPS_FLOOR,
    anomaly_cap: float | None = cdc.DEFAULT_ANOMALY_CAP,
) -> xr.Dataset:
    """The forecast anomaly, optionally capped.

    A thin wrapper so the ceiling applies to every scheme without threading it through
    each one: the scheme dispatch below has several return points, and duplicating the
    clip at each is how one of them ends up missing it.

    Applied on the GCM grid, BEFORE `interpolate_to_target_latlon`. Capping afterwards
    would leave a blown-up cell already smeared across its neighbours by the regrid.

    The cap only removes precipitation -- it cannot conserve, and whatever it clips is
    gone from the total. That is intended: the values it removes are ones no cell's
    observed climatology supports. But it means a cap moves the level, so it must be
    chosen on evidence rather than set defensively.
    """
    anomaly = _compute_anomaly_uncapped(
        reference,
        target,
        anomaly_type,
        debias_method=debias_method,
        dry_day_rule=dry_day_rule,
        anomaly_scheme=anomaly_scheme,
        eps_floor=eps_floor,
    )
    if anomaly_cap is None:
        return anomaly
    if anomaly_type != "multiplicative":
        msg = (
            f"anomaly_cap={anomaly_cap!r} was requested for an additive anomaly. The cap "
            "bounds a multiplier; an additive anomaly is a difference in the variable's "
            "own units, where a ceiling of this kind is meaningless."
        )
        raise ValueError(msg)
    if anomaly_cap <= 0:
        msg = f"anomaly_cap must be positive, got {anomaly_cap!r}."
        raise ValueError(msg)
    return anomaly.clip(max=anomaly_cap)


def _compute_anomaly_uncapped(
    reference: xr.Dataset,
    target: xr.Dataset,
    anomaly_type: str,
    *,
    debias_method: str,
    dry_day_rule: str,
    anomaly_scheme: str = cdc.ANOMALY_SCHEME_MONTHLY,
    eps_floor: float = cdc.DEFAULT_EPS_FLOOR,
) -> xr.Dataset:
    if anomaly_scheme not in cdc.ANOMALY_SCHEMES:
        msg = (
            f"Unknown anomaly scheme: {anomaly_scheme!r}; "
            f"expected one of {cdc.ANOMALY_SCHEMES}."
        )
        raise ValueError(msg)
    if anomaly_scheme not in cdc.EPS_BEARING_SCHEMES:
        check_scheme_compatibility(
            anomaly_scheme, anomaly_type, debias_method, dry_day_rule
        )
        if anomaly_scheme in cdc.YEARLY_ANOMALY_SCHEMES:
            return _yearly_anomaly(reference, target, anomaly_scheme)
        return _monthly_ratio_anomaly(reference, target, anomaly_scheme)
    # `monthly` and `monthly-taper` share the path below: both keep the ERA5 monthly
    # anchor and the eps construction, so both want the additive/multiplicative split,
    # the de-bias fold and the dry-day rule rather than the ratio-family dispatch. They
    # differ only in whether eps is a constant or a taper.
    reference_monthly = reference.groupby("date.month").mean("date")
    if anomaly_type == "additive":
        if debias_method != "none":
            msg = (
                f"debias_method={debias_method!r} was requested for an additive anomaly. The "
                "Jensen bias comes from the convexity of 1/(R + eps) and has no additive "
                "counterpart, so there is nothing to correct."
            )
            raise ValueError(msg)
        if dry_day_rule != "none":
            msg = (
                f"dry_day_rule={dry_day_rule!r} was requested for an additive anomaly. The "
                "rule exists because eps makes a zero-rainfall day come out positive; an "
                "additive anomaly has no such floor, and a 'dry day' is meaningless for "
                "temperature."
            )
            raise ValueError(msg)
        anomaly = target.groupby("date.month") - reference_monthly
    elif anomaly_type == "multiplicative":
        if anomaly_scheme == cdc.ANOMALY_SCHEME_MONTHLY_TAPER:
            return _monthly_taper_anomaly(
                reference,
                reference_monthly,
                target,
                eps_floor,
                debias_method=debias_method,
                dry_day_rule=dry_day_rule,
            )
        denominator = reference_monthly + 1
        if debias_method != "none":
            # Fold the factor into the denominator rather than dividing the anomaly by it.
            # `anomaly / factor` silently OUTER-BROADCASTS to (date, latitude, longitude,
            # month) -- no error raised -- which is a 12x blow-up of an eager multi-GB daily
            # array. Folding costs one 12-month temporary and is numerically identical.
            denominator = denominator * jensen_debias_factor(
                reference, reference_monthly, debias_method
            )
        anomaly = (target + 1).groupby("date.month") / denominator
        anomaly = apply_dry_day_rule(anomaly, target, dry_day_rule)
    else:
        msg = f"Unknown anomaly type: {anomaly_type}"
        raise ValueError(msg)
    anomaly = anomaly.drop_vars("month", errors="ignore")
    return anomaly


def check_debias_variable(
    target_variable: str, debias_method: str, dry_day_rule: str = "none"
) -> None:
    """Refuse a correction for a variable it has not been validated against.

    Called from the launchers as well as from the worker, so that ``--target-variable all``
    fails in a second rather than after submitting thousands of doomed jobs.
    """
    if debias_method != "none" and target_variable not in cdc.DEBIAS_VARIABLES:
        msg = (
            f"debias_method={debias_method!r} is not validated for {target_variable!r}. "
            f"Allowed: {list(cdc.DEBIAS_VARIABLES)}. Name the variable explicitly rather "
            "than using 'all'."
        )
        raise ValueError(msg)
    if dry_day_rule != "none" and target_variable not in cdc.DRY_DAY_VARIABLES:
        msg = (
            f"dry_day_rule={dry_day_rule!r} is not validated for {target_variable!r}. "
            f"Allowed: {list(cdc.DRY_DAY_VARIABLES)}. Name the variable explicitly rather "
            "than using 'all'."
        )
        raise ValueError(msg)


def generate_scenario_daily_main(
    target_variable: str,
    cmip6_experiment: str,
    year: str | int,
    gcm_member: str,
    output_dir: str | Path,
    write_output: bool = True,
    *,
    debias_method: str,
    dry_day_rule: str,
    anomaly_scheme: str = cdc.ANOMALY_SCHEME_MONTHLY,
    reference_years: str = cdc.REFERENCE_YEARS_ARG,
    eps_floor: float = cdc.DEFAULT_EPS_FLOOR,
    anomaly_cap: float | None = cdc.DEFAULT_ANOMALY_CAP,
) -> xr.Dataset:
    # NOTE: debias_method is deliberately keyword-only with NO default. A default of "none"
    # here would mean that forgetting to thread it through generate_scenario_annual_main
    # produces a silently undebiased run that reports success. Let mypy catch the call site.
    cdata = ClimateData(output_dir)
    check_debias_variable(target_variable, debias_method, dry_day_rule)
    reference_period = utils.parse_reference_years(reference_years)

    transform, anomaly_type = TRANSFORM_MAP[target_variable]
    source_paths = [
        cdata.extracted_cmip6_path(source_variable, cmip6_experiment, gcm_member)
        for source_variable in transform.source_variables
    ]

    print("loading historical reference")
    historical_reference = cdata.load_daily_results(
        scenario="historical",
        variable=target_variable,
        year="reference",
    )
    # compute anomaly, resample anomaly and compute scenario data
    # load reference (monthly) and target (daily for a given year)
    print(f"{gcm_member}: Loading reference")
    sref = transform(
        *[load_variable(vp, "reference", reference_period) for vp in source_paths]
    )

    print(f"{gcm_member}: Loading target")
    target = transform(*[load_variable(vp, year) for vp in source_paths])

    print(f"{gcm_member}: computing anomaly")
    v_anomaly = compute_anomaly(
        sref,
        target,
        anomaly_type,
        debias_method=debias_method,
        dry_day_rule=dry_day_rule,
        anomaly_scheme=anomaly_scheme,
        eps_floor=eps_floor,
        anomaly_cap=anomaly_cap,
    )

    print(f"{gcm_member}: resampling anomaly")
    resampled_anomaly = utils.interpolate_to_target_latlon(v_anomaly, method="linear")
    print(f"{gcm_member}: computing scenario data")
    if anomaly_type == "additive":
        scenario_data = historical_reference + resampled_anomaly.groupby("date.month")
    elif anomaly_scheme not in cdc.YEARLY_ANOMALY_SCHEMES:
        # monthly and the monthly-ratio family keep the ERA5 monthly anchor.
        scenario_data = historical_reference * resampled_anomaly.groupby("date.month")
    else:
        # The yearly anomaly is anchored to the annual level, so the level
        # comes from the day-weighted annual mean of the monthly reference.
        scenario_data = (
            utils.annual_mean_from_monthly(historical_reference) * resampled_anomaly
        )
    # Provenance: the output path encodes scenario/variable/year/member only, so without
    # this a yearly file is indistinguishable from a monthly one sitting beside it.
    scenario_data.attrs["debias_method"] = debias_method
    scenario_data.attrs["dry_day_rule"] = dry_day_rule
    scenario_data.attrs["anomaly_scheme"] = anomaly_scheme
    scenario_data.attrs["reference_years"] = reference_years
    scenario_data.attrs["eps_floor"] = eps_floor
    scenario_data.attrs["anomaly_cap"] = "none" if anomaly_cap is None else anomaly_cap

    if write_output is True:
        print(f"{gcm_member}: Writing output")
        cdata.save_raw_daily_results(
            scenario_data,
            scenario=cmip6_experiment,
            variable=target_variable,
            year=year,
            gcm_member=gcm_member,
            encoding_kwargs=transform.encoding_kwargs,
        )
    else:
        print(f"{gcm_member}: Returning output")

    return scenario_data


@click.command()
@clio.with_target_variable(list(TRANSFORM_MAP))
@clio.with_cmip6_experiment()
@clio.with_year(cdc.FORECAST_YEARS)
@clio.with_gcm_member()
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_debias_method()
@clio.with_dry_day_rule()
@clio.with_anomaly_scheme()
@clio.with_reference_years()
@clio.with_eps_floor()
@clio.with_anomaly_cap()
def generate_scenario_daily_task(
    target_variable: str,
    cmip6_experiment: str,
    year: str,
    gcm_member: str,
    output_dir: str,
    debias_method: str,
    dry_day_rule: str,
    anomaly_scheme: str,
    reference_years: str,
    eps_floor: float,
    anomaly_cap: float | None,
) -> None:
    generate_scenario_daily_main(
        target_variable,
        cmip6_experiment,
        year,
        gcm_member,
        output_dir,
        write_output=True,
        debias_method=debias_method,
        dry_day_rule=dry_day_rule,
        anomaly_scheme=anomaly_scheme,
        reference_years=reference_years,
        eps_floor=eps_floor,
        anomaly_cap=anomaly_cap,
    )


@click.command()
@clio.with_target_variable(TRANSFORM_MAP, allow_all=True)
@clio.with_cmip6_experiment(allow_all=True)
@clio.with_year(cdc.FORECAST_YEARS, allow_all=True)
@clio.with_output_directory(cdc.MODEL_ROOT)
@clio.with_debias_method()
@clio.with_dry_day_rule()
@clio.with_anomaly_scheme()
@clio.with_reference_years()
@clio.with_eps_floor()
@clio.with_anomaly_cap()
@clio.with_queue()
@clio.with_overwrite()
@clio.with_dry_run()
def generate_scenario_daily(
    target_variable: list[str],
    cmip6_experiment: list[str],
    year: list[str],
    output_dir: str,
    debias_method: str,
    dry_day_rule: str,
    anomaly_scheme: str,
    reference_years: str,
    eps_floor: float,
    anomaly_cap: float | None,
    queue: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    # Fail before submitting anything: with `-t all` a de-bias request would otherwise die
    # one job at a time, after the whole fan-out is already queued.
    for variable in target_variable:
        check_debias_variable(variable, debias_method, dry_day_rule)
    cdata = ClimateData(output_dir)
    target_variable = variables_for_anomaly_scheme(
        target_variable, anomaly_scheme, ANOMALY_TYPES
    )
    veyg = []
    complete = []
    for v, e, y in itertools.product(target_variable, cmip6_experiment, year):
        source_variables = TRANSFORM_MAP[v][0].source_variables
        gcms = cdata.get_gcms(source_variables)
        for g in gcms:
            path = cdata.raw_daily_results_path(e, v, y, g)
            if not path.exists() or overwrite:
                veyg.append((v, e, y, g))
            else:
                complete.append((v, e, y, g))
    if not veyg:
        print("All tasks already done.")
        return

    print(f"{len(complete)} tasks already done. Launching {len(veyg)} tasks")
    run_parallel_maybe_dry_run(
        runner="cdtask",
        task_name="generate scenario_daily",
        flat_node_args=(
            ("target-variable", "cmip6-experiment", "year", "gcm-member"),
            veyg,
        ),
        task_args={
            "output-dir": output_dir,
            "debias-method": debias_method,
            "dry-day-rule": dry_day_rule,
            "anomaly-scheme": anomaly_scheme,
            "reference-years": reference_years,
            "eps-floor": eps_floor,
            "anomaly-cap": anomaly_cap,
        },
        task_resources={
            "queue": queue,
            "cores": 1,
            "memory": "90G",
            "runtime": "20m",
            "project": "proj_rapidresponse",
        },
        max_attempts=2,
        dry_run=dry_run,
    )
