def load_raw_cmip_metadata() -> pd.DataFrame:
    """Loads metadata containing information about all CMIP6 models."""
    path = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
    return pd.read_csv(path)

meta = load_raw_cmip_metadata()

keep_sources = [
    'CAMS-CSM1-0',
    'CanESM5',
    'CNRM-ESM2-1',
    'GFDL-ESM4',
    'GISS-E2-1-G',
    'MIROC-ES2L',
    'MIROC6',
    'MRI-ESM2-0'
]
keep_experiments = [    
    'ssp119',
    'ssp126',
    'ssp245',
    'ssp370',
    'ssp585',
]

keep_variables = [
    "uas",
    "vas",
    "hurs",
    "tas",
    # "rsus",
    # "rlus",
    "ps",
    # "rsds",
    # "rlds",
    "pr",
    # "rsdsdiff",
]

keep_tables = [
    #"Amon",
    "day",
]


mask = (
    meta.source_id.isin(keep_sources)
    & meta.experiment_id.isin(keep_experiments)
    & meta.variable_id.isin(keep_variables)
    & meta.table_id.isin(keep_tables)
)

meta_sub = meta[mask]
meta_sub['dummy'] = "X"

pvs = ['source_id', 'experiment_id', 'variable_id']

meta_sub.groupby(pvs).dummy.apply(lambda s: ",".join(s.unique().tolist())).unstack()

import gcsfs
def load_cmip_data(zarr_path: str) -> xr.Dataset:
    """Loads a CMIP6 dataset from a zarr path."""
    gcs = gcsfs.GCSFileSystem(token="anon")  # noqa: S106
    mapper = gcs.get_mapper(zarr_path)
    ds = xr.open_zarr(mapper, consolidated=True)
    lon = (ds.lon + 180) % 360 - 180
    ds = ds.assign_coords(lon=lon).sortby("lon")
    ds = ds.drop_vars(
        ["lat_bnds", "lon_bnds", "time_bnds", "height", "time_bounds", "bnds"],
        errors="ignore",
    )
    return ds  # type: ignore[no-any-return]