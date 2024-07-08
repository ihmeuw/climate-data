from pathlib import Path

import pandas as pd
import xarray as xr
from rra_tools import parallel
import tqdm

from climate_downscale.data import DEFAULT_ROOT, ClimateDownscaleData

import warnings

warnings.filterwarnings('ignore')

cd_data = ClimateDownscaleData(output_dir)
paths = list(cd_data.extracted_cmip6.glob(f'*.nc'))

def extract_metadata(data_path: Path) -> tuple:
    variable, scenario, source, variant = data_path.stem.split('_')
    
    realization = variant.split('i')[0][1:]
    initialization = variant.split('i')[1].split('p')[0]
    physics = variant.split('p')[1].split('f')[0]
    forcing = variant.split('f')[1]
    
    
    ds = xr.open_dataset(data_path)
    year_start = ds['time.year'].min().item()
    year_end = ds['time.year'].max().item()
    return (variable, scenario, source, variant, realization, initialization, physics, forcing, year_start, year_end)

meta_list = parallel.run_parallel(
    extract_metadata, 
    paths, 
    num_cores=25,
    progress_bar=True,    
)

meta_df = (
    pd.DataFrame(
        meta_list, 
        columns=[
            'variable', 
            'scenario', 
            'source',
            'variant', 
            'realization', 
            'initialization', 
            'physics', 
            'forcing', 
            'year_start', 
            'year_end',
        ],
    ).assign(
        all_years=lambda x: (x.year_start <= 2020) & (x.year_end >= 2099),
        year_range=lambda x: x.apply(lambda r: f"{r.loc['year_start']}_{r.loc['year_end']}", axis=1),
    )
)

valid_scenarios = (
    meta_df
    .set_index(['variable', 'source', 'variant', 'scenario']).all_years
    .unstack()
    .fillna(False)
    .sum(axis=1)
    .rename('valid_scenarios')
)
year_range = (
    meta_df
    .set_index(['variable', 'source', 'variant', 'scenario']).year_range
    .unstack()
    .fillna("")
)
inclusion_df = pd.concat([
    year_range, 
    valid_scenarios,
    meta_df.drop(columns=['scenario', 'year_start', 'year_end', 'all_years', 'year_range']).drop_duplicates().set_index(['variable', 'source', 'variant'])
], axis=1)
inclusion_df['include'] = inclusion_df.valid_scenarios == 5