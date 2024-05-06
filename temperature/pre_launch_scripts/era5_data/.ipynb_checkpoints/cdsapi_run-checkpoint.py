import cdsapi
import argparse

parser = argparse.ArgumentParser(description='Year input')
parser.add_argument('year', metavar='Y', type=str,
                    help='the year of interest as a string')
args = parser.parse_args()

c = cdsapi.Client()
print(args)
c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': '2m_temperature',
        'year': args.year,
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'format': 'netcdf.zip',
        # 'format': 'grib',
        'time': [
            '00:00',
            '03:00',
            '06:00',
            '09:00',
            '12:00',
            '15:00',
            '18:00',
            '21:00',
        ],
    },
    '/mnt/share/erf/ERA5/three_hourly_temp_9km/{}_download.zip'.format(args.year))