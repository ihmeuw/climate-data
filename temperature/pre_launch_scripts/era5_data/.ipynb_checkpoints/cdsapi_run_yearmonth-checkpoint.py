import cdsapi
import argparse
import subprocess

parser = argparse.ArgumentParser(description='Year input')
parser.add_argument('year', metavar='Y', type=str,
                    help='the year of interest as an integer')
parser.add_argument('month', metavar='M', type=str,
                    help='the month of interest as an integer')

args = parser.parse_args()

c = cdsapi.Client()
print(args)
c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': '2m_temperature',
        'year': args.year,
        'month': args.month,
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
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
    },
    '/mnt/share/erf/ERA5/hourly_temp_9km/{}_{}_download.netcdf.zip'.format(int(args.year), int(args.month)))