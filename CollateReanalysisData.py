# This script loops through each location in the gridded datasets, selects the past X number of years of time series,
# then performs so co-correlation stochastic analysis in order to reproduce the 6-hr climatology of that grid.
#
# The process:
# - download all missing reanalysis data
# - find all the different latitude-longitude combinations
# - loop through each location and extract all the different variables into a single time series
# - analyses the time series to determine inter-dependencies
# - save the summaries for that location
# - repeat until all locations analysed

import os
import pathlib
import numpy as np
import pandas
import urllib.request
import xarray as xr
import matplotlib.pyplot as plt

# get current dir allowing for unix and windows differences
os.path.dirname(os.path.abspath(__file__))
dir_root = pathlib.Path(os.getcwd())

# set dir path of reanalysis data
reanalysis_dir = dir_root / "reanalysis_data"

start_year = 2017
years_of_data = np.linspace(start_year, 2017, 2017 - start_year + 1).astype(int).astype(str)
# make a dictionary for the download using the raw var name as reference. Each variable has its own dictionary
# containing information about the FTP parent, FTP sub directory, the local save format, the actual variable name within
# the netCDF file, and (if applicable) the pressure levels with which to extract from.

netcdf_data_dict = dict(
    tcdc={'parent': 'other_gauss/', 'sub_dir': 'tcdc.eatm.gauss.', 'save_format': 'tcdc.eatm.gauss.',
          'variable': 'tcdc', 'conversion_multiplier': [0.1], 'rounding': True},
    # conversion multiplier of 0.1 converts % to tenths
    lcb={'parent': 'other_gauss/', 'sub_dir': 'pres.lcb.gauss.', 'save_format': 'pres.lcb.gauss.', 'variable': 'pres',
         'conversion_multiplier': [0.01], 'rounding': False},
    mcb={'parent': 'other_gauss/', 'sub_dir': 'pres.mcb.gauss.', 'save_format': 'pres.mcb.gauss.', 'variable': 'pres',
         'conversion_multiplier': [0.01], 'rounding': False},
    hcb={'parent': 'other_gauss/', 'sub_dir': 'pres.hcb.gauss.', 'save_format': 'pres.hcb.gauss.', 'variable': 'pres',
         'conversion_multiplier': [0.01], 'rounding': False},
    # conversion multiplier of 0.01 converts pa to mb
    vwnd={'parent': 'pressure/', 'sub_dir': 'vwnd.', 'save_format': 'vwnd.', 'variable': 'vwnd', 'iso_levels':
        np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]),
          'conversion_multiplier': [1], 'rounding': False},
    uwnd={'parent': 'pressure/', 'sub_dir': 'uwnd.', 'save_format': 'uwnd.', 'variable': 'uwnd', 'iso_levels':
        np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]),
          'conversion_multiplier': [1], 'rounding': False},
    pres={'parent': 'surface/', 'sub_dir': 'pres.sfc.', 'save_format': 'pres.sfc.', 'variable': 'pres',
          'conversion_multiplier': [0.01], 'rounding': True},
    # conversion multiplier of 0.01 converts pa to mb
    air={'parent': 'surface_gauss/', 'sub_dir': 'air.2m.gauss.', 'save_format': 'air.2m.gauss.', 'variable': 'air',
         'conversion_multiplier': [1], 'rounding': False},
    pr_wtr={'parent': 'surface/', 'sub_dir': 'pr_wtr.eatm.', 'save_format': 'pr_wtr.eatm.', 'variable': 'pr_wtr',
            'conversion_multiplier': [1], 'rounding': False},
    # conversion multiplier of 1 indicates no change
)

ncep_url_root = 'ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis/'
url_end = '.nc'

# The directories must exist and data to be downloaded if not.
ncep_dirs = ['lcb', 'mcb', 'hcb', 'air', 'pres', 'tcdc', 'uwnd', 'vwnd', 'pr_wtr']
for i in ncep_dirs:
    var_dir = reanalysis_dir / i
    if not os.path.isdir(var_dir):
        os.mkdir(bytes(var_dir))
    # the file certainly exists once here.
    for j in years_of_data:
        file_name = os.path.join(var_dir, netcdf_data_dict[i]['save_format'] + j + url_end)
        if not os.path.isfile(file_name):
            # make the url for download
            url = ncep_url_root + netcdf_data_dict[i]['parent'] + netcdf_data_dict[i]['sub_dir'] + j + url_end
            print('Downloading ' + netcdf_data_dict[i]['save_format'] + j + url_end)
            urllib.request.urlretrieve(url, file_name)

# check whether the output dir exists, if not, make it
out_dir = os.path.join(reanalysis_dir, 'python_summaries')
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# establish the latitudes and longitudes available in the native format of the ncep reanalysis data. Each lat/lon
# location will be looped through in turn, and the statistics derived.
# to find a full list, load one of the files
ds = xr.open_dataset(os.path.join(reanalysis_dir, 'air', netcdf_data_dict['air']['save_format'] + '2017.nc'))
lats = ds.lat.values
lons = ds.lon.values
del ds
# find all combinations of lat and lon
locations = np.array(np.meshgrid(lats, lons)).T.reshape(-1, 2)
# plt.scatter(locations[:, 0], locations[:, 1]).show() # scatter plot demonstrating total coverage

#### OVERWRITE
locations = [[50, 50]]

# loop through each gridded location in the world extracting the lat and lon in each iteration
for lat, lon in locations:

    # variables to extract a time series of:
    # - pressure
    # - temperature at 2m
    # - total cloudiness
    # - wind speed at 2m
    # - wind speed at bottom cloud height
    # - wind speed at middle cloud height
    # - wind speed at top cloud height

    # Single dimension extraction variables: pressure, temperature, total cloudiness
    ncep_dirs_for_extraction = ['pres', 'tcdc', 'air', 'pr_wtr']
    # loop each reanalysis data type directory
    for i in ncep_dirs_for_extraction:
        var_dir_path = os.path.join(reanalysis_dir, i)
        files_in_var_dir = [f for f in os.listdir(var_dir_path) if os.path.isfile(os.path.join(var_dir_path, f))]

        # loop each file within the directory
        for j in files_in_var_dir:
            # extract the year of this file and check against the start year
            yr = j[-7:-3]

            if int(yr) >= start_year:
                # set the file path of the data file
                file_path = os.path.join(var_dir_path, j)
                # open the nc file as an xarray
                ds = xr.open_dataset(file_path)
                this_years_data = ds[netcdf_data_dict[i]['variable']].sel(lat=lat, lon=lon, method='nearest')

                # append this years data to the main data frame
                if 'historic_data' not in globals():
                    historic_data = this_years_data
                else:
                    historic_data = xr.concat([historic_data, this_years_data], dim='time')

                print("Extracted data from " + j)

        # squeeze away the extra dimensions
        # apply conversions to the different data variables depending on desired output (e.g. pascals to milibar)
        historic_data = historic_data.squeeze().values * np.array(netcdf_data_dict[i]['conversion_multiplier'])

        # if data should be rounded, round it
        if netcdf_data_dict[i]['rounding']:
            historic_data = np.round(historic_data)

        # assign historic_data to a variable named pres, tcdc, air.
        globals()[i] = historic_data
        del historic_data

    # now repeat the process for the three dimensional variables
    ncep_dirs_for_extraction = ['uwnd', 'vwnd']
    ncep_dirs_from_each_wnd = ['lcb', 'mcb', 'hcb']
    # loop each reanalysis data type directory
    for i in ncep_dirs_for_extraction:
        var_dir_path = os.path.join(reanalysis_dir, i)
        files_in_var_dir = [f for f in os.listdir(var_dir_path) if os.path.isfile(os.path.join(var_dir_path, f))]

        # loop each file within the directory
        for j in files_in_var_dir:
            # check file year against start year
            yr = j[-7:-3]

            if int(yr) >= start_year:
                # set the file path of the data file
                file_path = os.path.join(var_dir_path, j)
                # open the wind nc file as an xarray
                ds = xr.open_dataset(file_path)

                # specific extraction of uwnd and vwnd as it requires an isobaric level as an input. This isobaric level
                # corresponds to the middle cloud base height pressure as stored in lcb, mcb and hcb
                # find the current year of this file

                for k in ncep_dirs_from_each_wnd:
                    # load up the cloud base heights from the low, middle and high layers
                    cloud_base_file_path = os.path.join(reanalysis_dir, k,
                                                        netcdf_data_dict[k]['save_format'] + yr + '.nc')
                    cloud_base_ds = xr.open_dataset(cloud_base_file_path)
                    cloud_base = cloud_base_ds[netcdf_data_dict[k]['variable']].sel(lat=lat, lon=lon, method='nearest')
                    cloud_base = cloud_base * np.array(netcdf_data_dict[k]['conversion_multiplier'])
                    print("Extracted cloud base data from " + netcdf_data_dict[k]['save_format'] + yr + '.nc')
                    globals()[k] = cloud_base

                # extract the data from u/vwnd using the mcb as the level indicator and with nearest neighbour as guide
                this_years_data_lcb = ds[netcdf_data_dict[i]['variable']].sel(time=ds.time, lat=lat, lon=lon, level=lcb,
                                                                              method='nearest')
                this_years_data_mcb = ds[netcdf_data_dict[i]['variable']].sel(time=ds.time, lat=lat, lon=lon, level=mcb,
                                                                              method='nearest')
                this_years_data_hcb = ds[netcdf_data_dict[i]['variable']].sel(time=ds.time, lat=lat, lon=lon, level=hcb,
                                                                              method='nearest')
                this_years_data_ground = ds[netcdf_data_dict[i]['variable']].sel(time=ds.time, lat=lat, lon=lon,
                                                                                 level=1000, method='nearest')

                # append this years data to the main data frame
                # should 1 not exist, they all wont exist
                if 'historic_data_lcb' not in globals():
                    historic_data_lcb = this_years_data_lcb
                    historic_data_mcb = this_years_data_mcb
                    historic_data_hcb = this_years_data_hcb
                    historic_data_ground = this_years_data_ground
                else:
                    historic_data_lcb = xr.concat([historic_data_lcb, this_years_data_lcb], dim='time')
                    historic_data_mcb = xr.concat([historic_data_mcb, this_years_data_mcb], dim='time')
                    historic_data_hcb = xr.concat([historic_data_hcb, this_years_data_hcb], dim='time')
                    historic_data_ground = xr.concat([historic_data_ground, this_years_data_ground], dim='time')

                print("Extracted data from " + j)
                del lcb, mcb, hcb

        # squeeze away the extra dimensions
        # apply conversions to the different data variables depending on desired output (e.g. pascals to milibar)
        historic_data_lcb = historic_data_lcb.squeeze().values * np.array(netcdf_data_dict[i]['conversion_multiplier'])
        historic_data_mcb = historic_data_mcb.squeeze().values * np.array(netcdf_data_dict[i]['conversion_multiplier'])
        historic_data_hcb = historic_data_hcb.squeeze().values * np.array(netcdf_data_dict[i]['conversion_multiplier'])
        historic_data_ground = historic_data_ground.squeeze().values * np.array(
            netcdf_data_dict[i]['conversion_multiplier'])

        # assign historic_data to a variable named pres, tcdc, air.
        globals()[i + '_lcb'] = historic_data_lcb
        globals()[i + '_mcb'] = historic_data_mcb
        globals()[i + '_hcb'] = historic_data_hcb
        globals()[i + '_ground'] = historic_data_ground
        del historic_data_lcb, historic_data_mcb, historic_data_hcb, historic_data_ground

    # convert u and v windspeeds into a total wind speed and direction
    # method of deriving wind speed = hypot(uwnd,vwnd)
    wind_speed_lcb = np.hypot(uwnd_lcb, vwnd_lcb)
    wind_speed_mcb = np.hypot(uwnd_mcb, vwnd_mcb)
    wind_speed_hcb = np.hypot(uwnd_hcb, vwnd_hcb)
    wind_speed_ground = np.hypot(uwnd_ground, vwnd_ground)

    # method of deriving direction = atan2(vwnd, uwnd)
    wind_direction_lcb = np.arctan2(vwnd_lcb, uwnd_lcb)*180/np.pi
    wind_direction_mcb = np.arctan2(vwnd_mcb, uwnd_mcb)*180/np.pi
    wind_direction_hcb = np.arctan2(vwnd_hcb, uwnd_hcb)*180/np.pi
    wind_direction_ground = np.arctan2(vwnd_ground, uwnd_ground)*180/np.pi

    del uwnd_lcb, uwnd_mcb, uwnd_hcb, uwnd_ground, vwnd_lcb, vwnd_mcb, vwnd_hcb, vwnd_ground

print('finished')

# Some plots to assess cross correlation
# intention is to build a cross correlation plot. With x variables, we would have an x-by-x set of subplots.
plot_var_names = ['air', 'pr_wtr', 'pres', 'tcdc', 'wind_direction_ground', 'wind_direction_lcb', 'wind_direction_mcb',
                  'wind_direction_hcb', 'wind_speed_ground', 'wind_speed_lcb', 'wind_speed_mcb', 'wind_speed_hcb']
axes_names = ['T', 'w', 'p', 'cld', 'dir_g', 'dir_l', 'dir_m', 'dir_h', 'u_g', 'u_l', 'u_m', 'u_h']
N = len(plot_var_names)
# create n-by-n subplot
for n in range(N):
    for m in range(N):
        plt.subplot(N + 1, N + 1, (N + 1) * (n + 1) + (m + 1))
        if n == m:
            plt.hist(globals()[plot_var_names[n]])
        else:
            plt.scatter(globals()[plot_var_names[n]], globals()[plot_var_names[m]], marker=(1, 1), alpha=0.2,)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(axes_names[m])
        plt.ylabel(axes_names[n])
plt.show()