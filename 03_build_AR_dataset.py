#!/usr/bin/python3
# conda activate r62GEDI

import pandas as pd
import numpy as np
import os
import sys
import xarray as xr
from datetime import datetime 
from datetime import date, timedelta
import copy

sys.path.append('/path/to/functions/')
from TIME import create_list_of_dates, generate_leap_years, date_from_day_of_year, create_datetime_index
from REMOTE_SENSING_FUNCTIONS import find_extraction_coordinates
from STANDARD_FUNCTIONS import runcmd, write_pickle

Import_NCEP = False
scaling_factor = 0.08
aggregation_method = 'sum'
NORM_METRICS = True

data_path = 'path/to/where/you/are/storing/project/data'

################ FUNCTIONS ################################################

def apply_influence_function(scaling_factor, breakup_day_of_year, sub_df):
    IVT_df = sub_df[sub_df['KIVT'] != 0.0]
    t_values = np.arange(breakup_day_of_year)
    # func = scaling_factor**(t_values - breakup_day_of_year)
    func = np.exp(-scaling_factor*(t_values - breakup_day_of_year))-1
    # Function includes the specific heat of water as constant and number of seconds/day
    output = IVT_df.apply(lambda row: func[int(row['day_of_year'] - 1)] * row['KIVT'] * row['tmax (deg c)']*(4186 * 86400), axis=1)
    return output, func

def record_missing_years(df):
    missing_years = []
    for yr in df.groupby(df.index.year):
        yr_df = yr[1]
        if yr_df['Boolean_Breakup'].sum() == 0:
            missing_years.append(yr[0])
        else:
            pass
    return missing_years

def get_kivt(ar_dataset, lat_index, lon_index):
    
    # retrieving kivt data from ar_dataset
    kivtx = np.squeeze(ar_dataset['kivtx'].to_numpy())
    kivty = np.squeeze(ar_dataset['kivty'].to_numpy())
    kivt = np.sqrt((kivtx**2) + (kivty**2))
    del kivtx, kivty

    shapemap = np.squeeze(ar_dataset['shapemap'].to_numpy())
    shapemap = shapemap[:, lat_index, lon_index]

    shapemap[np.isnan(shapemap)] = 0.0

    vals = []
    for idx, i in enumerate(shapemap):
        # subtracting 1 fixes the problem but I'm not 100% sure if its correct
        x = kivt[idx, int(i)-1]
        vals.append(x)

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.0
    
    return vals

def add_missing_day(df, missing_month_day, aggregation_method = 'mean'):

	DF = copy.copy(daymet_data)
	missing_month = missing_month_day[0]
	missing_day = missing_month_day[1]

	years = DF.index.year.unique()

	for year in years:
		if datetime(year, missing_month, missing_day) in DF[DF.index.year == year].index:
			pass
		else:
			before = np.array(DF.loc[datetime(year, missing_month, missing_day) - timedelta(days=1)])
			after = np.array(DF.loc[datetime(year, missing_month, missing_day) + timedelta(days=1)])
			before_after = np.stack([before, after])

			if aggregation_method.upper() == 'MEAN':
				new_row = np.nanmean(before_after, axis = 0)

			elif aggregation_method.upper() == 'MAX':
				new_row = np.nanmax(before_after, axis = 0)

			elif aggregation_method.upper() == 'MIN':
				new_row = np.nanmin(before_after, axis = 0)

			DF.loc[datetime(year, missing_month, missing_day)] = new_row

	DF.sort_index(ascending=True, inplace = True)
	return DF

def get_metrics(NORM_METRICS, aggregation_method, df):
	years = np.array(df.index.year.unique())
	metrics = []
	day_of_year = []
	number_of_ARs_per_year = []

	for year in years:
		sub_df = df[df.index.year == year]
		if len(sub_df[sub_df['Boolean_Breakup'] == 1]) == 0:
			metrics.append(np.nan)
			continue

		breakup_index = sub_df[sub_df['Boolean_Breakup'] == 1].index[0]
		sub_df = sub_df.loc[:breakup_index]
		breakup_day_of_year = sub_df.shape[0]
		number_of_ARs = len(sub_df[sub_df['KIVT'] > 0.0])
		number_of_ARs_per_year.append(number_of_ARs)
		day_of_year.append(breakup_day_of_year)

		func_values = apply_threshold_function(year, sub_df, month_day)
		
		if aggregation_method.upper() == 'SUM':
			agg = func_values.sum()
			metrics.append(agg)
		elif aggregation_method.upper() == 'MEAN':
			agg = func_values.mean()
			metrics.append(agg)
		elif aggregation_method.upper() == 'MAX':
			agg = func_values.max()
			metrics.append(agg)

	metrics = np.array(metrics)
	if NORM_METRICS:
		metrics = (metrics - np.nanmin(metrics)) / (np.nanmax(metrics) - np.nanmin(metrics))
		number_of_ARs_per_year = np.array(number_of_ARs_per_year)
		day_of_year = np.array(day_of_year)
		
	return metrics, number_of_ARs_per_year, day_of_year

#################################################################################

# os.chdir('/mnt/locutus/remotesensing/r62/river_ice_breakup/atmospheric_rivers/NCEP_Temp_data')
# for year in range(1948, 2021+1):
# 	runcmd(f'wget https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis2/pressure/air.{year}.nc')

save_dict = {}

daymet_directory = f'{data_path}/Daymet_26_locations'
break_up_data = pd.read_pickle(f'{data_path}/01_BREAK_UP_DATA_WEBSCRAPED.pkl')
locations_data = pd.read_csv(f'{data_path}/Locations_Meta_Data_40_Sites.csv')
ar_dataset = xr.open_dataset(f'{data_path}/globalARcatalog_NCEP-NCAR_1948-2021_v3.0.nc')

dfs = []

for index, row in locations_data.iterrows():
	SITE = row['Site']
	RIVER = row['River']
	
	lat = float(row.Latitude)
	lon = float(row.Longitude)

	lon, lat, _, _ = find_extraction_coordinates(ar_dataset, [lon % 360, lat, lon % 360, lat], 0)
	lat_index = np.where(ar_dataset.lat.to_numpy() == lat)[0][0]
	lon_index = np.where(ar_dataset.lon.to_numpy() == lon)[0][0]

	# extracting shape variable data from our lat lon coordinates:
	shapemap = np.squeeze(ar_dataset['shapemap'].to_numpy())
	shapemap = shapemap[:, lat_index, lon_index]

	# extracting IVT from AR dataset using lat lon from location in question:
	kivt = get_kivt(ar_dataset=ar_dataset, lat_index=lat_index, lon_index=lon_index)

	# Bringing in the data from the river location in question
	site_breakups = break_up_data[break_up_data.Site == f'{row.Site} {row.River} River']
	site_breakups = site_breakups[(site_breakups.Year >= 1948) & (site_breakups.Year < 2022)]

	# aggregate the kivt data since it was given in 6 hour intervals not daily
	kivt_agg = np.nanmax(kivt.reshape(-1, 4), axis = 1)

	# Create corresponding time axis given the start and end dates of the ar_dataset
	list_of_dates = create_list_of_dates(date(1948, 1, 1), date(2021, 12, 31))

	# notice they are now the same length
	print('DO THE TIME ARRAY LENGTHS AGREE?', len(list_of_dates) == len(kivt_agg))

	df = pd.DataFrame()

	# Creating date information
	df.index = list_of_dates
	df.index = pd.to_datetime(df.index)
	df['day_of_year'] = df.index.dayofyear

	kivt_agg[np.isnan(kivt_agg)] = 0
	df['KIVT'] = kivt_agg
	df['KIVT'] = df['KIVT'].astype('float32')

	df['Boolean_Breakup'] = 0
	df.loc[np.array(site_breakups['Breakup Date']), 'Boolean_Breakup'] = 1
	df['Boolean_Breakup'] = df['Boolean_Breakup'].astype('int16')

	# removind KIVT data from the missing years
	missing_years = record_missing_years(df)
	df.loc[df.index.year.isin(missing_years), 'KIVT'] = 0.0
	
	site = SITE.replace(' ', '_')
	river = RIVER.replace(' ', '_')
	
	if os.path.exists(f'{data_path}/{site}_{river}.pkl'):
		print('DAYMET DATA AVAILABLE FOR SITE', SITE)
		daymet_data = pd.read_pickle(f'{data_path}/{site}_{river}.pkl')

		# make sure the calendars agree
		# Daymet reads in with 365 days/year (no leap years)
		daymet_data = pd.read_pickle(f'{data_path}/{site}_{river}.pkl')
		# create a datetime index using the year and yday columns
		daymet_data = create_datetime_index(daymet_data, 'year', 'yday', 'Date_IDX', True)
		# Since Daymet only goes to the 365 day of the year last day of the year is missing
		# this code will impute it!
		daymet_data = add_missing_day(daymet_data, (12, 31), aggregation_method = 'mean')
		# create a list of calendar dates from 1980-01-01 to 2022-12-31
		# and compare to this new index:
		daymet_dates_list = create_list_of_dates(date(1980, 1, 1), date(2022, 12, 31))
		print('DOES DAYMET DATETIME INDEX HAVE ALL DATES NO LEAP YEAR DAYS MISSING:')
		print(all(daymet_data.index == pd.to_datetime(daymet_dates_list)))

		df = pd.merge(df, daymet_data, left_index=True, right_index=True)
		df['day_of_year'] = df.index.dayofyear
		
	else:
		print('DAYMET DATA NOT AVAILABLE FOR SITE', row.Site)
		continue
	
	df['Location'] = f'{site}_{river}'
	
	metrics, number_of_ARs_per_year, day_of_year = get_metrics(NORM_METRICS, aggregation_method, df)
	
	save_dict[f'{site}_{river}'] = [number_of_ARs_per_year, day_of_year, metrics]
	dfs.append(df)
	
dfs = pd.concat(dfs)
dfs.to_pickle(f'{data_path}/daymet_26_locations_with_IVT.pkl')
write_pickle(f'{data_path}/post_function_results.pkl', save_dict)
