#!/usr/bin/python3
# conda activate r62GEDI
# mpiexec -n 3 python3 AR_Analysis.py

import os
import sys
import numpy as np
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import copy
from datetime import datetime
from datetime import date, timedelta
from scipy.stats import shapiro, ranksums, ttest_ind, ttest_rel, pearsonr, spearmanr

sys.path.append('/home/r62/repos/russ_repos/Functions/')
sys.path.append('/home/r62/repos/russ_repos/river_ice_breakup/')

from STANDARD_FUNCTIONS import runcmd, create_directory, write_pickle
from STATISTICS import summary_statistics
from REMOTE_SENSING_FUNCTIONS import find_extraction_coordinates
from TIME import create_list_of_dates
from RIVER_ICE_FUNCTIONS import get_missing_years
from VISUALIZATION_FUNCTIONS import Color_Palettes

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

############ PARAMETERS ##########################################

aggregation_method = 'sum'
method = 'exponential'
start_date_previous_year_month_day = (9, 15)

# hyperparameter tuning
scaling_factors = np.arange(0.06, 0.30+0.02, 0.02) # sigmas for normal distribution
sigma = sigmas = list(range(5, 45+5, 5))
center_of_biases = np.array([(11, 12), (11, 19), (11, 26), (12, 3), (12, 10),
							 (12, 17), (12, 24), (1, 1), (1, 8), (1, 15),
							 (1, 22), (1, 29), (2, 5), (2, 12)])
temp_thresholds = list(range(-20, 10+5, 2))
vars_to_measure = ['prcp (mm/day)', 'KIVT', 'prcp - IVT']
var_to_measure = vars_to_measure[rank]

###################################################################


color_names_hex = [
    '#0000FF',  # Blue
    '#008000',  # Green
    '#FF0000',  # Red
    '#00FFFF',  # Cyan
    '#FF00FF',  # Magenta
    '#FFFF00',  # Yellow
    '#000000',  # Black
    '#FFFFFF',  # White
    '#808080',  # Gray
    '#A52A2A',  # Brown
    '#808000',  # Olive
    '#FF7F50',  # Coral
    '#008080',  # Teal
    '#E6E6FA',  # Lavender
    '#D2B48C',  # Tan
    '#FA8072',  # Salmon
    '#800000',  # Maroon
    '#708090',  # Slategray
    '#006400',  # Darkgreen
    '#DA70D6',  # Orchid
    '#FFD700',  # Gold
    '#4B0082',  # Indigo
    '#FFDAB9',  # Peach
    '#CCCCFF',  # Periwinkle
    '#40E0D0'   # Turquoise
]

def generate_leap_years(start = 0, end = 3000):
    leap_years = []
    for year in range(start, end+1):
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            leap_years.append(year)
    return leap_years

def date_from_day_of_year(year, day_of_year):
    # Create a datetime object for January 1st of the given year
    base_date = datetime(year, 1, 1)
    # Add the number of days to the base date
    target_date = base_date + timedelta(days=day_of_year - 1)  # Subtract 1 because day_of_year is 1-based
    return target_date

def create_datetime_index(df, year_col_name, day_of_year_col_name, new_index_name='Date_IDX', drop=False):
    
    df[new_index_name] = df.apply(lambda row: date_from_day_of_year(int(row[year_col_name]), int(row[day_of_year_col_name])), axis=1)
    df[new_index_name] = pd.to_datetime(df[new_index_name], format='%Y-%m-%d')
    df.set_index(new_index_name, inplace=True)
    df.index = pd.to_datetime(df.index)
    if drop:
        df.drop(columns=[year_col_name, day_of_year_col_name], inplace=True, axis = 1)
    
    return df

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

def apply_threshold_function(sub_df, temp_threshold):
    
    IVT_df = sub_df[(sub_df['KIVT'] != 0.0) & (sub_df['tmin (deg c)'] <= temp_threshold)]
    # t_values = np.arange(breakup_day_of_year)
    output = np.array(IVT_df['KIVT']* IVT_df['tmin (deg c)'] * 4186 * 86400)
    return output

def apply_exponential_influence_function(sub_df, scaling_factor, breakup_doy, center_of_bias_date, IVT_Precip, dampening_parameter=100000000):
    
    center_bias_n = sub_df.loc[:center_of_bias_date].shape[0]
    t_values = np.arange(len(sub_df))
    t_values = t_values - center_bias_n
    
    right_data = sub_df.loc[center_of_bias_date:breakup_index]
    left_data = sub_df.loc[:center_of_bias_date]
        
    right_t_values = t_values[t_values > 0]
    left_t_values = t_values[t_values <= 0]
    
    right_skew_func = ((np.exp(-1*scaling_factor*((right_t_values) - breakup_doy)))-1)/dampening_parameter
    left_skew_func = ((np.exp(-1*scaling_factor*(-1*(left_t_values) - breakup_doy)))-1)/dampening_parameter   
    
    func = np.concatenate([left_skew_func, right_skew_func])
    sub_df['func'] = func
    
    IVT_df = sub_df[sub_df[IVT_Precip] != 0.0]
    output = IVT_df.apply(lambda row: row['func'] * row[IVT_Precip] * row['tmin (deg c)'] * (4186 * 86400), axis=1)

    return output, func, sub_df.index

def apply_normal_influence_function(sub_df, center_of_bias_date, sigma):

    mu = sub_df.index.get_loc(center_of_bias_date)
    
    # create list from 0 - breakup day for the domain of normal function
    t_values = np.arange(len(sub_df))
    sub_df['day_number'] = t_values
    
    # only want the days that have IVT
    IVT_df = sub_df[sub_df['KIVT'] != 0.0]
    
    # normal fuctions applied to days with IVT
    func = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (t_values - mu)**2 / (2 * sigma**2))
    output = IVT_df.apply(lambda row: func[int(row['day_number'])] *
                          row['KIVT'] * (row['tmin (deg c)']) * (4186 * 86400), axis=1)
    
    return output, func, sub_df.index
	
class Convert_coordinates():
    
    def __init__(self, arr):
        self.arr = np.array(arr)
        
    def convert_to_positive_scale(self):
        return self.arr % 360

    def convert_to_negative_scale(self):
        return ((self.arr - 180) % 360) - 180

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


def does_ar_happen_before_breakup(df, window):
    
    breakup_indices = df[df['Boolean_Breakup'] == 1].index
    mask = []
    IVT_data = []
    
    for i in breakup_indices:
        start = i - pd.Timedelta(days = window)
        ar_result = np.array(df.loc[start:i, 'KIVT'])

        if ar_result.sum() > 0:
            mask.append(1)
            IVT_data.append(np.array(df.loc[start:i, 'KIVT']).sum())

        else:
            mask.append(0)
            IVT_data.append(0.0)

    mask = np.array(mask).astype('bool')
    IVT_data = np.array(IVT_data)

    return IVT_data, mask

def get_pvalues_for_temp(df):

	window_range = list(range(1, 15)) # days
	alpha = 0.05
	pvalues_tmin, pvalues_tmax = [], []
	
	avgs_by_window = []
	
	for window in window_range:
		IVT_indices = df.index[df['KIVT'] != 0.0]
		avg_before_after, indices = [], []

		for index in IVT_indices:
			rows_before = df.loc[index - pd.Timedelta(window, "d") : index - pd.Timedelta(1, "d"), ['tmin (deg c)', 'tmax (deg c)', 'KIVT']]
			rows_after = df.loc[index:index + pd.Timedelta(window, "d"), ['tmin (deg c)', 'tmax (deg c)', 'KIVT']]

			rows = pd.concat([rows_before, rows_after], axis = 0)

			# maybe remove the AR events at the end because the 
			# time series cannot go to a full window
			if len(rows) != (window*2)+1:
				continue

			indices.append(index)
			before_avgs = np.array(rows.iloc[0:window, :].mean())
			after_avgs = np.array(rows.iloc[window+1:, :].mean())

			data = np.concatenate([before_avgs, after_avgs])

			avg_before_after.append(data)

		avg_before_after = np.stack(avg_before_after)
		avg_before_after = pd.DataFrame(avg_before_after)
		avg_before_after.columns = ['tmin_before', 'tmax_before', 'KIVT_before', 'tmin_after', 'tmax_after', 'KIVT_after']
		avg_before_after.index = indices
		
		# I want to save each DF by window
		avgs_by_window.append(avg_before_after)
		
		# less asks: is a < b on average?
		t_statistic_tmin, p_value_tmin = ttest_rel(a = avg_before_after.tmin_before, b = avg_before_after.tmin_after, alternative = 'less')
		t_statistic_tmax, p_value_tmax = ttest_rel(a = avg_before_after.tmax_before, b = avg_before_after.tmax_after, alternative = 'less')

		pvalues_tmin.append(p_value_tmin)
		pvalues_tmax.append(p_value_tmax)
	
	return pvalues_tmin, pvalues_tmax, avgs_by_window
	
daymet_directory = '/mnt/locutus/remotesensing/r62/river_ice_breakup/Daymet_25_locations'
break_up_data = pd.read_pickle('/mnt/locutus/remotesensing/r62/river_ice_breakup/Breakup_Data/01_BREAK_UP_DATA_WEBSCRAPED.pkl')
locations_data = pd.read_csv('/mnt/locutus/remotesensing/r62/river_ice_breakup/Breakup_Data/CMIP6_analysis/Locations_Meta_Data_40_Sites.csv')
ar_dataset = xr.open_dataset('/mnt/locutus/remotesensing/r62/river_ice_breakup/atmospheric_rivers/NCEP-NCAR/globalARcatalog_NCEP-NCAR_1948-2021_v3.0.nc')

for FILE in os.listdir(daymet_directory):
	
	FILE = FILE[:-10]
	RIVER = FILE.split('_')[-1]
	RIVER = FILE.split('_')[-1]
	SITE = FILE.split('_')[:-1]
	SITE = (' ').join(SITE)
	
	daymet_data = pd.read_pickle(f'{daymet_directory}/{FILE}_River.pkl')

	# make sure the calendars agree
	# Daymet reads in with 365 days/year (no leap years)
	# create a datetime index using the year and yday columns
	daymet_data = create_datetime_index(daymet_data, 'year', 'yday', 'Date_IDX', True)
	# Since Daymet only goes to the 365 day of the year last day of the year is missing
	# this code will impute it!
	daymet_data = add_missing_day(daymet_data, (12, 31), aggregation_method = 'mean')
	# create a list of calendar dates from 1980-01-01 to 2022-12-31
	# and compare to this new index:
	daymet_dates_list = create_list_of_dates(date(1980, 1, 1), date(2022, 12, 31))

	# separating out site information
	location_info = locations_data[(locations_data.Site == SITE) & (locations_data.River == f'{RIVER} River')]

	lat = float(location_info.Latitude)
	lon = float(location_info.Longitude)

	lon, lat, _, _ = find_extraction_coordinates(ar_dataset, [lon % 360, lat, lon % 360, lat], 0)
	lat_index = np.where(ar_dataset.lat.to_numpy() == lat)[0][0]
	lon_index = np.where(ar_dataset.lon.to_numpy() == lon)[0][0]

	# extracting shape variable data from our lat lon coordinates:
	shapemap = np.squeeze(ar_dataset['shapemap'].to_numpy())
	shapemap = shapemap[:, lat_index, lon_index]

	# extracting IVT from AR dataset using lat lon from location in question:
	kivt = get_kivt(ar_dataset=ar_dataset, lat_index=lat_index, lon_index=lon_index)
	
	# an example where T1 (188) is equal to a an AR LAT = 10
	# shapemap[188]

	# Bringing in the data from the river location in question
	site_breakups = break_up_data[break_up_data.Site == f'{SITE} {RIVER} River']
	site_breakups = site_breakups[(site_breakups.Year >= 1948) & (site_breakups.Year < 2022)]

	# aggregate the kivt data since it was given in 6 hour intervals not daily
	kivt_agg = np.nanmax(kivt.reshape(-1, 4), axis = 1)

	# Create corresponding time axis given the start and end dates of the ar_dataset
	list_of_dates = create_list_of_dates(date(1948, 1, 1), date(2021, 12, 31))

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

	df = pd.merge(df, daymet_data, left_index=True, right_index=True)
	# day of year must be reset since daymet data is slightly off due to leap years:
	df['day_of_year'] = df.index.dayofyear
	
	# I am going to create another column of precip - AR IVT
	precip = np.array(df['prcp (mm/day)'])
	mask = np.array(df['KIVT'] > 0.0).astype('bool')
	precip[mask] = 0.0
	df['prcp - IVT'] = precip
		
	hp_space, best_pearson_metrics, best_spearman_metrics = [], [], []

	for center_of_bias_month_day in center_of_biases:

		# Exponential bias function
		for SIGMA in scaling_factors: # sigma == scaling factor
			hp_space.append([SIGMA, center_of_bias_month_day])
			years = np.array(df.index.year.unique())
			metrics, day_of_year, number_of_ARs_per_year, OUTPUTS, breakup_dates = [], [], [], [], []

			for idx, year in enumerate(years):
				if idx == 0:
					continue

				start_date = f'{year-1}-{start_date_previous_year_month_day[0]}-{start_date_previous_year_month_day[1]}'
				sub_df = df[df.index.year.isin([year-1, year])].loc[start_date:]

				if len(sub_df[sub_df['Boolean_Breakup'] == 1]) == 0:
					continue

				# breakup date as date and recording it for plots later
				breakup_index = sub_df[sub_df['Boolean_Breakup'] == 1].index[0]
				breakup_dates.append(breakup_index)

				# This says if the center of bias is greater than the month of the breakup
				# date, then we are centering the bias in the previous year
				if center_of_bias_month_day[0] > breakup_index.month:
					center_year = year - 1
				else:
					center_year = year

				# center of bias as a date
				center_of_bias_date = f'{center_year}-{center_of_bias_month_day[0]}-{center_of_bias_month_day[1]}'

				# Record the number of ARs for each year
				number_of_ARs = len(sub_df[sub_df['KIVT'] > 0.0])
				number_of_ARs_per_year.append(number_of_ARs)

				# Record the breakup doy for each year
				breakup_doy = breakup_index.timetuple().tm_yday
				day_of_year.append(breakup_doy)

				output, func_values, func_dates = apply_exponential_influence_function(sub_df=sub_df,
														   scaling_factor=SIGMA,
														   breakup_doy=breakup_doy,
														   IVT_Precip = var_to_measure,
														   center_of_bias_date=center_of_bias_date)



				if aggregation_method.upper() == 'SUM':
					agg = output.sum()
					metrics.append(agg)
				elif aggregation_method.upper() == 'MEAN':
					agg = output.mean()
					metrics.append(agg)
				elif aggregation_method.upper() == 'MAX':
					agg = output.max()
					metrics.append(agg)

				OUTPUTS.append(output)

			metrics = np.array(metrics)
			number_of_ARs_per_year = np.array(number_of_ARs_per_year)
			day_of_year = np.array(day_of_year)

			pearson_metrics = pearsonr(day_of_year, metrics)
			# spearman_metrics = spearmanr(day_of_year, metrics)						

			best_pearson_metrics.append(np.array(pearson_metrics))
			# best_spearman_metrics.append(np.array(spearman_metrics))

	# Get the best results
	pearson_idx = np.argmin(np.stack(best_pearson_metrics)[:, 0])
	# spearman_idx = np.argmin(np.stack(best_spearman_metrics)[:, 0])
			
	best_pearson_hps = hp_space[pearson_idx]
	# best_spearman_hps = hp_space[spearman_idx]
	
	center_of_bias_month_day = best_pearson_hps[1]
	
	metrics = []
	for idx, year in enumerate(years):
		if idx == 0:
			continue

		start_date = f'{year-1}-{start_date_previous_year_month_day[0]}-{start_date_previous_year_month_day[1]}'
		sub_df = df[df.index.year.isin([year-1, year])].loc[start_date:]

		if len(sub_df[sub_df['Boolean_Breakup'] == 1]) == 0:
			continue

		# breakup date as date and recording it for plots later
		breakup_index = sub_df[sub_df['Boolean_Breakup'] == 1].index[0]

		# This says if the center of bias is greater than the month of the breakup
		# date, then we are centering the bias in the previous year
		if center_of_bias_month_day[0] > breakup_index.month:
			center_year = year - 1
		else:
			center_year = year

		# center of bias as a date
		center_of_bias_date = f'{center_year}-{center_of_bias_month_day[0]}-{center_of_bias_month_day[1]}'

		# Record the number of ARs for each year
		number_of_ARs = len(sub_df[sub_df['KIVT'] > 0.0])

		# Record the breakup doy for each year
		breakup_doy = breakup_index.timetuple().tm_yday

		output_pearson, func_values_pearson, func_dates_pearson = apply_exponential_influence_function(sub_df=sub_df,
												   scaling_factor=best_pearson_hps[0],
												   breakup_doy=breakup_doy,
												   IVT_Precip = var_to_measure,
												   center_of_bias_date=center_of_bias_date)
		agg = output_pearson.sum()
		metrics.append(agg)
		
	if var_to_measure == 'prcp (mm/day)':
		var_name = 'Precip'

	elif var_to_measure == 'KIVT':
		var_name = 'IVT'

	elif var_to_measure == 'prcp - IVT':
		var_name = 'prcp-IVT'

	pvalues_tmin, pvalues_tmax, avg_before_after = get_pvalues_for_temp(df)
	
	print(f'WRITING OUTPUT FILE FOR {var_name}_{FILE}_River.pkl', flush = True)
	
	pkl_dict = {'metrics' : metrics,
				'day_of_year' : day_of_year, 
				'number_of_ARs_per_year' : number_of_ARs_per_year,
				'pvalues_tmin' : pvalues_tmin,
				'pvalues_tmax' : pvalues_tmax,
			    'best_pearson_hps' : best_pearson_hps,
			    'func_values_pearson' : func_values_pearson,
			    'func_dates' : func_dates,
			    'avg_before_after' : avg_before_after,
			    'df' : df}
	
	write_pickle(f'/mnt/locutus/remotesensing/r62/river_ice_breakup/atmospheric_rivers/Results/outputs/exponential/{var_name}_{FILE}_River.pkl', pkl_dict)