#!/usr/bin/python3
# conda activate r62GEDI

import pandas as pd
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import requests
from datetime import datetime

path_to_breakup_data = input('Enter the absolute path to where you want your data imported to:')
if path_to_breakup_data.endswith('/'):
	path_to_breakup_data = path_to_breakup_data[:-1]

site_numbers = range(461, 494+1) # You can find these from the NWS site
LIST = []
off_dfs = []
off_dfs_idx = []

expected_cols = ['River', 'Location', 'Year', 'Breakup Date', 'Ice Moved', 'First Boat',
                     'Un-safe Person', 'Unsafe Vehicle', 'Last Ice', 'Remarks']

for idx, site in enumerate(site_numbers):

	driver = webdriver.Chrome(executable_path="/usr/lib/chromium-browser/chromedriver")
	driver.get('https://www.weather.gov/aprfc/breakupDB?site={site}'.format(site=site))
	time.sleep(5)

	soup = BeautifulSoup(driver.page_source, 'lxml')
	driver.close()

	tables = soup.find_all('table')
	table = tables[-1]
	dfs = pd.read_html(str(table))
	print('NUMBER OF DFs CREATED', len(dfs))

	if len(dfs[0].columns) == len(expected_cols):
		print('SITE:', str(site), 'IMPORTED CORRECTLY')
		LIST.append(dfs[0])

	else:
		print('SITE:', str(site), '**** NOT IMPORTED CORRECTLY ****')
		off_dfs.append(dfs[0])
		off_dfs_idx.append(idx)
		
driver.quit()
final = pd.concat(LIST)

final = final[final.Year > 1895]

dates = []
unformatted_idx = []

for idx, i in enumerate(final['Breakup Date']):
    try:
        date_object = datetime.strptime(str(i), '%Y-%m-%d').date()
        dates.append(date_object)
    except:
        unformatted_idx.append(idx)
		
final.reset_index(drop=True, inplace=True)
final = final.drop(unformatted_idx)
final['Breakup Date'] = np.array(dates, dtype='datetime64')

days_after_equinox = []
for d in final['Breakup Date']:
    year = d.year
    d = d.date()
    v_equinox = datetime.strptime('{year}-03-21'.format(year=year), '%Y-%m-%d').date()
    diff = d - v_equinox
    days_after_equinox.append(int(diff.days))

final['Days from Equinox'] = np.array(days_after_equinox)
final.sort_values(by=['Year'], inplace = True)
breakup_dates = pd.to_datetime(final['Breakup Date'], errors = 'coerce')

# manual fix of one of the dates
print('One value fails')
print(final.loc[breakup_dates.isna(), 'Breakup Date'])
print()
print('looks like it was meant to be 1933-05-05')
print(breakup_dates.loc[1390:1396])
final.loc[1392, 'Breakup Date'] = '1933-05-05'
final['Breakup Date'] = pd.to_datetime(final['Breakup Date'], errors = 'raise')
final['Site'] = final['Location'] + ' ' + final['River']
final.to_csv(f'{path_to_breakup_data}/01_BREAK_UP_DATA_WEBSCRAPED.csv')
final.to_pickle(f'{path_to_breakup_data}/01_BREAK_UP_DATA_WEBSCRAPED.pkl')



