import pandas as pd
import numpy as np
import datetime

df_list = []

cities = ['Torino', 'Genova', 'Milano', 'Trento', 'Venezia', 'Trieste', 'Bologna', 'Firenze', 'Ancona', 'Roma', 'Bari', 'Napoli', 'Catanzaro', 'Palermo', 'Cagliari']
pop_df = pd.read_csv('pop_it_reg_2010_2020.csv').to_numpy()

# Create dataset and sum
date_range = pd.date_range(start='2010-01-01', end='2020-12-31')
pop = np.zeros((date_range.shape[0], len(cities)))
count_leap = 0
count_non_leap = 0

for i in range(11):
    if date_range[count_leap*366 + count_non_leap*365].year == 2012 or date_range[count_leap*366 + count_non_leap*365].year == 2016 or date_range[count_leap*366 + count_non_leap*365].year == 2020:
        pop[365 * count_non_leap + 366 * count_leap: 365 * count_non_leap + 366 * (count_leap+1), :] = np.repeat(pop_df[i][None,:], 366, axis=0)
        count_leap += 1
    else:
        pop[365 * count_non_leap + 366 * count_leap: 365 * (count_non_leap+1) + 366 * count_leap, :] = np.repeat(pop_df[i][None,:], 365, axis=0)
        count_non_leap += 1

df_all = pd.DataFrame(date_range, columns=['DATA'])
df_all['DATA'] = df_all['DATA'].astype(str)
date_keys = set(df_all.keys())
df_all['TMEAN'] = 0.0 
df_all['TMIN'] = 0.0
df_all['TMAX'] = 0.0
df_all['HUMID'] = 0.0
pop_list = pd.DataFrame(date_range, columns=['DATA'])
pop_list['POPS'] = 0.0

for i in range(len(cities)):
    df_list.append(pd.read_csv(cities[i]+'_concatenated.csv', sep = ';'))
    df_list[i]['DATA'] = df_list[i]['DATA'].astype(str)
    df_list[i]['TMEAN'] = df_list[i]['TMEAN'].astype(float)
    df_list[i]['TMIN'] = df_list[i]['TMIN'].astype(float)
    df_list[i]['TMAX'] = df_list[i]['TMAX'].astype(float)
    df_list[i]['HUMID'] = df_list[i]['HUMID'].astype(float)
for i in range(len(cities)):
    data_keys = set(df_list[i]['DATA'].tolist())
    pop_list.loc[df_all['DATA'].isin(data_keys), 'POPS'] += pop[df_all['DATA'].isin(data_keys),i]
    df_mask = df_list[i]
    df_all.loc[df_all['DATA'].isin(data_keys),'TMEAN'] += pop[df_all['DATA'].isin(data_keys),i] * df_mask['TMEAN']
    df_all.loc[df_all['DATA'].isin(data_keys),'TMIN'] += pop[df_all['DATA'].isin(data_keys),i] * df_mask['TMIN']
    df_all.loc[df_all['DATA'].isin(data_keys),'TMAX'] += pop[df_all['DATA'].isin(data_keys),i] * df_mask['TMAX']
    df_all.loc[df_all['DATA'].isin(data_keys),'HUMID'] += pop[df_all['DATA'].isin(data_keys),i] * df_mask['HUMID']

df_all['TMEAN'] /= pop_list['POPS']
df_all['TMIN'] /= pop_list['POPS']
df_all['TMAX'] /= pop_list['POPS']
df_all['HUMID'] /= pop_list['POPS']
df_all.to_csv('national_temps_humid.csv', index=False, sep=';', quotechar='"', quoting=1)
