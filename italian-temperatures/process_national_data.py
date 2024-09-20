import pandas as pd
import numpy as np
import datetime

df_list = []

cities = ['Torino', 'Genova', 'Milano', 'Trento', 'Venezia', 'Trieste', 'Bologna', 'Firenze', 'Ancona', 'Roma', 'Bari', 'Napoli', 'Catanzaro', 'Palermo', 'Cagliari']
pop_df = pd.read_csv('popolazione_regioni_italiane_2010_2020_completo.csv').to_numpy()#[28.663, 5.416, 23.864, 13.607, 18.391, 7.855, 22.446, 31.456, 9.401, 28.001, 29.349, 18.033, 15.080, 25.832, 24.090]

#creo dataset con tutte le date e ne sommo una a una
date_range = pd.date_range(start='2010-01-01', end='2020-12-31')
pop = np.zeros((date_range.shape[0], len(cities)))
count_bi = 0
count_non_bi = 0
print(date_range.shape)
print(pop.shape)
for i in range(11):
    if date_range[count_bi*366 + count_non_bi*365].year == 2012 or date_range[count_bi*366 + count_non_bi*365].year == 2016 or date_range[count_bi*366 + count_non_bi*365].year == 2020:
        pop[365 * count_non_bi + 366 * count_bi: 365 * count_non_bi + 366 * (count_bi+1), :] = np.repeat(pop_df[i][None,:], 366, axis=0)
        count_bi += 1
    else:
        print('SHAPE')
        print(np.repeat(pop_df[i][None,:], 365, axis=0).shape )
        pop[365 * count_non_bi + 366 * count_bi: 365 * (count_non_bi+1) + 366 * count_bi, :] = np.repeat(pop_df[i][None,:], 365, axis=0)
        count_non_bi += 1

print(date_range.shape)
df_all = pd.DataFrame(date_range, columns=['DATA'])
df_all['DATA'] = df_all['DATA'].astype(str)
date_keys = set(df_all.keys())
df_all['TMEDIA'] = 0.0 
df_all['TMIN'] = 0.0
df_all['TMAX'] = 0.0
df_all['UMID'] = 0.0
pop_list = pd.DataFrame(date_range, columns=['DATA'])
pop_list['POPS'] = 0.0#np.zeros((10,1))

for i in range(len(cities)):
    df_list.append(pd.read_csv(cities[i]+'_concatenated.csv', sep = ';'))
    #df_list[i].set_index('DATA', inplace=True)
    df_list[i]['DATA'] = df_list[i]['DATA'].astype(str)
    df_list[i]['TMEDIA'] = df_list[i]['TMEDIA'].astype(float)
    df_list[i]['TMIN'] = df_list[i]['TMIN'].astype(float)
    df_list[i]['TMAX'] = df_list[i]['TMAX'].astype(float)
    df_list[i]['UMID'] = df_list[i]['UMID'].astype(float)
for i in range(len(cities)):
    #print(str(df_list[i]['DATA']))
    #print(df_list[i].keys())
    print(df_list[i]['DATA'].tolist())
    data_keys = set(df_list[i]['DATA'].tolist())
    print(data_keys)
    #print(df_all['DATA'].astype(str))
    print(len(df_all['DATA'].isin(data_keys)))
    #print(df_all['DATA'])
    pop_list.loc[df_all['DATA'].isin(data_keys), 'POPS'] += pop[df_all['DATA'].isin(data_keys),i]
    print(sum(df_all['DATA'].isin(data_keys)))
    print(pop[df_all['DATA'].isin(data_keys),i])
    df_mask = df_list[i]#[df_all['DATA'].isin(data_keys)]
    df_all.loc[df_all['DATA'].isin(data_keys),'TMEDIA'] += pop[df_all['DATA'].isin(data_keys),i] * df_mask['TMEDIA']
    df_all.loc[df_all['DATA'].isin(data_keys),'TMIN'] += pop[df_all['DATA'].isin(data_keys),i] * df_mask['TMIN']
    df_all.loc[df_all['DATA'].isin(data_keys),'TMAX'] += pop[df_all['DATA'].isin(data_keys),i] * df_mask['TMAX']
    df_all.loc[df_all['DATA'].isin(data_keys),'UMID'] += pop[df_all['DATA'].isin(data_keys),i] * df_mask['UMID']

df_all['TMEDIA'] /= pop_list['POPS']
df_all['TMIN'] /= pop_list['POPS']
df_all['TMAX'] /= pop_list['POPS']
df_all['UMID'] /= pop_list['POPS']
df_all.to_csv('national_temps_umid.csv', index=False, sep=';', quotechar='"', quoting=1)
