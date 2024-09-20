import pandas as pd

df_list = []

# Lista città di interesse.
cities = ['Torino', 'Genova', 'Milano', 'Trento', 'Venezia', 'Trieste', 'Bologna', 'Firenze', 'Ancona', 'Roma', 'Bari', 'Napoli', 'Catanzaro', 'Palermo', 'Cagliari']
# Lista aree cità.
areas = [28.663, 5.416, 23.864, 13.607, 18.391, 7.855, 22.446, 31.456, 9.401, 28.001, 29.349, 18.033, 15.080, 25.832, 24.090]

# Dataset con tutte le date e ne sommo una a una.
date_range = pd.date_range(start='2010-01-01', end='2020-12-31')
df_all = pd.DataFrame(date_range, columns=['DATA'])
df_all['DATA'] = df_all['DATA'].astype(str)

# Inizializzo colonne temperatura.
df_all['TMEDIA'] = 0.0
df_all['TMIN'] = 0.0
df_all['TMAX'] = 0.0
df_all['UMID'] = 0.0

# Dataframe date-aree.
areas_list = pd.DataFrame(date_range, columns=['DATA'])
areas_list['AREAS'] = 0.0

# Loop sulle città.
for i in range(len(cities)):
    df_list.append(pd.read_csv(cities[i] + '_concatenated.csv', sep=';'))
    df_list[i]['DATA'] = df_list[i]['DATA'].astype(str)
    df_list[i]['TMEDIA'] = df_list[i]['TMEDIA'].astype(float)
    df_list[i]['TMIN'] = df_list[i]['TMIN'].astype(float)
    df_list[i]['TMAX'] = df_list[i]['TMAX'].astype(float)
    df_list[i]['UMID'] = df_list[i]['UMID'].astype(float)
    
    # Estraggo date condivise.
    com_dates = df_all['DATA'].isin(df_list[i]['DATA'])
    
    # Somma cumulata delle aree.
    areas_list.loc[com_dates, 'AREAS'] += areas[i]
    
    # Somma pesata temperature.
    df_all.loc[com_dates, 'TMEDIA'] += areas[i] * df_list[i].loc[df_list[i]['DATA'].isin(df_all['DATA']), 'TMEDIA'].values
    df_all.loc[com_dates, 'TMIN'] += areas[i] * df_list[i].loc[df_list[i]['DATA'].isin(df_all['DATA']), 'TMIN'].values
    df_all.loc[com_dates, 'TMAX'] += areas[i] * df_list[i].loc[df_list[i]['DATA'].isin(df_all['DATA']), 'TMAX'].values
    df_all.loc[com_dates, 'UMID'] += areas[i] * df_list[i].loc[df_list[i]['DATA'].isin(df_all['DATA']), 'UMID'].values

# Medie pesate temperature.
df_all.loc[:, 'TMEDIA'] /= areas_list.loc[:, 'AREAS']
df_all.loc[:, 'TMIN']   /= areas_list.loc[:, 'AREAS']
df_all.loc[:, 'TMAX']   /= areas_list.loc[:, 'AREAS']
df_all.loc[:, 'UMID']   /= areas_list.loc[:, 'AREAS']

# CSV
df_all.to_csv('national_temps.csv', index=False, sep=';', quotechar='"', quoting=1)
df_all.to_pickle('national_temps.pkl')
