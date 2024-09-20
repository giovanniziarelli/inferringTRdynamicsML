import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.constrained_layout.use"] = True
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})


width_pixels  = 300#600#337
height_pixels = 200#500#266

# Desired DPI
dpi = 100

# Calculate figure size in inches
width_in_inches  = width_pixels / dpi
height_in_inches = height_pixels / dpi


def add_days(data_str, n):
    """
    Aggiunge N giorni a una data specificata in formato yyyy-mm-dd.

    Parameters:
    data_str (str): La data in formato yyyy-mm-dd.
    n (int): Il numero di giorni da aggiungere.

    Returns:
    str: La nuova data in formato yyyy-mm-dd.
    """
    data = datetime.strptime(data_str, "%Y-%m-%d")
    nuova_data = data + timedelta(days=n)
    return nuova_data.strftime("%Y-%m-%d")

def add_year(data_str):
    """
    Aggiunge un anno a una data specificata in formato yyyy-mm-dd.

    Parameters:
    data_str (str): La data in formato yyyy-mm-dd.

    Returns:
    str: La nuova data in formato yyyy-mm-dd.
    """
    data = datetime.strptime(data_str, "%Y-%m-%d")
    try:
        nuova_data = data.replace(year=data.year + 1)
    except ValueError:
        # Questo gestisce il caso del 29 febbraio in un anno bisestile
        nuova_data = data.replace(year=data.year + 1, day=28)

    return nuova_data.strftime("%Y-%m-%d")

def lista_date(data_str, n):
    """
    Restituisce una lista di date comprese tra la data iniziale e la data più n giorni.

    Parameters:
    data_str (str): La data in formato yyyy-mm-dd.
    n (int): Il numero di giorni da aggiungere.

    Returns:
    list: Lista di date in formato yyyy-mm-dd.
    """
    data_iniziale = datetime.strptime(data_str, "%Y-%m-%d")
    lista_di_date = []

    for i in range(n + 1):
        nuova_data = data_iniziale + timedelta(days=i)
        lista_di_date.append(nuova_data.strftime("%Y-%m-%d"))

    return lista_di_date

df = pd.read_csv('national_temps_umid.csv', sep = ';')
print(df)

T_cut = 196#49#98
T_wave = 196
n_per_season = int(T_wave / T_cut)
n_years =10
file_path_temp = 'tmedia_national_length_missing_' + str(T_cut) + '.csv'
file_path_umid = 'umid_national_length_missing_' + str(T_cut) + '.csv'

data_min  = '2010-10-21'
data_list = [data_min]
for i in range(n_years):
    for j in range(n_per_season-1):
        data_list.append(add_days(data_list[-1], T_cut))
    if i != n_years - 1:
        data_min = add_year(data_min)
        data_list.append(data_min)
print(data_list)

df['DATA'] = df['DATA'].astype(str)
df.set_index('DATA')
tmedia = np.zeros((len(data_list), T_cut+1))
umid   = np.zeros((len(data_list), T_cut+1))
for i in range(len(data_list)):
    lista_data = lista_date(data_list[i], T_cut)
    dates = df['DATA'].isin(lista_data)
    print(lista_data)
    tmedia[i, :] = df.loc[dates, 'TMEDIA'].values
    umid[i, :]   = df.loc[dates, 'UMID'].values

np.savetxt(file_path_temp, tmedia)
np.savetxt(file_path_umid, umid)
directory_img = '/home/giovanni/Desktop/LDNets/italian-temperatures/img/'
if not os.path.exists(directory_img):
    os.mkdir(directory_img)
directory_img = os.path.join(directory_img, 'tmedia_umid_national_length' + str(T_cut) + '/')
if not os.path.exists(directory_img):
    os.mkdir(directory_img)

seasons = [r'2010-2011', r'2011-2012',r'2012-2013',r'2013-2014',r'2014-2015',r'2015-2016',r'2016-2017',r'2017-2018',r'2018-2019',r'2019-2020']
seasons_r = ['2010-2011', '2011-2012','2012-2013','2013-2014','2014-2015','2015-2016','2016-2017','2017-2018','2018-2019','2019-2020']

for i, tm in enumerate(tmedia):
    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    lista_data = lista_date(data_list[i], T_cut)
    plt.plot(tm, linewidth = 1.5, color='navy')
    plt.ylabel(r'Temperature [°C]')
    plt.xticks(ticks=np.arange(0, len(lista_data), 56), labels=lista_data[::56], rotation=45)
    # Salvataggio del grafico come PDF
    plt.savefig(os.path.join(directory_img, f'temp_'+seasons_r[i]+'.pdf'), format='pdf')
    # Chiudere la figura per liberare memoria
    plt.close()

plt.figure(figsize=(3*width_in_inches, 3*height_in_inches), dpi=dpi)
for i, tm in enumerate(tmedia):
    plt.plot(tm, linewidth = 1.5, label=seasons_r[i])
plt.legend()
plt.xticks(ticks=np.arange(0, len(lista_data), 56), labels=lista_data[::56], rotation=45)
    # Salvataggio del grafico come PDF
plt.savefig(os.path.join(directory_img, f'temp_all.pdf'), format='pdf')
    # Chiudere la figura per liberare memoria
plt.close()


for i, um in enumerate(umid):
    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    lista_data = lista_date(data_list[i], T_cut)
    plt.plot(um, linewidth = 1.5, color='peru')
    plt.ylabel(r'Relative Umidity [\%]')
    plt.xticks(ticks=np.arange(0, len(lista_data), 56), labels=lista_data[::56], rotation=45)
    # Salvataggio del grafico come PDF
    plt.savefig(os.path.join(directory_img, f'umid_'+seasons_r[i]+'.pdf'), format='pdf')
    # Chiudere la figura per liberare memoria
    plt.close()

plt.figure(figsize=(3*width_in_inches, 3*height_in_inches), dpi=dpi)
for i, um in enumerate(umid):
    plt.plot(um, linewidth = 1.5, label=seasons_r[i])
plt.legend()
plt.xticks(ticks=np.arange(0, len(lista_data), 56), labels=lista_data[::56], rotation=45)
    # Salvataggio del grafico come PDF
plt.savefig(os.path.join(directory_img, f'umid_all.pdf'), format='pdf')
# Chiudere la figura per liberare memoria
plt.close()
