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


def add_days(date_str, n):
    """
    Add n days to given date
    
    Parameters:
    date_str (str): date string format yyyy-mm-dd.
    n (int): days to add.

    Returns:
    str: format yyyy-mm-dd.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    new_date = date + timedelta(days=n)
    return new_date.strftime("%Y-%m-%d")

def add_year(date_str):
    """
    Add one year to given date
    
    Parameters:
    date_str (str): date string format yyyy-mm-dd.

    Returns:
    str: format yyyy-mm-dd.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    try:
        new_date = date.replace(year=date.year + 1)
    except ValueError:
        # 29th february in leap years
        new_date = date.replace(year=date.year + 1, day=28)

    return new_date.strftime("%Y-%m-%d")

def list_date(date_str, n):
    """
    List of dates betwwn date_str and date_str + n (days)

    Parameters:
    date_str (str): date string format yyyy-mm-dd.
    n (int): days to add.

    Returns:
    list: dates in format yyyy-mm-dd.
    
    """
    initial_date = datetime.strptime(date_str, "%Y-%m-%d")
    dates_list = []

    for i in range(n + 1):
        new_date = initial_date + timedelta(days=i)
        dates_list.append(new_date.strftime("%Y-%m-%d"))

    return dates_list

df = pd.read_csv('national_temps_humid.csv', sep = ';')

T_cut = 196
T_wave = 196
n_per_season = int(T_wave / T_cut)
n_years =10
file_path_temp = 'tmean_national_length_' + str(T_cut) + '.csv'
file_path_humid = 'humid_national_length_' + str(T_cut) + '.csv'

date_min  = '2010-10-21'
date_list = [date_min]
for i in range(n_years):
    for j in range(n_per_season-1):
        date_list.append(add_days(date_list[-1], T_cut))
    if i != n_years - 1:
        date_min = add_year(date_min)
        date_list.append(date_min)

df['DATA'] = df['DATA'].astype(str)
df.set_index('DATA')
tmean = np.zeros((len(date_list), T_cut+1))
humid   = np.zeros((len(date_list), T_cut+1))

for i in range(len(date_list)):
    l_d = list_date(date_list[i], T_cut)
    dates = df['DATA'].isin(l_d)
    tmean[i, :] = df.loc[dates, 'TMEAN'].values
    humid[i, :]   = df.loc[dates, 'HUMID'].values

np.savetxt(file_path_temp, tmean)
np.savetxt(file_path_humid, humid)
directory_img = 'img/'
if not os.path.exists(directory_img):
    os.mkdir(directory_img)
directory_img = os.path.join(directory_img, 'tmean_humid_national_length' + str(T_cut) + '/')
if not os.path.exists(directory_img):
    os.mkdir(directory_img)

seasons = [r'2010-2011', r'2011-2012',r'2012-2013',r'2013-2014',r'2014-2015',r'2015-2016',r'2016-2017',r'2017-2018',r'2018-2019',r'2019-2020']
seasons_r = ['2010-2011', '2011-2012','2012-2013','2013-2014','2014-2015','2015-2016','2016-2017','2017-2018','2018-2019','2019-2020']

for i, tm in enumerate(tmean):
    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    lista_date = list_date(date_list[i], T_cut)
    plt.plot(tm, linewidth = 1.5, color='navy')
    plt.ylabel(r'Temperature [°C]')
    plt.xticks(ticks=np.arange(0, len(lista_date), 56), labels=lista_date[::56], rotation=45)
    plt.savefig(os.path.join(directory_img, f'temp_'+seasons_r[i]+'.pdf'), format='pdf')
    plt.close()

plt.figure(figsize=(3*width_in_inches, 3*height_in_inches), dpi=dpi)
for i, tm in enumerate(tmean):
    plt.plot(tm, linewidth = 1.5, label=seasons_r[i])
plt.legend()
plt.xticks(ticks=np.arange(0, len(lista_date), 56), labels=lista_date[::56], rotation=45)
plt.savefig(os.path.join(directory_img, f'temp_all.pdf'), format='pdf')
plt.close()


for i, um in enumerate(humid):
    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    lista_date = list_date(date_list[i], T_cut)
    plt.plot(um, linewidth = 1.5, color='peru')
    plt.ylabel(r'Relative Umidity [\%]')
    plt.xticks(ticks=np.arange(0, len(lista_date), 56), labels=lista_date[::56], rotation=45)
    plt.savefig(os.path.join(directory_img, f'humid_'+seasons_r[i]+'.pdf'), format='pdf')
    plt.close()

plt.figure(figsize=(3*width_in_inches, 3*height_in_inches), dpi=dpi)
for i, um in enumerate(humid):
    plt.plot(um, linewidth = 1.5, label=seasons_r[i])
plt.legend()
plt.xticks(ticks=np.arange(0, len(lista_date), 56), labels=lista_date[::56], rotation=45)
plt.savefig(os.path.join(directory_img, f'humid_all.pdf'), format='pdf')
plt.close()
