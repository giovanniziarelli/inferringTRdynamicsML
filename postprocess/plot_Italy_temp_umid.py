import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})

n_tests = 20 
num_train = 9
undetection_coef = 0.23
gamma = 1/1.2

#[3 8 4 9 2 6 0 1 5 7]
#ITA  (senza rd_2)
order_season_train = [4, 1, 5, 0, 7, 2, 3, 6, 9]
order_season_testg = [8]

#ITA 2 (con rd_2)
#order_season_train = [3, 8, 4, 9, 2, 6, 0, 1, 5]
#order_season_testg = [7]

#ITA 4 (con rd_4)
#order_season_train = [8, 5, 0, 2, 1, 9, 7, 3, 6]
#order_season_testg = [4]

#7
seasons = [r'2010-2011', r'2011-2012',r'2012-2013',r'2013-2014',r'2014-2015',r'2015-2016',r'2016-2017',r'2017-2018',r'2018-2019',r'2019-2020']
seasons_r = ['2010-2011', '2011-2012','2012-2013','2013-2014','2014-2015','2015-2016','2016-2017','2017-2018','2018-2019','2019-2020']

plot_susceptible_train = 0
plot_exposed_train = 0
plot_infected_train = 1
plot_beta_train = 1
plot_cases_train = 1

plot_susceptible_testg = 0
plot_exposed_testg = 0
plot_infected_testg = 1
plot_beta_testg = 1
plot_cases_testg = 1

S_train     = []
E_train     = []
I_train     = []
beta_train  = []
cases_train = []
rt_train    = []
delta_train = []

S_testg     = []
E_testg     = []
I_testg     = []
beta_testg  = []
cases_testg = []
rt_testg    = []
delta_testg = []

output_folder = '/home/giovanni/Desktop/LDNets/pprocess_images/ita_temp_umid/'#niente#_2#_4
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

output_folder_train = output_folder + 'train/'
if not os.path.exists(output_folder_train):
    os.mkdir(output_folder_train)
output_folder_testg = output_folder + 'testg/'
if not os.path.exists(output_folder_testg):
    os.mkdir(output_folder_testg)

#os.mkdir(output_folder_train) 
#os.mkdir(output_folder_testg)

for i in range(n_tests):
    #folder_name_train = '/home/giovanni/Desktop/LDNets/neurons_10_196_ITALY_lay3_MSE_doppia_temp_umidity_Tobs_29_trial_'+str(i+1)+'/train/'
    #folder_name_testg = '/home/giovanni/Desktop/LDNets/neurons_10_196_ITALY_lay3_MSE_doppia_temp_umidity_Tobs_29_trial_'+str(i+1)+'/testg/' 
    folder_name_train = '/home/giovanni/Desktop/LDNets/try_neurons_4_196_lay2_umidity_Tobs_50_trialsim'+str(i+1)+'/train/'#niente#_rd_2#_rd_4
    folder_name_testg = '/home/giovanni/Desktop/LDNets/try_neurons_4_196_lay2_umidity_Tobs_50_trialsim'+str(i+1)+'/testg/'#niente#_rd_2#_rd_4 
    if i == 0:
        cases_real_train = np.loadtxt(folder_name_train + 'cases_train.txt')
        cases_real_testg = undetection_coef * cases_real_train[num_train:]
        cases_real_train = undetection_coef * cases_real_train[:num_train]
        
    if np.any(np.isnan(np.loadtxt(folder_name_train + 'S_rec_train.txt'))):
        pass
    else:
        S = np.loadtxt(folder_name_train + 'S_rec_train.txt')
        E = np.loadtxt(folder_name_train + 'E_rec_train.txt')
        S_train.append(S)    
        E_train.append(E)    
        I_train.append(np.loadtxt(folder_name_train + 'I_rec_train.txt'))   
        delta_train.append(np.loadtxt(folder_name_train + 'delta_train.txt'))

        beta_train.append(np.loadtxt(folder_name_train + 'beta_rec_train.txt'))
        S_week = S[:,::7*2]
        E_week = E[:,::7*2]
        cases_estim = S_week[:,:-1] + E_week[:,:-1] - S_week[:,1:] - E_week[:,1:]
        #cases_train.append(np.loadtxt(folder_name_train + 'cases_train.txt'))
        cases_train.append(undetection_coef * cases_estim)
        rt_train.append(beta_train[-1] * S_train[-1] / gamma)
        delta_testg.append(np.loadtxt(folder_name_testg + 'delta_testg.txt'))

    if np.any(np.isnan(np.loadtxt(folder_name_testg + 'S_rec_testg.txt'))):
        pass
    else:
        S = np.loadtxt(folder_name_testg + 'S_rec_testg.txt')
        E = np.loadtxt(folder_name_testg + 'E_rec_testg.txt')
        S_testg.append(S)    
        E_testg.append(E)    
        I_testg.append(np.loadtxt(folder_name_testg + 'I_rec_testg.txt'))   

        beta_testg.append(np.loadtxt(folder_name_testg + 'beta_rec_testg.txt'))
        S_week = S[::7*2][None,:]
        E_week = E[::7*2][None,:]
        cases_estim = S_week[:,:-1] + E_week[:,:-1] - S_week[:,1:] - E_week[:,1:]
        #cases_train.append(np.loadtxt(folder_name_train + 'cases_train.txt'))
        cases_testg.append(undetection_coef * cases_estim)
        rt_testg.append(beta_testg[-1] * S_testg[-1] / gamma)

S_3d_train     = np.array(S_train)
E_3d_train     = np.array(E_train)
I_3d_train     = np.array(I_train)
beta_3d_train  = np.array(beta_train)
cases_3d_train = np.array(cases_train)
rt_3d_train    = np.array(rt_train)
delta_3d_train = np.array(delta_train)

median_S_train    = np.median(S_3d_train, axis = 0)
median_E_train    = np.median(E_3d_train, axis = 0)
median_I_train    = np.median(I_3d_train, axis = 0)
median_beta_train = np.median(beta_3d_train, axis = 0)
median_cases_train = np.median(cases_3d_train, axis = 0)
median_rt_train   = np.median(rt_3d_train, axis = 0)
median_delta_train = np.median(delta_3d_train, axis = 0)

q_value_min = 0.2
q_value_max = 0.8

quantile_0_05_S_train    = np.quantile(S_3d_train, q_value_min, axis = 0)
quantile_0_05_E_train    = np.quantile(E_3d_train, q_value_min, axis = 0)
quantile_0_05_I_train    = np.quantile(I_3d_train, q_value_min, axis = 0)
quantile_0_05_beta_train = np.quantile(beta_3d_train, q_value_min, axis = 0)
quantile_0_05_cases_train = np.quantile(cases_3d_train, q_value_min, axis = 0)
quantile_0_05_rt_train = np.quantile(rt_3d_train, q_value_min, axis = 0)
quantile_0_05_delta_train = np.quantile(delta_3d_train, q_value_min, axis = 0)

quantile_0_95_S_train    = np.quantile(S_3d_train, q_value_max, axis = 0)
quantile_0_95_E_train    = np.quantile(E_3d_train, q_value_max, axis = 0)
quantile_0_95_I_train    = np.quantile(I_3d_train, q_value_max, axis = 0)
quantile_0_95_beta_train = np.quantile(beta_3d_train, q_value_max, axis = 0)
quantile_0_95_cases_train = np.quantile(cases_3d_train, q_value_max, axis = 0)
quantile_0_95_rt_train = np.quantile(rt_3d_train, q_value_max, axis = 0)
quantile_0_95_delta_train = np.quantile(delta_3d_train, q_value_max, axis = 0)

S_3d_testg    = np.array(S_testg)
E_3d_testg    = np.array(E_testg)
I_3d_testg    = np.array(I_testg)
beta_3d_testg = np.array(beta_testg)
cases_3d_testg = np.array(cases_testg)
rt_3d_testg = np.array(rt_testg)
delta_3d_testg = np.array(delta_testg)

median_S_testg    = np.median(S_3d_testg, axis = 0)
median_E_testg    = np.median(E_3d_testg, axis = 0)
median_I_testg    = np.median(I_3d_testg, axis = 0)
median_beta_testg = np.median(beta_3d_testg, axis = 0)
median_cases_testg = np.median(cases_3d_testg, axis = 0)
median_rt_testg = np.median(rt_3d_testg, axis = 0)
median_delta_testg = np.median(delta_3d_testg, axis = 0)

quantile_0_05_S_testg    = np.quantile(S_3d_testg, q_value_min, axis = 0)
quantile_0_05_E_testg    = np.quantile(E_3d_testg, q_value_min, axis = 0)
quantile_0_05_I_testg    = np.quantile(I_3d_testg, q_value_min, axis = 0)
quantile_0_05_beta_testg = np.quantile(beta_3d_testg, q_value_min, axis = 0)
quantile_0_05_cases_testg = np.quantile(cases_3d_testg, q_value_min, axis = 0)
quantile_0_05_rt_testg = np.quantile(rt_3d_testg, q_value_min, axis = 0)
quantile_0_05_delta_testg = np.quantile(delta_3d_testg, q_value_min, axis = 0)

quantile_0_95_S_testg    = np.quantile(S_3d_testg, q_value_max, axis = 0)
quantile_0_95_E_testg    = np.quantile(E_3d_testg, q_value_max, axis = 0)
quantile_0_95_I_testg    = np.quantile(I_3d_testg, q_value_max, axis = 0)
quantile_0_95_beta_testg = np.quantile(beta_3d_testg, q_value_max, axis = 0)
quantile_0_95_cases_testg = np.quantile(cases_3d_testg, q_value_max, axis = 0)
quantile_0_95_rt_testg = np.quantile(rt_3d_testg, q_value_max, axis = 0)
quantile_0_95_delta_testg = np.quantile(delta_3d_testg, q_value_max, axis = 0)

Tfin = 196 
t = np.linspace(0,Tfin, median_E_train.shape[1])

T_obs = 50 
highlight_color = 'green'
color_real = 'black'
color_median = 'red'
color_area = 'salmon'
color_median_inf = 'gold'
color_area_inf = 'khaki'
color_median_rt = 'darkviolet'
color_area_rt = 'plum'
color_median_tr = 'blue'
color_area_tr = 'lightblue'

width_pixels  = 320#600#337
height_pixels = 200#500#266

# Desired DPI
dpi = 100

# Calculate figure size in inches
width_in_inches  = width_pixels / dpi
height_in_inches = height_pixels / dpi

# Create the figure with the calculated size
label_quantiles = "{:.2f}".format(0.10) + '-' + "{:.2f}".format(0.90) + ' quantiles'
if plot_susceptible_train:
    for i in range(median_S_train.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,median_S_train[i, :], label = 'Median', color = color_median, linewidth=1.5)
        plt.fill_between(t, quantile_0_05_S_train[i,:], quantile_0_95_S_train[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        #plt.legend()
        plt.xlim([0, Tfin])
        plt.xlabel(r'days')
        plt.ylabel(r'Susceptible')
        plt.title(seasons[order_season_train[i]])
        plt.savefig(output_folder_train + 'susc_' + seasons_r[order_season_train[i]] +'_train.pdf', format='pdf', bbox_inches='tight')
        plt.close()
if plot_exposed_train:
    for i in range(median_E_train.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,median_E_train[i, :], label = 'Median', color = color_median, linewidth=1.5)
        plt.fill_between(t, quantile_0_05_E_train[i,:], quantile_0_95_E_train[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        #plt.legend()
        plt.xlim([0, Tfin])
        plt.ylabel(r'Exposed')
        plt.xlabel(r'days')
        plt.title(seasons[order_season_train[i]])
        plt.savefig(output_folder_train + 'exp_' + seasons_r[order_season_train[i]] +'_train.pdf', format='pdf', bbox_inches='tight')
        plt.close()
if plot_infected_train:
    for i in range(median_I_train.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,median_I_train[i, :], label = 'Median', color = color_median_inf, linewidth=1.5)
        plt.fill_between(t, quantile_0_05_I_train[i,:], quantile_0_95_I_train[i,:], color = color_area_inf, alpha=0.5, label=label_quantiles, linewidth=0)
        #plt.legend()
        plt.xlim([0, Tfin])
        plt.ylabel(r'Infected')
        plt.xlabel(r'days')
        plt.title(seasons[order_season_train[i]])
        plt.savefig(output_folder_train + 'inf_' + seasons_r[order_season_train[i]] +'_train.pdf', format='pdf', bbox_inches='tight')
        plt.close()
if plot_beta_train:
    for i in range(median_I_train.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,median_beta_train[i, :], label = 'Median', color = color_median_tr, linewidth=1.5)
        plt.fill_between(t, quantile_0_05_beta_train[i,:], quantile_0_95_beta_train[i,:], color = color_area_tr, alpha=0.5, label=label_quantiles, linewidth=0)
        #plt.legend()
        plt.xlim([0, Tfin])
        plt.ylim([0, 1.8])
        plt.ylabel(r'Transmission rate')
        plt.xlabel(r'days')
        plt.title(seasons[order_season_train[i]])
        plt.savefig(output_folder_train + 'beta_' + seasons_r[order_season_train[i]] +'_train.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,median_rt_train[i, :], label = 'Median', color = color_median_rt, linewidth=1.5)
        plt.fill_between(t, quantile_0_05_rt_train[i,:], quantile_0_95_rt_train[i,:], color = color_area_rt, alpha=0.5, label=label_quantiles, linewidth=0)
        #plt.legend()
        plt.xlim([0, Tfin])
        plt.ylim([0, 1.3])
        plt.ylabel(r'Reproduction number')
        plt.xlabel(r'days')
        plt.title(seasons[order_season_train[i]])
        plt.savefig(output_folder_train + 'rt_' + seasons_r[order_season_train[i]] +'_train.pdf', format='pdf', bbox_inches='tight')
        plt.close()


if plot_cases_train:
    for i in range(median_I_train.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(np.arange(1, cases_real_train.shape[1]+1), cases_real_train[i, :], 'o', alpha = 0.5, color = color_real, label = 'Real points', linewidth=1.5, markersize=2.5)
        plt.plot(np.arange(1, cases_real_train.shape[1]+1),median_cases_train[i, :], '-o', label = 'Median', color = color_median, linewidth=1.5, markersize=2.5)
        plt.fill_between(np.arange(1, cases_real_train.shape[1]+1), quantile_0_05_cases_train[i,:], quantile_0_95_cases_train[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        #plt.legend()
        plt.xlim([0, cases_real_train.shape[1]+1])
        plt.ylim([0, 0.016])
        plt.ylabel(r'Cases')
        plt.xlabel(r'weeks')
        plt.title(seasons[order_season_train[i]])
        plt.savefig(output_folder_train + 'cases_' + seasons_r[order_season_train[i]] +'_train.pdf', format='pdf', bbox_inches='tight')
        plt.close()

if plot_susceptible_testg:
    for i in range(1):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,median_S_testg[:], label = 'Median', color = color_median, linewidth=1.5)
        plt.fill_between(t, quantile_0_05_S_testg[:], quantile_0_95_S_testg[:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.axvspan(0, T_obs, facecolor=highlight_color, alpha=0.1)
        #plt.legend()
        plt.xlim([0, Tfin])
        plt.ylabel(r'Susceptible')
        plt.xlabel(r'days')
        plt.title(seasons[order_season_testg[i]])
        plt.savefig(output_folder_testg + 'susc_' + seasons_r[order_season_testg[i]] +'_testg.pdf', format='pdf', bbox_inches='tight')
        plt.close()
if plot_exposed_testg:
    for i in range(1):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,median_E_testg[:], label = 'Median', color = color_median, linewidth=1.5)
        plt.fill_between(t, quantile_0_05_E_testg[:], quantile_0_95_E_testg[:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.axvspan(0, T_obs, facecolor=highlight_color, alpha=0.1)
        #plt.legend()
        plt.xlim([0, Tfin])
        plt.ylabel(r'Exposed')
        plt.xlabel(r'days')
        plt.title(seasons[order_season_testg[i]])
        plt.savefig(output_folder_testg + 'exp_' + seasons_r[order_season_testg[i]] +'_testg.pdf', format='pdf', bbox_inches='tight')
        plt.close()
if plot_infected_testg:
    for i in range(1):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,median_I_testg[:], label = 'Median', color = color_median_inf, linewidth=1.5)
        plt.fill_between(t, quantile_0_05_I_testg[:], quantile_0_95_I_testg[:], color = color_area_inf, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.axvspan(0, T_obs, facecolor=highlight_color, alpha=0.1)
        #plt.legend()
        plt.xlim([0, Tfin])
        plt.ylabel(r'Infected')
        plt.xlabel(r'days')
        plt.title(seasons[order_season_testg[i]])
        plt.savefig(output_folder_testg + 'inf_' + seasons_r[order_season_testg[i]] +'_testg.pdf', format='pdf', bbox_inches='tight')
        plt.close()
if plot_beta_testg:
    for i in range(1):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,median_beta_testg[:], label = 'Median', color = color_median_tr, linewidth=1.5)
        plt.fill_between(t, quantile_0_05_beta_testg[:], quantile_0_95_beta_testg[:], color = color_area_tr, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.axvspan(0, T_obs, facecolor=highlight_color, alpha=0.1)
        #plt.legend()
        plt.xlim([0, Tfin])
        plt.ylim([0, 1.8])
        plt.ylabel(r'Transmission rate')
        plt.xlabel(r'days')
        plt.title(seasons[order_season_testg[i]])
        plt.savefig(output_folder_testg + 'beta_' + seasons_r[order_season_testg[i]] +'_testg.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,median_rt_testg[:], label = 'Median', color = color_median_rt, linewidth=1.5)
        plt.fill_between(t, quantile_0_05_rt_testg[:], quantile_0_95_rt_testg[:], color = color_area_rt, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.axvspan(0, T_obs, facecolor=highlight_color, alpha=0.1)
        #plt.legend()
        plt.xlim([0, Tfin])
        plt.ylim([0, 1.3])
        plt.ylabel(r'Reproduction number')
        plt.xlabel(r'days')
        plt.title(seasons[order_season_testg[i]])
        plt.savefig(output_folder_testg + 'rt_' + seasons_r[order_season_testg[i]] +'_testg.pdf', format='pdf', bbox_inches='tight')
        plt.close()

if plot_cases_testg:
    for i in range(1):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(np.arange(1, cases_real_testg.shape[1]+1), cases_real_testg.squeeze(), 'o', alpha = 0.5, color = color_real, label = 'Real points', linewidth=1.5, markersize=2.5)
        plt.plot(np.arange(1, cases_real_testg.shape[1]+1),median_cases_testg[0, :], '-o', label = 'Median', color = color_median, linewidth=1.5, markersize=2.5)
        plt.fill_between(np.arange(1, cases_real_testg.shape[1]+1), quantile_0_05_cases_testg[0,:], quantile_0_95_cases_testg[0,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.axvspan(0, int(T_obs/7), facecolor=highlight_color, alpha=0.1)
        #plt.legend()
        plt.xlim([0, cases_real_testg.shape[1]+1])
        plt.ylim([0, 0.016])
        plt.ylabel(r'Cases')
        plt.xlabel(r'weeks')
        plt.title(seasons[order_season_testg[i]])
        plt.savefig(output_folder_testg + 'cases_' + seasons_r[order_season_testg[i]] +'_testg.pdf', format='pdf', bbox_inches='tight')
        plt.close()

#da Trentini A/H1N1, A/H3N2, B
reff_per_year = np.array([[1.29, 0, 1.28], [0, 1.31, 0], [1.11, 0, 1.33], [1.1, 1.11, 0], [1.14, 1.14, 1.1], [1.08, 1.09, 1.25],[0, 1.16, 0], [1.41, 0, 1.27], [1.22, 1.19, 0], [1.11, 1.14, 1.23]])
perc_per_year = np.array([[0.609, 0.041, 0.353], [0.04, 0.87, 0.126], [0.186, 0.038, 0.776], [0.307, 0.693, 0.0], [0.43, 0.35, 0.22], [0.122, 0.169, 0.709], [0.004, 0.944, 0.052],[0.372, 0.023, 0.605], [0.5, 0.5, 0], [0.268, 0.402, 0.33]])
from cycler import cycler
# Set the color cycle to the "Accent" palette
colors = plt.cm.tab20c.colors
plt.rc('axes', prop_cycle=cycler(color=colors))
print(quantile_0_05_delta_train)
reff_perc = np.sum(reff_per_year * perc_per_year, axis = 1)
errors_train = np.vstack(((median_delta_train - quantile_0_05_delta_train ) * np.ones_like(reff_perc[order_season_train]), (quantile_0_95_delta_train - median_delta_train) * np.ones_like(reff_perc[order_season_train])))
errors_testg = np.vstack(((median_delta_testg - quantile_0_05_delta_testg ) * np.ones_like(reff_perc[order_season_testg]), (quantile_0_95_delta_testg - median_delta_testg) * np.ones_like(reff_perc[order_season_testg])))
p = np.polyfit(reff_perc[order_season_train], median_delta_train,1)
ev_x = np.linspace(min(reff_perc), max(reff_perc), 1000)
evals = np.polyval(p, ev_x)
fig = plt.figure(figsize=(1.9*width_in_inches, 2*height_in_inches), dpi=dpi)
col = np.array(order_season_train)/10
cmap = plt.get_cmap('tab10')  # Use any colormap you prefer
col = cmap(np.linspace(0, 1, 10))
#plt.scatter(reff_perc[order_season_train], median_delta_train,c=col, cmap = 'tab10')
plt.errorbar(reff_perc[order_season_train], median_delta_train, errors_train, fmt = 'o',color='k', markersize=0.0001, alpha=0.2 )
for i in range(9):
    plt.scatter(reff_perc[order_season_train[i]], median_delta_train[i],s=100,c=col[order_season_train[i]], label=seasons[order_season_train[i]])#, labels=seasons[order_season_train])
plt.plot(ev_x, evals)
print(reff_perc[order_season_testg])
print(median_delta_testg)
plt.errorbar(reff_perc[order_season_testg], median_delta_testg, errors_testg, fmt = 'o',color='k', markersize=0.0001, alpha=0.2 )
plt.scatter(reff_perc[order_season_testg], np.array(median_delta_testg).reshape((1,)), s=100,c='white',edgecolors = 'k', marker='*', label=seasons[order_season_testg[0]])#, labels=seasons[order_season_testg])
#plt.legend(frameon=False)
plt.xlabel(r'Weighted effective reproduction number')
plt.ylabel(r'Reconstructed lineage parameter')
handles, labels = plt.gca().get_legend_handles_labels()
print(handles)
print(labels)
handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: list(map(str,labels))[k])] )
plt.legend(handles, labels, frameon=False, fontsize='small')
plt.savefig(output_folder_train + 'deltas.pdf', format='pdf', bbox_inches='tight')
plt.show()
print('coeff_cor lin')
print(np.corrcoef(reff_perc[order_season_train], median_delta_train))
