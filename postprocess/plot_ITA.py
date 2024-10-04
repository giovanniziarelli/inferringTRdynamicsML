import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

n_tests = 10
plot_susceptible_train = 0
plot_exposed_train = 0
plot_infected_train = 0
plot_beta_train = 0

plot_susceptible_testg = 0
plot_exposed_testg = 0
plot_infected_testg = 0
plot_beta_testg = 1

uncertainty = '0_1'

S_train    = []
E_train    = []
I_train    = []
beta_train = []

S_testg    = []
E_testg    = []
I_testg    = []
beta_testg = []


output_folder = '/home/giovanni/Desktop/LDNets/pprocess_images/uncertainty_' + str(uncertainty) +'/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

output_folder_train = output_folder + 'train/'
if not os.path.exists(output_folder_train):
    os.mkdir(output_folder_train)
output_folder_testg = output_folder + 'testg/'
if not os.path.exists(output_folder_testg):
    os.mkdir(output_folder_testg)

os.mkdir(output_folder_train) 
os.mkdir(output_folder_testg)

for i in range(n_tests):
    folder_name_train = '/home/giovanni/Desktop/LDNets/neurons_4_56_days_random_20_lay2_noised_' + uncertainty + '_' + str(i+1) + '/train/'
    folder_name_testg = '/home/giovanni/Desktop/LDNets/neurons_4_56_days_random_20_lay2_noised_' + uncertainty + '_' + str(i+1) + '/testg/'
    if i == 0:
        S_real    = np.loadtxt(folder_name_train + 'S_real.txt')
        E_real    = np.loadtxt(folder_name_train + 'I_real.txt')
        I_real    = np.loadtxt(folder_name_train + 'R_real.txt')
        beta_real = np.loadtxt(folder_name_train + 'beta_real.txt')

    S_train.append(np.loadtxt(folder_name_train + 'S_rec_train.txt'))    
    E_train.append(np.loadtxt(folder_name_train + 'I_rec_train.txt'))    
    I_train.append(np.loadtxt(folder_name_train + 'R_rec_train.txt'))   

    S_testg.append(np.loadtxt(folder_name_testg + 'S_rec_testg.txt'))    
    E_testg.append(np.loadtxt(folder_name_testg + 'I_rec_testg.txt'))    
    I_testg.append(np.loadtxt(folder_name_testg + 'R_rec_testg.txt'))   

    beta_train.append(np.loadtxt(folder_name_train + 'beta_rec_train.txt'))
    beta_testg.append(np.loadtxt(folder_name_testg + 'beta_rec_testg.txt'))

S_3d_train    = np.array(S_train)
E_3d_train    = np.array(E_train)
I_3d_train    = np.array(I_train)
beta_3d_train = np.array(beta_train)

median_S_train    = np.median(S_3d_train, axis = 0)
median_E_train    = np.median(E_3d_train, axis = 0)
median_I_train    = np.median(I_3d_train, axis = 0)
median_beta_train = np.median(beta_3d_train, axis = 0)

q_value_min = 0.1
q_value_max = 0.9

quantile_0_05_S_train    = np.quantile(S_3d_train, q_value_min, axis = 0)
quantile_0_05_E_train    = np.quantile(E_3d_train, q_value_min, axis = 0)
quantile_0_05_I_train    = np.quantile(I_3d_train, q_value_min, axis = 0)
quantile_0_05_beta_train = np.quantile(beta_3d_train, q_value_min, axis = 0)

quantile_0_95_S_train    = np.quantile(S_3d_train, q_value_max, axis = 0)
quantile_0_95_E_train    = np.quantile(E_3d_train, q_value_max, axis = 0)
quantile_0_95_I_train    = np.quantile(I_3d_train, q_value_max, axis = 0)
quantile_0_95_beta_train = np.quantile(beta_3d_train, q_value_max, axis = 0)

S_3d_testg    = np.array(S_testg)
E_3d_testg    = np.array(E_testg)
I_3d_testg    = np.array(I_testg)
beta_3d_testg = np.array(beta_testg)

median_S_testg    = np.median(S_3d_testg, axis = 0)
median_E_testg    = np.median(E_3d_testg, axis = 0)
median_I_testg    = np.median(I_3d_testg, axis = 0)
median_beta_testg = np.median(beta_3d_testg, axis = 0)

quantile_0_05_S_testg    = np.quantile(S_3d_testg, q_value_min, axis = 0)
quantile_0_05_E_testg    = np.quantile(E_3d_testg, q_value_min, axis = 0)
quantile_0_05_I_testg    = np.quantile(I_3d_testg, q_value_min, axis = 0)
quantile_0_05_beta_testg = np.quantile(beta_3d_testg, q_value_min, axis = 0)

quantile_0_95_S_testg    = np.quantile(S_3d_testg, q_value_max, axis = 0)
quantile_0_95_E_testg    = np.quantile(E_3d_testg, q_value_max, axis = 0)
quantile_0_95_I_testg    = np.quantile(I_3d_testg, q_value_max, axis = 0)
quantile_0_95_beta_testg = np.quantile(beta_3d_testg, q_value_max, axis = 0)

Tfin = 56
t = np.linspace(0,Tfin, E_real.shape[1])

highlight_color = 'green'
color_real = 'black'
color_median = 'blue'
color_area = 'lightblue'
T_obs = 14

width_pixels  = 337
height_pixels = 266

# Desired DPI
dpi = 150

# Calculate figure size in inches
width_in_inches  = width_pixels / dpi
height_in_inches = height_pixels / dpi

# Create the figure with the calculated size
label_quantiles = "{:.2f}".format(q_value_min) + '-' + "{:.2f}".format(q_value_max) + ' quantiles'
if plot_susceptible_train:
    for i in range(S_real.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,S_real[i, :], '--', alpha = 0.6, color = color_real, label = 'Real points', linewidth=3)
        plt.plot(t,median_S_train[i, :], label = 'Median', color = color_median, linewidth=3)
        plt.fill_between(t, quantile_0_05_S_train[i,:], quantile_0_95_S_train[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.legend()
        plt.xlim([0, Tfin])
        plt.savefig(output_folder_train + 'susc_' + str(i+1) +'_train.pdf', format='pdf')
        plt.show()
if plot_exposed_train:
    for i in range(E_real.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,E_real[i, :], '--', alpha = 0.6, color = color_real, label = 'Real points', linewidth=3)
        plt.plot(t,median_E_train[i, :], label = 'Median', color = color_median, linewidth=3)
        plt.fill_between(t, quantile_0_05_E_train[i,:], quantile_0_95_E_train[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.legend()
        plt.xlim([0, Tfin])
        plt.savefig(output_folder_train + 'exp_' + str(i+1) +'_train.pdf', format='pdf')
        plt.show()
if plot_infected_train:
    for i in range(I_real.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,I_real[i, :], '--', alpha = 0.6, color = color_real, label = 'Real points', linewidth=3)
        plt.plot(t,median_I_train[i, :], label = 'Median', color = color_median, linewidth=3)
        plt.fill_between(t, quantile_0_05_I_train[i,:], quantile_0_95_I_train[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.legend()
        plt.xlim([0, Tfin])
        plt.savefig(output_folder_train + 'inf_' + str(i+1) +'_train.pdf', format='pdf')
        plt.show()
if plot_beta_train:
    for i in range(I_real.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,beta_real[i, :], '--', alpha = 0.6, color = color_real, label = 'Real points', linewidth=3)
        plt.plot(t,median_beta_train[i, :], label = 'Median', color = color_median, linewidth=3)
        plt.fill_between(t, quantile_0_05_beta_train[i,:], quantile_0_95_beta_train[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.legend()
        plt.xlim([0, Tfin])
        plt.savefig(output_folder_train + 'beta_' + str(i+1) +'_train.pdf', format='pdf')
        plt.show()

if plot_susceptible_testg:
    for i in range(S_real.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,S_real[i, :], '--', alpha = 0.6, color = color_real, label = 'Real points', linewidth=3)
        plt.plot(t,median_S_testg[i, :], label = 'Median', color = color_median, linewidth=3)
        plt.fill_between(t, quantile_0_05_S_testg[i,:], quantile_0_95_S_testg[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.axvspan(0, T_obs, facecolor=highlight_color, alpha=0.1)
        plt.legend()
        plt.xlim([0, Tfin])
        plt.savefig(output_folder_testg + 'susc_' + str(i+1) +'_testg.pdf', format='pdf')
        plt.show()
if plot_exposed_testg:
    for i in range(E_real.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,E_real[i, :], '--', alpha = 0.6, color = color_real, label = 'Real points', linewidth=3)
        plt.plot(t,median_E_testg[i, :], label = 'Median', color = color_median, linewidth=3)
        plt.fill_between(t, quantile_0_05_E_testg[i,:], quantile_0_95_E_testg[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.axvspan(0, T_obs, facecolor=highlight_color, alpha=0.1)
        plt.legend()
        plt.xlim([0, Tfin])
        plt.savefig(output_folder_testg + 'exp_' + str(i+1) +'_testg.pdf', format='pdf')
        plt.show()
if plot_infected_testg:
    for i in range(I_real.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,I_real[i, :], '--', alpha = 0.6, color = color_real, label = 'Real points', linewidth=3)
        plt.plot(t,median_I_testg[i, :], label = 'Median', color = color_median, linewidth=3)
        plt.fill_between(t, quantile_0_05_I_testg[i,:], quantile_0_95_I_testg[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.axvspan(0, T_obs, facecolor=highlight_color, alpha=0.1)
        plt.legend()
        plt.xlim([0, Tfin])
        plt.savefig(output_folder_testg + 'inf_' + str(i+1) +'_testg.pdf', format='pdf')
        plt.show()
if plot_beta_testg:
    for i in range(I_real.shape[0]):
        fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
        plt.plot(t,beta_real[i, :], '--', alpha = 0.6, color = color_real, label = 'Real points', linewidth=3)
        plt.plot(t,median_beta_testg[i, :], label = 'Median', color = color_median, linewidth=3)
        plt.fill_between(t, quantile_0_05_beta_testg[i,:], quantile_0_95_beta_testg[i,:], color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
        plt.axvspan(0, T_obs, facecolor=highlight_color, alpha=0.1)
        plt.legend()
        plt.xlim([0, Tfin])
        plt.savefig(output_folder_testg + 'beta_' + str(i+1) +'_testg.pdf', format='pdf')
        plt.show()



