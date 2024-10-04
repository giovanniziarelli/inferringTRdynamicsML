import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

sns.set(style="whitegrid")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amssymb}')

n_tests = 20
T_obs_vec = [15,36,57,78,99,120,141]
T_obs_vec_str = ['14','35','56','77','98','119','140']

delta_train_true = np.zeros((49,1))
delta_testg_true = np.zeros((48,1))
delta_train_true = np.loadtxt('deltas_synt/delta_train_real.txt')
delta_testg_true = np.loadtxt('deltas_synt/delta_test_real.txt')
delta_train_mat_l  = np.zeros((49,n_tests, 6))
delta_testg_mat_l  = np.zeros((48,n_tests, 6))
beta_0_testg_mat_l = np.zeros((48, n_tests, 6))

delta_train_mean = np.zeros((49,1))

err_train       = np.zeros((len(T_obs_vec), n_tests))
err_testg       = np.zeros((len(T_obs_vec), n_tests))
err_train_testg = np.zeros((len(T_obs_vec), n_tests))
output_folder = '' # ex '/home/giovanni/Desktop/LDNets/pprocess_images/errs_T_obs_vec/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
for j in range(len(T_obs_vec)):

    for i in range(n_tests):
        folder_name       = '' # ex '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_0_size_50_trialsim_' +str(i+1) + '_T_obs_' +str(T_obs_vec[j])+'/train/'
        folder_name_testg = '' # ex '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_0_size_50_trialsim_' +str(i+1) + '_T_obs_' +str(T_obs_vec[j])+'/testg/'
        if j > 0:
            delta_train_mat_l[:,i,j-1] = np.loadtxt(folder_name + 'delta_train.txt')
            delta_testg_mat_l[:,i,j-1] = np.loadtxt(folder_name_testg + 'delta_testg.txt')
            beta_0_testg_mat_l[:,i,j-1] = np.loadtxt(folder_name_testg + 'beta_real_testg.txt')[:,0]
        if np.any(np.isnan(np.loadtxt(folder_name + 'train_error.txt'))) or len(os.listdir(folder_name)) == 0:
            pass
        else:
            S = np.loadtxt(folder_name + 'S_rec_train.txt')
            E = np.loadtxt(folder_name + 'E_rec_train.txt')
            S_week = S[:,::7*2]
            E_week = E[:,::7*2]
            cases_estim = S_week[:,:-1] + E_week[:,:-1] - S_week[:,1:] - E_week[:,1:]
            cases_real_train = np.loadtxt(folder_name + 'cases_real.txt') 

            cases_train = cases_estim
            err_train[j, i] = np.sqrt(np.mean((cases_real_train - cases_estim)**2))
            

            S = np.loadtxt(folder_name_testg + 'S_rec_testg.txt')
            E = np.loadtxt(folder_name_testg + 'E_rec_testg.txt')
            S_week = S[:,::7*2]
            E_week = E[:,::7*2]
            cases_estim = S_week[:,:-1] + E_week[:,:-1] - S_week[:,1:] - E_week[:,1:]
            cases_real_testg = np.loadtxt(folder_name_testg + 'cases_real_testg.txt') 

            cases_testg = cases_estim
            err_testg[j, i] = np.sqrt(np.mean((cases_real_testg - cases_estim)**2))#np.loadtxt(folder_name + 'testg_error.txt')
            err_train_testg[j, i] = err_testg[j,i] / err_train[j,i] 

q_value_min = 0.1
q_value_max = 0.9
median_err_train = np.zeros((len(T_obs_vec),))
qmin_err_train = np.zeros((len(T_obs_vec),))
qmax_err_train = np.zeros((len(T_obs_vec),))

median_err_testg = np.zeros((len(T_obs_vec),))
qmin_err_testg = np.zeros((len(T_obs_vec),))
qmax_err_testg = np.zeros((len(T_obs_vec),))

median_err_train_testg = np.zeros((len(T_obs_vec),))
qmin_err_train_testg = np.zeros((len(T_obs_vec),))
qmax_err_train_testg = np.zeros((len(T_obs_vec),))

for i in range(len(T_obs_vec)):
    idx = np.argwhere(np.isnan(err_testg[i,:]) == False)
    median_err_train[i] = np.median(err_train[i,idx])
    qmin_err_train[i] = np.quantile(err_train[i, idx], q_value_min)
    qmax_err_train[i] = np.quantile(err_train[i, idx], q_value_max)

    median_err_testg[i] = np.median(err_testg[i,idx])
    qmin_err_testg[i] = np.quantile(err_testg[i, idx], q_value_min)
    qmax_err_testg[i] = np.quantile(err_testg[i, idx], q_value_max)

    median_err_train_testg[i] = np.median(err_train_testg[i,idx])
    qmin_err_train_testg[i] = np.quantile(err_train_testg[i, idx], q_value_min)
    qmax_err_train_testg[i] = np.quantile(err_train_testg[i, idx], q_value_max)

width_pixels  = 600#337
height_pixels = 500#266

# Desired DPI
dpi = 100

# Calculate figure size in inches
width_in_inches  = width_pixels / dpi
height_in_inches = height_pixels / dpi

#color_median = 'blue'
#color_area   = 'lightblue'

label_quantiles = "{:.2f}".format(q_value_min) + '-' + "{:.2f}".format(q_value_max) + ' quantiles'

fig_violin = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
#sns.boxplot(err_testg.T)
lab = T_obs_vec_str*(2*n_tests)

hue_l = ['train'] * (len(T_obs_vec_str)*n_tests)
hue_l.extend(['testg'] * (len(T_obs_vec_str)*n_tests))

err_train_df = np.maximum(err_train.T, 1e-4 * np.ones_like(err_train.T))
err_testg_df = np.maximum(err_testg.T, 9e-4 * np.ones_like(err_testg.T))

data = np.concatenate((err_train_df,err_testg_df), axis = 0)

data_fl = data.flatten()
df = pd.DataFrame({
    'errors': data_fl,
    'time window': lab,
    'datasets': hue_l
})
df = df[df['datasets']=='testg']

sns.set_theme(style="ticks", palette="pastel")

# Load the example tips dataset
# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="time window", y="errors",
            hue="datasets", data=df, linewidth = 1.5)
sns.despine(offset=10, trim=True)

plt.yscale('log')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
plt.xlabel(r'time window', fontsize=15)
plt.ylabel(r'error', fontsize=15)
plt.savefig(output_folder + 'boxplot.pdf', format='pdf', bbox_inches='tight')
plt.show()

titles = [r"$T_{\mathrm{obs}} = 35$",r"$T_{\mathrm{obs}} = 56$",r"$T_{\mathrm{obs}} = 77$",r"$T_{\mathrm{obs}} = 98$",r"$T_{\mathrm{obs}} = 119$",r"$T_{\mathrm{obs}} = 140$" ]

fig_idelta, ax = plt.subplots(2,3, figsize=(3/2*width_in_inches, height_in_inches), dpi=dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amssymb}')

pal = sns.color_palette("pastel",6)
pal.as_hex()
for i in range(2):
    for j in range(3):
        delta_testg_mean = np.zeros((48,))
        beta_0_testg_mean = np.zeros((48,))
        for k  in range(48):
            idx = np.isnan(delta_testg_mat_l[k,:,i*3+j]) 
            delta_testg_mean[k] = np.mean(delta_testg_mat_l[k, idx == False,i*3+j])
            beta_0_testg_mean[k] = np.mean(beta_0_testg_mat_l[k, idx == False,i*3+j])
        sc = ax[i,j].scatter(delta_testg_mean, delta_testg_true.squeeze(), c = beta_0_testg_mean, cmap = 'Accent', alpha = 0.4)
        ax[i,j].set_title(titles[3*i+j])
        if i == 1:
            ax[i,j].set_xlabel(r"$\delta_r$")
        if j == 0:
            ax[i,j].set_ylabel(r"$\delta_t$")
plt.subplots_adjust(top=0.85, bottom=0.2, wspace=0.3, hspace=0.4)
cbar_ax = fig_idelta.add_axes([0.2, 0.05, 0.6, 0.03])
cbar = fig_idelta.colorbar(sc, cax = cbar_ax, orientation = 'horizontal')
cbar.set_label(r'IC for transmission rate ($\beta_0$)')

plt.savefig(output_folder + 'delta_train_color_b0_tobs.pdf', format='pdf', bbox_inches='tight')
plt.show()
