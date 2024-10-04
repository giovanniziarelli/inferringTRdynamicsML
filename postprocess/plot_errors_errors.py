import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
sns.set(style="whitegrid")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

n_tests = 20 
errors_tex = ['0', '$1.0 \cdot 10^{-3}$', '$5.0 \cdot 10^{-3}$', '$1.0 \cdot 10^{-2}$', '$5.0 \cdot 10^{-2}$', '$1.0 \cdot 10^{-1}$']
errors = ['0', '1e-3', '5e-3', '1e-2', '5e-2', '1e-1']
errors_abs = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

err_train       = np.zeros((len(errors), n_tests))
err_testg       = np.zeros((len(errors), n_tests))

delta_train_true = np.zeros((49,1))
delta_testg_true = np.zeros((48,1))
delta_train_true = np.loadtxt('deltas_synt/delta_train_real.txt')
delta_testg_true = np.loadtxt('deltas_synt/delta_test_real.txt')
delta_train_mat_l  = np.zeros((49,n_tests, 6))
beta_0_train_mat_l = np.zeros((49,n_tests, 6))
delta_testg_mat_l  = np.zeros((48,n_tests, 6))
beta_0_testg_mat_l = np.zeros((48, n_tests, 6))

delta_train_mean = np.zeros((49,1))

err_train_testg = np.zeros((len(errors), n_tests))
output_folder = '/home/giovanni/Desktop/LDNets/pprocess_images/errs_errors/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
for j in range(len(errors)):

    for i in range(n_tests):
        folder_name = ''         # ex '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_'+errors[j]+'_size_50_trialsim_' +str(i+1) + '_T_obs_78/train/'
        folder_name_GT =''       # ex '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_0_size_50_trialsim_' +str(i+1) + '_T_obs_78/train/'
        folder_name_testg =''    # ex '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_'+errors[j]+'_size_50_trialsim_' +str(i+1) + '_T_obs_78/testg/'
        folder_name_testg_GT ='' # ex '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_0_size_50_trialsim_' +str(i+1) + '_T_obs_78/testg/'
        
        delta_train_mat_l[:,i,j]  = np.loadtxt(folder_name + 'delta_train.txt')
        beta_0_train_mat_l[:,i,j] = np.loadtxt(folder_name_GT + 'beta_real.txt')[:,0]
        delta_testg_mat_l[:,i,j]  = np.loadtxt(folder_name_testg + 'delta_testg.txt')
        beta_0_testg_mat_l[:,i,j] = np.loadtxt(folder_name_testg_GT + 'beta_real_testg.txt')[:,0]

        if np.any(np.isnan(np.loadtxt(folder_name + 'S_rec_train.txt'))):
            pass
        else:
            S = np.loadtxt(folder_name + 'S_rec_train.txt')
            E = np.loadtxt(folder_name + 'E_rec_train.txt')
            S_week = S[:,::7*2]
            E_week = E[:,::7*2]
            cases_estim = S_week[:,:-1] + E_week[:,:-1] - S_week[:,1:] - E_week[:,1:]
            cases_real_train = np.loadtxt(folder_name_GT + 'cases_real.txt') 

            cases_train = cases_estim
            err_train[j, i] = np.sqrt(np.mean((cases_real_train - cases_estim)**2))
            
            S = np.loadtxt(folder_name_testg + 'S_rec_testg.txt')
            E = np.loadtxt(folder_name_testg + 'E_rec_testg.txt')
            
            S_week = S[:,::7*2]
            E_week = E[:,::7*2]
            cases_estim = S_week[:,:-1] + E_week[:,:-1] - S_week[:,1:] - E_week[:,1:]
            cases_real_testg = np.loadtxt(folder_name_testg_GT + 'cases_real_testg.txt') 

            cases_testg = cases_estim
            err_testg[j, i] = np.sqrt(np.mean((cases_real_testg - cases_estim)**2))#np.loadtxt(folder_name + 'testg_error.txt')
            err_train_testg[j, i] = err_testg[j,i] / err_train[j,i] 

q_value_min = 0.1
q_value_max = 0.9
median_err_train = np.zeros((len(errors),))
qmin_err_train = np.zeros((len(errors),))
qmax_err_train = np.zeros((len(errors),))

median_err_testg = np.zeros((len(errors),))
qmin_err_testg = np.zeros((len(errors),))
qmax_err_testg = np.zeros((len(errors),))

median_err_train_testg = np.zeros((len(errors),))
qmin_err_train_testg = np.zeros((len(errors),))
qmax_err_train_testg = np.zeros((len(errors),))

for i in range(len(errors)):
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

color_median = 'blue'
color_area   = 'lightblue'


label_quantiles = "{:.2f}".format(q_value_min) + '-' + "{:.2f}".format(q_value_max) + ' quantiles'

fig_violin = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
lab = errors_tex*(2*n_tests)

hue_l = ['train'] * (len(errors)*n_tests)
hue_l.extend(['testg'] * (len(errors)*n_tests))

err_train_df = err_train.T
err_testg_df = err_testg.T

data = np.concatenate((err_train_df,err_testg_df), axis = 1)

data_fl = data.flatten()
df = pd.DataFrame({
    'errors': data_fl,
    'uncertainty': lab,
    'datasets': hue_l
})
sns.set_theme(style="ticks", palette="pastel")
color_p = sns.color_palette('pastel')

hue_order = ['testg', 'train']
sns.boxplot(x="uncertainty", y="errors",
            hue="datasets", hue_order = hue_order, data=df, linewidth = 1.5)
sns.despine(offset=10, trim=True)

plt.yscale('log')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
plt.xlabel(r'uncertainty', fontsize=15)
plt.ylabel(r'error', fontsize=15)
plt.savefig(output_folder + 'boxplot.pdf', format='pdf', bbox_inches='tight')
plt.show()

titles = [r"$u = 0$", r"$u = 1.0 \cdot 10^{-3}$", r"$u = 5.0 \cdot 10^{-3}$", r"$ u = 1.0 \cdot 10^{-2}$", r"$u = 5.0 \cdot 10^{-2}$", r"$u = 1.0 \cdot 10^{-1}$"]

fig_idelta, ax = plt.subplots(2,3, figsize=(3/2*width_in_inches, height_in_inches), dpi=dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

pal = sns.color_palette("pastel",6)
pal.as_hex()

for i in range(2):
    for j in range(3):
        delta_testg_mean = np.mean(delta_testg_mat_l[:,:,i*3+j], axis = 1)
        beta_0_testg_mean = np.mean(beta_0_testg_mat_l[:,:,i*3+j], axis = 1)
        sc = ax[i,j].scatter(delta_testg_mean, delta_testg_true.squeeze(), c = beta_0_testg_mean, cmap = 'Accent',alpha = 0.4)
        ax[i,j].set_title(titles[3*i+j])
        if i == 1:
            ax[i,j].set_xlabel(r"$\delta_r$")
        if j == 0:
            ax[i,j].set_ylabel(r"$\delta_t$")
plt.subplots_adjust(top=0.85, bottom=0.2, wspace=0.3, hspace=0.4)
cbar_ax = fig_idelta.add_axes([0.2, 0.05, 0.6, 0.03])
cbar = fig_idelta.colorbar(sc, cax = cbar_ax, orientation = 'horizontal')
cbar.set_label(r'IC for transmission rate ($\beta_0$)')

# Adjust layout
plt.savefig(output_folder + 'delta_train_color_b0_.pdf', format='pdf', bbox_inches='tight')
plt.show()
