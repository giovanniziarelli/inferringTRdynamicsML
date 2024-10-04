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
delta_train_true = np.loadtxt('/home/giovanni/Desktop/LDNets/postprocess/deltas_synt/delta_train_real.txt')
delta_testg_true = np.loadtxt('/home/giovanni/Desktop/LDNets/postprocess/deltas_synt/delta_test_real.txt')
delta_train_mat_l  = np.zeros((49,n_tests, 6))
delta_testg_mat_l  = np.zeros((48,n_tests, 6))
beta_0_testg_mat_l = np.zeros((48, n_tests, 6))

delta_train_mean = np.zeros((49,1))

err_train       = np.zeros((len(T_obs_vec), n_tests))
err_testg       = np.zeros((len(T_obs_vec), n_tests))
err_train_testg = np.zeros((len(T_obs_vec), n_tests))
output_folder = '/home/giovanni/Desktop/LDNets/pprocess_images/errs_T_obs_vec/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
for j in range(len(T_obs_vec)):

    for i in range(n_tests):
        folder_name = '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_0_size_50_trialsim_' +str(i+1) + '_T_obs_' +str(T_obs_vec[j])+'/train/'
        folder_name_testg = '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_0_size_50_trialsim_' +str(i+1) + '_T_obs_' +str(T_obs_vec[j])+'/testg/'
        if j > 0:
            delta_train_mat_l[:,i,j-1] = np.loadtxt(folder_name + 'delta_train.txt')
            delta_testg_mat_l[:,i,j-1] = np.loadtxt(folder_name_testg + 'delta_testg.txt')
            beta_0_testg_mat_l[:,i,j-1] = np.loadtxt(folder_name_testg + 'beta_real_testg.txt')[:,0]
        if np.any(np.isnan(np.loadtxt(folder_name + 'train_error.txt'))) or len(os.listdir(folder_name)) == 0:
            pass
        else:
            S = np.loadtxt(folder_name + 'S_rec_train.txt')
            E = np.loadtxt(folder_name + 'E_rec_train.txt')
            print(S.shape) 
            S_week = S[:,::7*2]
            E_week = E[:,::7*2]
            cases_estim = S_week[:,:-1] + E_week[:,:-1] - S_week[:,1:] - E_week[:,1:]
            print(cases_estim.shape)
            cases_real_train = np.loadtxt(folder_name + 'cases_real.txt') 
            #cases_train.append(np.loadtxt(folder_name_train + 'cases_train.txt'))

            cases_train = cases_estim
            err_train[j, i] = np.sqrt(np.mean((cases_real_train - cases_estim)**2))
            

            S = np.loadtxt(folder_name_testg + 'S_rec_testg.txt')
            E = np.loadtxt(folder_name_testg + 'E_rec_testg.txt')
            print(S.shape) 
            S_week = S[:,::7*2]
            E_week = E[:,::7*2]
            cases_estim = S_week[:,:-1] + E_week[:,:-1] - S_week[:,1:] - E_week[:,1:]
            print(cases_estim.shape)
            cases_real_testg = np.loadtxt(folder_name_testg + 'cases_real_testg.txt') 
            #cases_train.append(np.loadtxt(folder_name_train + 'cases_train.txt'))

            cases_testg = cases_estim
#           for i in range(5):
#               plt.plot(cases_real_testg[i,:], '-o')
#               plt.plot(cases_estim[i,:], '--o')
#           plt.show()
            err_testg[j, i] = np.sqrt(np.mean((cases_real_testg - cases_estim)**2))#np.loadtxt(folder_name + 'testg_error.txt')
            err_train_testg[j, i] = err_testg[j,i] / err_train[j,i] 
print(err_train)
idx = (err_train > 0)
print(idx)
print(err_train[idx])
#err_train = np.delete(err_train, idx, axis=1)
#err_testg = np.delete(err_testg, idx, axis=1)
#err_train_testg = np.delete(err_train_testg, idx, axis=1)

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
print(err_testg[0,:])
for i in range(len(T_obs_vec)):
    idx = np.argwhere(np.isnan(err_testg[i,:]) == False)
    #idx = err_train[i,:] > 0
    median_err_train[i] = np.median(err_train[i,idx])
    print(median_err_train.shape)
    qmin_err_train[i] = np.quantile(err_train[i, idx], q_value_min)
    qmax_err_train[i] = np.quantile(err_train[i, idx], q_value_max)

    median_err_testg[i] = np.median(err_testg[i,idx])
    print(median_err_testg.shape)
    qmin_err_testg[i] = np.quantile(err_testg[i, idx], q_value_min)
    qmax_err_testg[i] = np.quantile(err_testg[i, idx], q_value_max)

    median_err_train_testg[i] = np.median(err_train_testg[i,idx])
    print(median_err_train_testg.shape)
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

print('qui')
print(T_obs_vec)
print(median_err_train)
print(median_err_testg)

fig_err_train = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.plot(T_obs_vec, median_err_train, '--o', color = color_median)
plt.fill_between(T_obs_vec, qmin_err_train, qmax_err_train, color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
plt.legend()
plt.xlabel(r'size')
plt.title(r'Training error')
plt.savefig(output_folder + 'training_err_train.pdf', format='pdf', bbox_inches='tight')
plt.show()

fig_violin = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
#sns.boxplot(err_testg.T)
lab = T_obs_vec_str*(2*n_tests)

hue_l = ['train'] * (len(T_obs_vec_str)*n_tests)
hue_l.extend(['testg'] * (len(T_obs_vec_str)*n_tests))

#err_train_df = err_train.T
#err_testg_df = err_testg.T

err_train_df = np.maximum(err_train.T, 1e-4 * np.ones_like(err_train.T))
err_testg_df = np.maximum(err_testg.T, 9e-4 * np.ones_like(err_testg.T))

data = np.concatenate((err_train_df,err_testg_df), axis = 0)

print(data)
data_fl = data.flatten()
print(data_fl.shape)
print(len(hue_l))
print(len(lab))
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

fig_err_testg = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.plot(T_obs_vec, median_err_testg, '--o', color = color_median)
plt.fill_between(T_obs_vec, qmin_err_testg, qmax_err_testg, color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
plt.legend()
plt.xlabel(r'size')
plt.title(r'Testing error')
plt.savefig(output_folder + 'testging_err_testg.pdf', format='pdf', bbox_inches='tight')
plt.show()

fig_err_train_testg = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.plot(T_obs_vec, median_err_train_testg, '--o', color = color_median)
plt.fill_between(T_obs_vec, qmin_err_train_testg, qmax_err_train_testg, color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
plt.legend()
plt.xlabel(r'size')
plt.title(r'Error ratio')
plt.savefig(output_folder + 'train_testging_err_train_testg.pdf', format='pdf', bbox_inches='tight')
plt.show()

titles = [r"$T_{\mathrm{obs}} = 35$",r"$T_{\mathrm{obs}} = 56$",r"$T_{\mathrm{obs}} = 77$",r"$T_{\mathrm{obs}} = 98$",r"$T_{\mathrm{obs}} = 119$",r"$T_{\mathrm{obs}} = 140$" ]
for deg_pol in range(4):
    fig_idelta, ax = plt.subplots(2,3, figsize=(3/2*width_in_inches, height_in_inches), dpi=dpi)
    ####for i in range(1):
    ####    ax[0].scatter(delta_train_mat[:,i], delta_train_true.squeeze())
    ####    ax[1].scatter(delta_testg_mat[:,i], delta_testg_true.squeeze())
    #plt.legend()
    pal = sns.color_palette("pastel",6)
    pal.as_hex()
    print(delta_testg_mat_l)
    for i in range(2):
        for j in range(3):
            delta_testg_mean = np.zeros((48,))
            for k  in range(48):
                idx = np.isnan(delta_testg_mat_l[k,:,i*3+j]) 
                delta_testg_mean[k] = np.mean(delta_testg_mat_l[k, idx == False,i*3+j])
            coef_testg = np.polyfit(delta_testg_mean, delta_testg_true.squeeze(),2 + deg_pol)
            x_testg = np.linspace(min(delta_testg_mean), max(delta_testg_mean), 1000)
            ax[i,j].scatter(delta_testg_mean, delta_testg_true.squeeze(), color = pal[3*i+j], alpha = 0.2)
            ax[i,j].plot(x_testg, np.polyval(coef_testg, x_testg), color = pal[3*i+j], linewidth = 3)
            ax[i,j].set_title(titles[3*i+j])
            if i == 1:
                ax[i,j].set_xlabel(r"$\delta_r$")
            if j == 0:
                ax[i,j].set_ylabel(r"$\delta_t$")
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(output_folder + 'delta_train_dp_'+str(deg_pol)+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()

fig_idelta, ax = plt.subplots(2,3, figsize=(3/2*width_in_inches, height_in_inches), dpi=dpi)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amssymb}')

####for i in range(1):
####    ax[0].scatter(delta_train_mat[:,i], delta_train_true.squeeze())
####    ax[1].scatter(delta_testg_mat[:,i], delta_testg_true.squeeze())
#plt.legend()
pal = sns.color_palette("pastel",6)
pal.as_hex()
print(delta_testg_mat_l)
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
plt.subplots_adjust(top=0.85, bottom=0.2, wspace=0.3, hspace=0.4)#fig_idelta.colorbar(sc, label = r'$\beta_0$')
cbar_ax = fig_idelta.add_axes([0.2, 0.05, 0.6, 0.03])#[0.85, 0.15, 0.03, 0.7])
cbar = fig_idelta.colorbar(sc, cax = cbar_ax, orientation = 'horizontal')#ax=ax.ravel().tolist(), shrink=0.7, orientation='vertical')
cbar.set_label(r'IC for transmission rate ($\beta_0$)')

plt.savefig(output_folder + 'delta_train_color_b0_tobs.pdf', format='pdf', bbox_inches='tight')
plt.show()
