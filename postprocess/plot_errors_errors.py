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
delta_train_true = np.loadtxt('/home/giovanni/Desktop/LDNets/postprocess/deltas_synt/delta_train_real.txt')
delta_testg_true = np.loadtxt('/home/giovanni/Desktop/LDNets/postprocess/deltas_synt/delta_test_real.txt')
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
        folder_name = '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_'+errors[j]+'_size_50_trialsim_' +str(i+1) + '_T_obs_78/train/'
        folder_name_GT = '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_0_size_50_trialsim_' +str(i+1) + '_T_obs_78/train/'
        folder_name_testg = '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_'+errors[j]+'_size_50_trialsim_' +str(i+1) + '_T_obs_78/testg/'
        folder_name_testg_GT = '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_0_size_50_trialsim_' +str(i+1) + '_T_obs_78/testg/'
        
        delta_train_mat_l[:,i,j] = np.loadtxt(folder_name + 'delta_train.txt')
        beta_0_train_mat_l[:,i,j] = np.loadtxt(folder_name_GT + 'beta_real.txt')[:,0]
        delta_testg_mat_l[:,i,j] = np.loadtxt(folder_name_testg + 'delta_testg.txt')
        beta_0_testg_mat_l[:,i,j] = np.loadtxt(folder_name_testg_GT + 'beta_real_testg.txt')[:,0]

        if np.any(np.isnan(np.loadtxt(folder_name + 'S_rec_train.txt'))):
            pass
        else:
            S = np.loadtxt(folder_name + 'S_rec_train.txt')
            E = np.loadtxt(folder_name + 'E_rec_train.txt')
            print(S.shape) 
            S_week = S[:,::7*2]
            E_week = E[:,::7*2]
            cases_estim = S_week[:,:-1] + E_week[:,:-1] - S_week[:,1:] - E_week[:,1:]
            print(cases_estim.shape)
            cases_real_train = np.loadtxt(folder_name_GT + 'cases_real.txt') 
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
            cases_real_testg = np.loadtxt(folder_name_testg_GT + 'cases_real_testg.txt') 
            #cases_train.append(np.loadtxt(folder_name_train + 'cases_train.txt'))

            cases_testg = cases_estim
#           for i in range(5):
#               plt.plot(cases_real_testg[i,:], '-o')
#               plt.plot(cases_estim[i,:], '--o')
#           plt.show()
            err_testg[j, i] = np.sqrt(np.mean((cases_real_testg - cases_estim)**2))#np.loadtxt(folder_name + 'testg_error.txt')
            err_train_testg[j, i] = err_testg[j,i] / err_train[j,i] 

print(err_testg)
idx = (err_train > 0)
print(idx)
print(err_train[idx])
#err_train = np.delete(err_train, idx, axis=1)
#err_testg = np.delete(err_testg, idx, axis=1)
#err_train_testg = np.delete(err_train_testg, idx, axis=1)

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

print(median_err_testg)
# Desired DPI
dpi = 100

# Calculate figure size in inches
width_in_inches  = width_pixels / dpi
height_in_inches = height_pixels / dpi

color_median = 'blue'
color_area   = 'lightblue'


label_quantiles = "{:.2f}".format(q_value_min) + '-' + "{:.2f}".format(q_value_max) + ' quantiles'

fig_err_train = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.plot(errors, median_err_train, '--o', color = color_median)
plt.fill_between(errors, qmin_err_train, qmax_err_train, color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
plt.legend()
plt.xlabel(r'size')
plt.title(r'Training error')
plt.savefig(output_folder + 'training_err_train.pdf', format='pdf', bbox_inches='tight')
plt.show()

fig_violin = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
#sns.boxplot(err_testg.T)
lab = errors_tex*(2*n_tests)

hue_l = ['train'] * (len(errors)*n_tests)
hue_l.extend(['testg'] * (len(errors)*n_tests))

err_train_df = err_train.T
err_testg_df = err_testg.T

data = np.concatenate((err_train_df,err_testg_df), axis = 1)

print(data.shape)
data_fl = data.flatten()
print(data_fl.shape)
print(len(hue_l))
print(len(lab))
df = pd.DataFrame({
    'errors': data_fl,
    'uncertainty': lab,
    'datasets': hue_l
})
#df = df[df['datasets']=='testg']
sns.set_theme(style="ticks", palette="pastel")
color_p = sns.color_palette('pastel')

# Load the example tips dataset
# Draw a nested boxplot to show bills by day and time
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

fig_err_testg = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.loglog(errors_abs, median_err_testg[1:] / errors_abs , '-o', color = 'lightblue', markersize = 10, label = r'testg')
plt.loglog(errors_abs, median_err_train[1:] / errors_abs , '-o', color = 'coral', markersize = 10, label = r'train')
plt.loglog(np.array(errors_abs), 1 / 1e2 / np.array(errors_abs) , '--o', color = 'black', markersize = 10, alpha = 0.5, label = r'$1/u$')
plt.legend()
plt.xlabel(r'$u$')
plt.grid(True, which="both")
#plt.legend([r'1/$u$', r'$Q05$/$u$'])
plt.title(r'$Q05$/$u$')
plt.savefig(output_folder + 'testging_err_testg.pdf', format='pdf', bbox_inches='tight')

plt.show()

fig_err_train_testg = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.plot(errors, median_err_train_testg, '--o', color = color_median)
plt.fill_between(errors, qmin_err_train_testg, qmax_err_train_testg, color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
plt.legend()
plt.xlabel(r'size')
plt.title(r'Error ratio')
plt.savefig(output_folder + 'train_testging_err_train_testg.pdf', format='pdf', bbox_inches='tight')
plt.show()

#delta_train_median = np.median(delta_train_mat, axis = 1)
#titles = [r"$u = 0$",r"$u = 1e-3$",r"$u = 5e-3$",r"$u = 1e-2$",r"$u = 5e-2$",r"$u = 1e-1$" ]
titles = [r"$u = 0$", r"$u = 1.0 \cdot 10^{-3}$", r"$u = 5.0 \cdot 10^{-3}$", r"$ u = 1.0 \cdot 10^{-2}$", r"$u = 5.0 \cdot 10^{-2}$", r"$u = 1.0 \cdot 10^{-1}$"]
for deg_pol in range(4):
    fig_idelta, ax = plt.subplots(2,3, figsize=(3/2*width_in_inches, height_in_inches), dpi=dpi)
    ####for i in range(1):
    ####    ax[0].scatter(delta_train_mat[:,i], delta_train_true.squeeze())
    ####    ax[1].scatter(delta_testg_mat[:,i], delta_testg_true.squeeze())
    #plt.legend()
    pal = sns.color_palette("pastel",6)
    pal.as_hex()

    for i in range(2):
        for j in range(3):
            delta_testg_mean = np.mean(delta_testg_mat_l[:,:,i*3+j], axis = 1)
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
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

####for i in range(1):
####    ax[0].scatter(delta_train_mat[:,i], delta_train_true.squeeze())
####    ax[1].scatter(delta_testg_mat[:,i], delta_testg_true.squeeze())
#plt.legend()
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
#plt.subplots_adjust(left = 0.1, right=0.75, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
plt.subplots_adjust(top=0.85, bottom=0.2, wspace=0.3, hspace=0.4)#fig_idelta.colorbar(sc, label = r'$\beta_0$')
cbar_ax = fig_idelta.add_axes([0.2, 0.05, 0.6, 0.03])#[0.85, 0.15, 0.03, 0.7])
cbar = fig_idelta.colorbar(sc, cax = cbar_ax, orientation = 'horizontal')#ax=ax.ravel().tolist(), shrink=0.7, orientation='vertical')
cbar.set_label(r'IC for transmission rate ($\beta_0$)')

# Adjust layout
#plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(output_folder + 'delta_train_color_b0_.pdf', format='pdf', bbox_inches='tight')
plt.show()
