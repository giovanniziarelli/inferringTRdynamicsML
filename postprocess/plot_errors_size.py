import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

sns.set(style="whitegrid")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsfonts,amssymb}')

n_tests = 30
size = [25,50, 75, 100, 150]
size_str = ['25','50','75', '100', '150']
err_train       = np.zeros((len(size), n_tests))
err_testg       = np.zeros((len(size), n_tests))
err_train_testg = np.zeros((len(size), n_tests))
train_times     = np.zeros((len(size), n_tests)) 

output_folder = '' # ex '/home/giovanni/Desktop/LDNets/pprocess_images/errs_size_30/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
for j in range(len(size)):

    for i in range(n_tests):
        folder_name       = '' # ex '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_0_size_'+str(size[j])+'_trialsim_' +str(i+1) + '_/train/'
        folder_name_testg = '' # ex '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay2_err_0_size_'+str(size[j])+'_trialsim_' +str(i+1) + '_/testg/'
        if np.any(np.isnan(np.loadtxt(folder_name + 'train_error.txt'))):
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
            train_times[j, i] = np.sum(np.loadtxt(folder_name + 'train_times.txt'))

            S = np.loadtxt(folder_name_testg + 'S_rec_testg.txt')
            E = np.loadtxt(folder_name_testg + 'E_rec_testg.txt')
            S_week = S[:,::7*2]
            E_week = E[:,::7*2]
            cases_estim = S_week[:,:-1] + E_week[:,:-1] - S_week[:,1:] - E_week[:,1:]
            cases_real_testg = np.loadtxt(folder_name_testg + 'cases_real_testg.txt') 

            cases_testg = cases_estim
            err_testg[j, i] = np.sqrt(np.mean((cases_real_testg - cases_estim)**2))
            err_train_testg[j, i] = err_testg[j,i] / err_train[j,i] 

q_value_min = 0.1
q_value_max = 0.9
median_err_train = np.zeros((len(size),))
qmin_err_train = np.zeros((len(size),))
qmax_err_train = np.zeros((len(size),))

median_err_testg = np.zeros((len(size),))
qmin_err_testg = np.zeros((len(size),))
qmax_err_testg = np.zeros((len(size),))

median_err_train_testg = np.zeros((len(size),))
qmin_err_train_testg = np.zeros((len(size),))
qmax_err_train_testg = np.zeros((len(size),))

qmin_train_times = np.zeros((len(size),))
qmax_train_times = np.zeros((len(size),))
mean_train_times = np.zeros((len(size),))

for i in range(len(size)):
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

    idx = np.argwhere(np.isnan(err_train[i,:]) == False)
    mean_train_times[i] = np.median(train_times[i,idx])
    qmin_train_times[i] = np.quantile(train_times[i, idx], q_value_min)
    qmax_train_times[i] = np.quantile(train_times[i, idx], q_value_max)

width_pixels  = 600#337
height_pixels = 500#266

# Desired DPI
dpi = 100

# Calculate figure size in inches
width_in_inches  = width_pixels / dpi
height_in_inches = height_pixels / dpi

color_median = 'black'
color_area   = 'lightgrey'

label_quantiles = "{:.2f}".format(q_value_min) + '-' + "{:.2f}".format(q_value_max) + ' quantiles'

fig_violin = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
#sns.boxplot(err_testg.T)
lab = size_str*(2*n_tests)

hue_l = ['train'] * (len(size_str)*n_tests)
hue_l.extend(['testg'] * (len(size_str)*n_tests))

err_train_df = np.maximum(err_train.T, 1e-4 * np.ones_like(err_train.T))
err_testg_df = np.maximum(err_testg.T, 9e-4 * np.ones_like(err_testg.T))

data = np.concatenate((err_train_df,err_testg_df), axis = 0)

data_fl = data.flatten()
df = pd.DataFrame({
    'errors': data_fl,
    'size': lab,
    'datasets': hue_l
})

sns.set_theme(style="ticks", palette="pastel")

# Load the example tips dataset
# Draw a nested boxplot to show bills by day and time
hue_order = ['testg', 'train']
sns.boxplot(x="size", y="errors",
            hue="datasets",hue_order = hue_order, data=df, linewidth=1.5 )
plt.yscale('log')
plt.ylim([1e-4, 1e-1])
sns.despine(offset=10, trim=True)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
plt.xlabel(r'size', fontsize=15)
plt.ylabel(r'error', fontsize=15)
plt.savefig(output_folder + 'boxplot.pdf', format='pdf', bbox_inches='tight')
plt.show()

width_pixels  = 300#337
height_pixels = 300#266

# Desired DPI
dpi = 100

# Calculate figure size in inches
width_in_inches  = width_pixels / dpi
height_in_inches = height_pixels / dpi
fig_tt = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.plot(np.arange(1, len(size)+1), mean_train_times,'-s', color = 'purple', alpha = 0.8, linewidth = 0.5)
plt.fill_between(np.arange(1, len(size)+1), qmin_train_times, qmax_train_times, color = 'm', alpha=0.4, label=label_quantiles, linewidth=0)
plt.yscale('log')

plt.xticks(np.arange(1, len(size)+1), size_str)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
plt.xlabel(r'size', fontsize=15)
plt.ylabel(r'training time [s]', fontsize=15)
plt.grid(True, which='both')
plt.savefig(output_folder + 'train_times.pdf', format='pdf', bbox_inches='tight')
plt.show()

