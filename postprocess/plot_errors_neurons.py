import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

sns.set(style="whitegrid")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsfonts,amssymb}')

n_tests = 20
neurons = [4,8,10,12]
err_train       = np.zeros((len(neurons), n_tests))
err_testg       = np.zeros((len(neurons), n_tests))
err_train_testg = np.zeros((len(neurons), n_tests))
output_folder = '' # ex '/home/giovanni/Desktop/LDNets/pprocess_images/errs_neurons/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
for j in range(len(neurons)):

    for i in range(n_tests):
        folder_name       = '' # ex '/home/giovanni/Desktop/LDNets/neurons_'+ str(neurons[j])+'_364_synthetic_lay2_err_0_size_50_trialsim_' +str(i+1) + '/train/'
        folder_name_testg = '' # ex '/home/giovanni/Desktop/LDNets/neurons_'+ str(neurons[j])+'_364_synthetic_lay2_err_0_size_50_trialsim_' +str(i+1) + '/testg/'
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
median_err_train = np.zeros((len(neurons),))
qmin_err_train = np.zeros((len(neurons),))
qmax_err_train = np.zeros((len(neurons),))

median_err_testg = np.zeros((len(neurons),))
qmin_err_testg = np.zeros((len(neurons),))
qmax_err_testg = np.zeros((len(neurons),))

median_err_train_testg = np.zeros((len(neurons),))
qmin_err_train_testg = np.zeros((len(neurons),))
qmax_err_train_testg = np.zeros((len(neurons),))

for i in range(len(neurons)):
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

color_median = 'black'
color_area   = 'lightgrey'


label_quantiles = "{:.2f}".format(q_value_min) + '-' + "{:.2f}".format(q_value_max) + ' quantiles'

fig_violin = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
lab = ['4', '8', '10', '12']*(2*n_tests)

hue_l = ['train'] * (4*n_tests)
hue_l.extend(['testg'] * (4*n_tests))

err_train_df = err_train.T
err_testg_df = err_testg.T

data = np.concatenate((err_train_df,err_testg_df), axis = 0)

data_fl = data.flatten()
df = pd.DataFrame({
    'errors': data_fl,
    'neurons': lab,
    'datasets': hue_l
})
df = df[df['datasets'] == 'testg']
sns.set_theme(style="ticks", palette="pastel")

# Load the example tips dataset
# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="neurons", y="errors",
            hue="datasets", data=df, linewidth = 1.5)
sns.despine(offset=10, trim=True)

plt.yscale('log')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
plt.xlabel(r'neurons', fontsize=15)
plt.ylabel(r'error', fontsize=15)
plt.savefig(output_folder + 'boxplot.pdf', format='pdf', bbox_inches='tight')
plt.show()

