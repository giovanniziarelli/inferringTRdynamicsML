import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from matplotlib.ticker import FuncFormatter
sns.set(style="whitegrid")

plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsfonts, amssymb}')
####plt.rcParams.update({
####    "font.family": "serif",    # Set font family to serif
####    "font.serif": ["Times"],   # Set specific serif font, e.g., Times
####    "font.size": 15,           # Set global font size
####    "axes.labelsize": 15,      # Set the size of the axis labels
####    "xtick.labelsize": 12,     # Set the size of the x-tick labels
####    "ytick.labelsize": 12,     # Set the size of the y-tick labels
####    "axes.titlesize": 16,      # Set the size of the axis titles
####})
n_tests = 20
layers = [2,3,4]
layers_str = ['2','3','4']
err_train       = np.zeros((len(layers), n_tests))
err_testg       = np.zeros((len(layers), n_tests))
err_train_testg = np.zeros((len(layers), n_tests))
output_folder = '/home/giovanni/Desktop/LDNets/pprocess_images/errs_layers/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
for j in range(len(layers)):

    for i in range(n_tests):
        folder_name = '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay'+str(layers[j])+'_err_0_size_50_trialsim_' +str(i+1) + '/train/'
        folder_name_testg = '/home/giovanni/Desktop/LDNets/neurons_4_364_synthetic_lay'+str(layers[j])+'_err_0_size_50_trialsim_' +str(i+1) + '/testg/'
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
median_err_train = np.zeros((len(layers),))
qmin_err_train = np.zeros((len(layers),))
qmax_err_train = np.zeros((len(layers),))

median_err_testg = np.zeros((len(layers),))
qmin_err_testg = np.zeros((len(layers),))
qmax_err_testg = np.zeros((len(layers),))

median_err_train_testg = np.zeros((len(layers),))
qmin_err_train_testg = np.zeros((len(layers),))
qmax_err_train_testg = np.zeros((len(layers),))
print(err_testg[0,:])
for i in range(len(layers)):
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
print(layers)
print(median_err_train)
print(median_err_testg)

fig_err_train = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.plot(layers, median_err_train, '--o', color = color_median)
plt.fill_between(layers, qmin_err_train, qmax_err_train, color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
plt.legend()
plt.xlabel(r'size')
plt.title(r'Training error')
plt.savefig(output_folder + 'training_err_train.pdf', format='pdf', bbox_inches='tight')
plt.show()

fig_violin = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
#sns.boxplot(err_testg.T)
lab = layers_str*(2*n_tests)

hue_l = ['train'] * (len(layers_str)*n_tests)
hue_l.extend(['testg'] * (len(layers_str)*n_tests))

err_train_df = err_train.T
err_testg_df = err_testg.T

data = np.concatenate((err_train_df,err_testg_df), axis = 0)

print(data.shape)
data_fl = data.flatten()
print(data_fl.shape)
print(len(hue_l))
print(len(lab))
df = pd.DataFrame({
    'errors': data_fl,
    'layers': lab,
    'datasets': hue_l
})

sns.set_theme(style="ticks", palette="pastel")
df = df[df['datasets'] == 'testg']
# Load the example tips dataset
# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="layers", y="errors",
            hue="datasets", data=df, linewidth = 1.5)
sns.despine(offset=10, trim=True)

plt.yscale('log')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
plt.xlabel(r'layers', fontsize=15)
plt.ylabel(r'error', fontsize=15)
plt.savefig(output_folder + 'boxplot.pdf', format='pdf', bbox_inches='tight')
plt.show()

####fig_violin,ax1 = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
####ax2 = ax1.twinx()
#####sns.boxplot(err_testg.T)
####lab = layers_str*(2*n_tests)

####hue_l = ['train'] * (len(layers_str)*n_tests)
####hue_l.extend(['testg'] * (len(layers_str)*n_tests))

####err_train_df = err_train.T
####err_testg_df = err_testg.T

####data = np.concatenate((err_train_df,err_testg_df), axis = 0)
####print(data.shape)
####data_fl = data.flatten()
####print(data_fl.shape)
####print(len(hue_l))
####print(len(lab))
####df = pd.DataFrame({
####    'errors': data_fl,
####    'layers': lab,
####    'datasets': hue_l
####})
####data_train = df[df['datasets'] == 'train']
####data_testg = df[df['datasets'] == 'testg']
####custom_params = {"axes.spines.right": False, "axes.spines.left": False, "axes.spines.top": False}
####sns.set_theme(style="ticks", palette="pastel", rc=custom_params)
#####sns.set_theme(palette="pastel")


##### Load the example tips dataset
##### Draw a nested boxplot to show bills by day and time
########sns.boxplot(x="layers", y="errors",
########            hue="datasets", palette = ["m", "lightblue"], data=df)
########sns.despine(offset=10, trim=True)
########plt.xlabel(r'layers', fontsize=15)
########plt.ylabel(r'error', fontsize=15)
########plt.show()
########fig_err_testg = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
########plt.plot(layers, median_err_testg, '--o', color = color_median)
########plt.fill_between(layers, qmin_err_testg, qmax_err_testg, color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
########plt.legend()
########plt.xlabel(r'size')
########plt.title(r'Testing error')
########plt.savefig(output_folder + 'testging_err_testg.pdf', format='pdf', bbox_inches='tight')
########plt.show()

####ax1.set_yscale("logit")
####sns.boxplot(x="layers", y="errors",
####            hue=0, hue_order=[0,1], dodge=True, legend=False, data=data_train, ax = ax1)
####ax1.set_ylim([1e-3, 2.3e-3])
####ax1.spines['left'].set_color('lightblue')
####ax1.spines['left'].set_linewidth(1.8)
####ax1.spines['right'].set_color('none')

####ax1.tick_params(color='lightblue')
####ax1.set_ylabel(r'error', fontsize=15, color='lightblue')
####sns.boxplot(x="layers", y="errors",
####            hue=1, hue_order=[0,1], dodge=True, legend=False, data=data_testg, ax = ax2)
####ax2.tick_params(color='orange')
#####sns.despine(offset=10, trim=True)
####ax1.set_xlabel(r'layers', fontsize=15)
####ax2.set_ylabel(r'error', fontsize=15, color='orange')

####ax2.spines['right'].set_color('orange')
####ax2.spines['right'].set_linewidth(1.5)
####ax2.spines['left'].set_color('none')
####ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:0.4f}'))
####custom_params = {"axes.spines.right": False, "axes.spines.left": False, "axes.spines.top": False}
####sns.set_theme(style="ticks", palette="pastel", rc=custom_params)
####for label in (ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels()):
####    label.set_fontname('serif')
####    label.set_fontsize(12)
####for label in (ax1.get_xticklabels() + ax2.get_yticklabels()):
####    label.set_color('black')  # Set consistent color
#####plt.title(r'Training-Testing errors')
#####sns.despine(offset=10, trim=True, ax=ax1)
#####sns.despine(offset=10, trim=True, ax=ax2, left=True)
#####plt.rcParams['text.usetex'] = True
####plt.show()
####fig_violin, ax1 = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
####ax2 = ax1.twinx()

##### Set global LaTeX parameters for Matplotlib
####plt.rcParams.update({
####    'text.usetex': True,     # Use LaTeX for text rendering
####    'font.family': 'serif', # Use serif fonts
####    'font.serif': ['Times'], # Use Times font for serif
####    'axes.titlesize': 16,    # Title size
####    'axes.labelsize': 15,    # Label size
####    'xtick.labelsize': 12,   # X-tick label size
####    'ytick.labelsize': 12,   # Y-tick label size
####    'legend.fontsize': 12    # Legend font size
####})

##### Set theme for Seaborn
####custom_params = {"axes.spines.right": False, "axes.spines.left": False, "axes.spines.top": False}
####sns.set_theme(style="ticks", palette="pastel", rc=custom_params)

##### Plot boxplots
####ax1.set_yscale("logit")
####sns.boxplot(x="layers", y="errors", hue=0, hue_order=[0, 1], dodge=True, legend=False, data=data_train, ax=ax1)#, palette={"train": "m"})
####ax1.set_ylim([1e-3, 2.3e-3])
####ax1.spines['left'].set_color('lightblue')
####ax1.spines['left'].set_linewidth(1.8)
####ax1.spines['right'].set_color('none')
####ax1.tick_params(color='lightblue')
####ax1.set_ylabel(r'error', fontsize=15, color='lightblue')

####sns.boxplot(x="layers", y="errors", hue=1, hue_order=[0, 1], dodge=True, legend=False, data=data_testg, ax=ax2)#, palette={"testg": "orange"})
####ax2.tick_params(color='orange')
####ax2.spines['right'].set_color('orange')
####ax2.spines['right'].set_linewidth(1.5)
####ax2.spines['left'].set_color('none')
####ax2.set_ylabel(r'error', fontsize=15, color='orange')

##### Format y-axis labels
####ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:0.4f}'))

##### Set x-axis label
####ax1.set_xlabel(r'layers', fontsize=15)

##### Ensure all labels use LaTeX formatting
####for label in (ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels()):
####    label.set_fontname('serif')
####    label.set_fontsize(12)
####    label.set_color('black')  # Set consistent color

##### Adjust spines and remove extra decorations
####sns.despine(offset=10, trim=True, ax=ax1)
####sns.despine(offset=10, trim=True, ax=ax2, left=True)

##### Show the plot
####plt.show()
fig_err_testg = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.plot(layers, median_err_testg, '--o', color = color_median)
plt.fill_between(layers, qmin_err_testg, qmax_err_testg, color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
plt.legend()
plt.xlabel(r'size')
plt.title(r'Testing error')
plt.savefig(output_folder + 'testging_err_testg.pdf', format='pdf', bbox_inches='tight')
plt.show()
fig_err_train_testg = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)
plt.plot(layers, median_err_train_testg, '--o', color = color_median)
plt.fill_between(layers, qmin_err_train_testg, qmax_err_train_testg, color = color_area, alpha=0.5, label=label_quantiles, linewidth=0)
plt.legend()
plt.xlabel(r'size')
plt.title(r'Error ratio')
plt.savefig(output_folder + 'train_testging_err_train_testg.pdf', format='pdf', bbox_inches='tight')
plt.show()

