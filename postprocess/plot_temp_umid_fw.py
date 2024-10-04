import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
from cycler import cycler
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
colors = plt.cm.tab20c.colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
mpl.rcParams["figure.constrained_layout.use"] = True
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
plt.rc('axes', prop_cycle=cycler(color=colors))
    #plt.set_cmap('Accent')
plt.set_cmap('Accent')

undetection_coef = 0.23
gamma = 1/1.2

output_folder = '/home/giovanni/Desktop/LDNets/pprocess_images/ita_temp_umid_fw/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

#os.mkdir(output_folder_train) 
#os.mkdir(output_folder_testg)

folder_name = '/home/giovanni/Desktop/LDNets/temp_prova_neurons_4_196_lay2_umidity_Tobs_50_trialsim3_rd_2/testg/'

I_temp_A   = undetection_coef * np.loadtxt(folder_name + 'I_rec_testg_temp_0.txt')   
I_temp_f   = undetection_coef * np.loadtxt(folder_name + 'I_rec_testg_temp_1.txt')   
I_temp_phi = undetection_coef * np.loadtxt(folder_name + 'I_rec_testg_temp_2.txt')   
I_temp_Tm  = undetection_coef * np.loadtxt(folder_name + 'I_rec_testg_temp_3.txt')   
I_umid     = undetection_coef * np.loadtxt(folder_name + 'I_rec_testg_umid.txt')   


beta_temp_A   = np.loadtxt(folder_name + 'beta_rec_temp_testg_0.txt')   
beta_temp_f   = np.loadtxt(folder_name + 'beta_rec_temp_testg_1.txt')   
beta_temp_phi = np.loadtxt(folder_name + 'beta_rec_temp_testg_2.txt')   
beta_temp_Tm  = np.loadtxt(folder_name + 'beta_rec_temp_testg_3.txt')   
beta_umid     = np.loadtxt(folder_name + 'beta_rec_umid_testg.txt')   

temps_A   = np.loadtxt(folder_name + 'temps_0.txt')
temps_f   = np.loadtxt(folder_name + 'temps_1.txt')
temps_phi = np.loadtxt(folder_name + 'temps_2.txt')
temps_Tm  = np.loadtxt(folder_name + 'temps_3.txt')
umids     = np.loadtxt(folder_name + 'umids.txt')

f_temp_A   = 2 * np.diff(beta_temp_A)
f_temp_f   = 2 * np.diff(beta_temp_f)
f_temp_phi = 2 * np.diff(beta_temp_phi)
f_temp_Tm  = 2 * np.diff(beta_temp_Tm)
f_umid     = 2 * np.diff(beta_umid)

Tfin = 196 

t_num      = np.loadtxt(folder_name + 't_num.txt')
t          = t_num[::2]
width_pixels  = 200#600#337
height_pixels = 200#500#266

# Desired DPI
dpi = 100

# Calculate figure size in inches
width_in_inches  = width_pixels / dpi
height_in_inches = height_pixels / dpi

colors = ["turquoise", "salmon"]
cmap = LinearSegmentedColormap.from_list("turquoise_salmon", colors, N=temps_A.shape[0])
fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
for k in range(temps_A.shape[0]):
    ax[0].plot(t, I_temp_A[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[1].plot(t, beta_temp_A[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[2].plot(t_num, temps_A[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))

#ax[0].set_title('Infected') 
ax[0].set_ylabel(r'Detected infected')
ax[0].set_xlabel(r'days')
#ax[1].set_title('Transmission Rate') 
ax[1].set_ylabel(r'Transmission rate')
ax[1].set_xlabel('days')
#ax[2].set_title('Temperature')
ax[2].set_ylabel(r'Temperature [°C]')
#ax[2].set_ylabel(r'Relative Umidity [\%]')
ax[2].set_xlabel('days')
zoom_ax = inset_axes(ax[0], width="30%", height="30%", loc="upper right")  # Riquadro di zoom in alto a destra
zoom_ax.set_xlim([54, 59])  # Limiti per lo zoom sull'asse x (modifica a seconda del tuo intervallo)
zoom_ax.set_ylim([0.029, 0.0315])  # Limiti per lo zoom sull'asse y (modifica a seconda del tuo intervallo)
zoom_ax.tick_params(axis='both', which='major', labelsize=7)
# Loop per plottare i dati anche nel riquadro di zoom
for k in range(temps_A.shape[0]):
    zoom_ax.plot(t, I_temp_A[k, :], '-', linewidth=2, color=cmap(k))

# Rimuovere etichette nel riquadro di zoom
#zoom_ax.set_xticks([])
#zoom_ax.set_yticks([])

fig.tight_layout()
plt.savefig(output_folder + 'prova_test_temp_amplitude.pdf')
#plt.show()

fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
for k in range(temps_f.shape[0]):
    ax[0].plot(t, I_temp_f[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[1].plot(t, beta_temp_f[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[2].plot(t_num, temps_f[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))

#ax[0].set_title('Infected') 
ax[0].set_ylabel(r'Detected infected')
ax[0].set_xlabel(r'days')
#ax[1].set_title('Transmission Rate') 
ax[1].set_ylabel(r'Transmission rate')
ax[1].set_xlabel('days')
#ax[2].set_title('Temperature')
ax[2].set_ylabel(r'Temperature [°C]')
#ax[2].set_ylabel(r'Relative Umidity [\%]')
ax[2].set_xlabel('days')
zoom_ax = inset_axes(ax[0], width="30%", height="30%", loc="upper right")  # Riquadro di zoom in alto a destra
zoom_ax.set_xlim([49, 63])  # Limiti per lo zoom sull'asse x (modifica a seconda del tuo intervallo)
zoom_ax.set_ylim([0.020, 0.0315])  # Limiti per lo zoom sull'asse y (modifica a seconda del tuo intervallo)
zoom_ax.tick_params(axis='both', which='major', labelsize=7)
# Loop per plottare i dati anche nel riquadro di zoom
for k in range(temps_A.shape[0]):
    zoom_ax.plot(t, I_temp_f[k, :], '-', linewidth=2, color=cmap(k))


fig.tight_layout()
plt.savefig(output_folder + 'prova_test_temp_frequency.pdf')
#plt.show()

fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
for k in range(temps_phi.shape[0]):
    ax[0].plot(t, I_temp_phi[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[1].plot(t, beta_temp_phi[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[2].plot(t_num, temps_phi[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))

#ax[0].set_title('Infected') 
ax[0].set_ylabel(r'Detected infected')
ax[0].set_xlabel(r'days')
#ax[1].set_title('Transmission Rate') 
ax[1].set_ylabel(r'Transmission rate')
ax[1].set_xlabel('days')
#ax[2].set_title('Temperature')
ax[2].set_ylabel(r'Temperature [°C]')
#ax[2].set_ylabel(r'Relative Umidity [\%]')
ax[2].set_xlabel('days')
zoom_ax = inset_axes(ax[0], width="30%", height="30%", loc="upper right")  # Riquadro di zoom in alto a destra
zoom_ax.set_xlim([53, 59])  # Limiti per lo zoom sull'asse x (modifica a seconda del tuo intervallo)
zoom_ax.set_ylim([0.029, 0.032])  # Limiti per lo zoom sull'asse y (modifica a seconda del tuo intervallo)
zoom_ax.tick_params(axis='both', which='major', labelsize=7)
# Loop per plottare i dati anche nel riquadro di zoom
for k in range(temps_A.shape[0]):
    zoom_ax.plot(t, I_temp_phi[k, :], '-', linewidth=2, color=cmap(k))

fig.tight_layout()
plt.savefig(output_folder + 'prova_test_temp_phase.pdf')
#plt.show()

fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
for k in range(temps_Tm.shape[0]):
    ax[0].plot(t, I_temp_Tm[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[1].plot(t, beta_temp_Tm[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[2].plot(t_num, temps_Tm[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))

#ax[0].set_title('Infected') 
ax[0].set_ylabel(r'Detected infected')
ax[0].set_xlabel(r'days')
#ax[1].set_title('Transmission Rate') 
ax[1].set_ylabel(r'Transmission rate')
ax[1].set_xlabel('days')
#ax[2].set_title('Temperature')
ax[2].set_ylabel(r'Temperature [°C]')
#ax[2].set_ylabel(r'Relative Umidity [\%]')
ax[2].set_xlabel('days')
zoom_ax = inset_axes(ax[0], width="30%", height="30%", loc="upper right")  # Riquadro di zoom in alto a destra
zoom_ax.set_xlim([50, 65])  # Limiti per lo zoom sull'asse x (modifica a seconda del tuo intervallo)
zoom_ax.set_ylim([0.022, 0.032])  # Limiti per lo zoom sull'asse y (modifica a seconda del tuo intervallo)
zoom_ax.tick_params(axis='both', which='major', labelsize=7)
# Loop per plottare i dati anche nel riquadro di zoom
for k in range(temps_A.shape[0]):
    zoom_ax.plot(t, I_temp_Tm[k, :], '-', linewidth=2, color=cmap(k))

fig.tight_layout()
plt.savefig(output_folder + 'prova_test_temp_Tmean.pdf')
#plt.show()

colors = ["goldenrod", "forestgreen"]
cmap = LinearSegmentedColormap.from_list("goldenrod_forestgreen", colors, N=temps_A.shape[0])
fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
for k in range(umids.shape[0]):
    ax[0].plot(t, I_umid[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[1].plot(t, beta_umid[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[2].plot(t_num, umids[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))

#ax[0].set_title('Infected') 
ax[0].set_ylabel(r'Detected infected')
ax[0].set_xlabel(r'days')
#ax[1].set_title('Transmission Rate') 
ax[1].set_ylabel(r'Transmission rate')
ax[1].set_xlabel('days')
#ax[2].set_title('Temperature')
#ax[2].set_ylabel(r'Temperature [°C]')
ax[2].set_ylabel(r'Relative Umidity [\%]')
ax[2].set_xlabel('days')
zoom_ax = inset_axes(ax[0], width="30%", height="30%", loc="upper right")  # Riquadro di zoom in alto a destra
zoom_ax.set_xlim([50, 62])  # Limiti per lo zoom sull'asse x (modifica a seconda del tuo intervallo)
zoom_ax.set_ylim([0.022, 0.032])  # Limiti per lo zoom sull'asse y (modifica a seconda del tuo intervallo)
zoom_ax.tick_params(axis='both', which='major', labelsize=7)
# Loop per plottare i dati anche nel riquadro di zoom
for k in range(temps_A.shape[0]):
    zoom_ax.plot(t, I_umid[k, :], '-', linewidth=2, color=cmap(k))

fig.tight_layout()
plt.savefig(output_folder + 'prova_test_temp_umid.pdf')
#plt.show()
colors = ["turquoise", "salmon"]
cmap = LinearSegmentedColormap.from_list("turquoise_salmon", colors, N=temps_A.shape[0])

fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
for k in range(temps_A.shape[0]):
    ax[0].plot(t, I_temp_A[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[1].plot(t[:-1], f_temp_A[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[2].plot(t_num, temps_A[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))

#ax[0].set_title('Infected') 
ax[0].set_ylabel(r'Detected infected')
ax[0].set_xlabel(r'days')
#ax[1].set_title('Transmission Rate') 
ax[1].set_ylabel(r'Rhs Model')
ax[1].set_xlabel('days')
#ax[2].set_title('Temperature')
ax[2].set_ylabel(r'Temperature [°C]')
#ax[2].set_ylabel(r'Relative Umidity [\%]')
ax[2].set_xlabel('days')
zoom_ax = inset_axes(ax[0], width="30%", height="30%", loc="upper right")  # Riquadro di zoom in alto a destra
zoom_ax.set_xlim([54, 59])  # Limiti per lo zoom sull'asse x (modifica a seconda del tuo intervallo)
zoom_ax.set_ylim([0.029, 0.0315])  # Limiti per lo zoom sull'asse y (modifica a seconda del tuo intervallo)
zoom_ax.tick_params(axis='both', which='major', labelsize=7)
# Loop per plottare i dati anche nel riquadro di zoom
for k in range(temps_A.shape[0]):
    zoom_ax.plot(t, I_temp_A[k, :], '-', linewidth=2, color=cmap(k))

fig.tight_layout()
plt.savefig(output_folder + 'prova_test_temp_amplitude_f.pdf')
#plt.show()

fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
for k in range(temps_f.shape[0]):
    ax[0].plot(t, I_temp_f[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[1].plot(t[:-1], f_temp_f[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[2].plot(t_num, temps_f[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))

#ax[0].set_title('Infected') 
ax[0].set_ylabel(r'Detected infected')
ax[0].set_xlabel(r'days')
#ax[1].set_title('Transmission Rate') 
ax[1].set_ylabel(r'Rhs Model')
ax[1].set_xlabel('days')
#ax[2].set_title('Temperature')
ax[2].set_ylabel(r'Temperature [°C]')
#ax[2].set_ylabel(r'Relative Umidity [\%]')
ax[2].set_xlabel('days')
zoom_ax = inset_axes(ax[0], width="30%", height="30%", loc="upper right")  # Riquadro di zoom in alto a destra
zoom_ax.set_xlim([49, 63])  # Limiti per lo zoom sull'asse x (modifica a seconda del tuo intervallo)
zoom_ax.set_ylim([0.020, 0.0315])  # Limiti per lo zoom sull'asse y (modifica a seconda del tuo intervallo)
zoom_ax.tick_params(axis='both', which='major', labelsize=7)
# Loop per plottare i dati anche nel riquadro di zoom
for k in range(temps_A.shape[0]):
    zoom_ax.plot(t, I_temp_f[k, :], '-', linewidth=2, color=cmap(k))

fig.tight_layout()
plt.savefig(output_folder + 'prova_test_temp_frequency_f.pdf')
#plt.show()

fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
for k in range(temps_phi.shape[0]):
    ax[0].plot(t, I_temp_phi[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[1].plot(t[:-1], f_temp_phi[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[2].plot(t_num, temps_phi[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))

#ax[0].set_title('Infected') 
ax[0].set_ylabel(r'Detected infected')
ax[0].set_xlabel(r'days')
#ax[1].set_title('Transmission Rate') 
ax[1].set_ylabel(r'Rhs Model')
ax[1].set_xlabel('days')
#ax[2].set_title('Temperature')
ax[2].set_ylabel(r'Temperature [°C]')
#ax[2].set_ylabel(r'Relative Umidity [\%]')
ax[2].set_xlabel('days')
zoom_ax = inset_axes(ax[0], width="30%", height="30%", loc="upper right")  # Riquadro di zoom in alto a destra
zoom_ax.set_xlim([53, 59])  # Limiti per lo zoom sull'asse x (modifica a seconda del tuo intervallo)
zoom_ax.set_ylim([0.029, 0.032])  # Limiti per lo zoom sull'asse y (modifica a seconda del tuo intervallo)
zoom_ax.tick_params(axis='both', which='major', labelsize=7)
# Loop per plottare i dati anche nel riquadro di zoom
for k in range(temps_A.shape[0]):
    zoom_ax.plot(t, I_temp_phi[k, :], '-', linewidth=2, color=cmap(k))

fig.tight_layout()
plt.savefig(output_folder + 'prova_test_temp_phase_f.pdf')
#plt.show()

fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
for k in range(temps_Tm.shape[0]):
    ax[0].plot(t, I_temp_Tm[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[1].plot(t[:-1], f_temp_Tm[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[2].plot(t_num, temps_Tm[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))

#ax[0].set_title('Infected') 
ax[0].set_ylabel(r'Detected infected')
ax[0].set_xlabel(r'days')
#ax[1].set_title('Transmission Rate') 
ax[1].set_ylabel(r'Rhs model')
ax[1].set_xlabel('days')
#ax[2].set_title('Temperature')
ax[2].set_ylabel(r'Temperature [°C]')
#ax[2].set_ylabel(r'Relative Umidity [\%]')
ax[2].set_xlabel('days')
zoom_ax = inset_axes(ax[0], width="30%", height="30%", loc="upper right")  # Riquadro di zoom in alto a destra
zoom_ax.set_xlim([50, 65])  # Limiti per lo zoom sull'asse x (modifica a seconda del tuo intervallo)
zoom_ax.set_ylim([0.022, 0.032])  # Limiti per lo zoom sull'asse y (modifica a seconda del tuo intervallo)
zoom_ax.tick_params(axis='both', which='major', labelsize=7)
# Loop per plottare i dati anche nel riquadro di zoom
for k in range(temps_A.shape[0]):
    zoom_ax.plot(t, I_temp_Tm[k, :], '-', linewidth=2, color=cmap(k))

fig.tight_layout()
plt.savefig(output_folder + 'prova_test_temp_Tmean_f.pdf')
#plt.show()

fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
colors = ["goldenrod", "forestgreen"]
cmap = LinearSegmentedColormap.from_list("goldenrod_forestgreen", colors, N=temps_A.shape[0])
for k in range(umids.shape[0]):
    ax[0].plot(t, I_umid[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[1].plot(t[:-1], f_umid[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))
    ax[2].plot(t_num, umids[k,:], '-',linewidth = 2, label = 'Real', color = cmap(k))

#ax[0].set_title('Infected') 
ax[0].set_ylabel(r'Detected infected')
ax[0].set_xlabel(r'days')
#ax[1].set_title('Transmission Rate') 
ax[1].set_ylabel(r'Rhs Model')
ax[1].set_xlabel('days')
#ax[2].set_title('Temperature')
#ax[2].set_ylabel(r'Temperature [°C]')
ax[2].set_ylabel(r'Relative Umidity [\%]')
ax[2].set_xlabel('days')
zoom_ax = inset_axes(ax[0], width="30%", height="30%", loc="upper right")  # Riquadro di zoom in alto a destra
zoom_ax.set_xlim([50, 62])  # Limiti per lo zoom sull'asse x (modifica a seconda del tuo intervallo)
zoom_ax.set_ylim([0.022, 0.032])  # Limiti per lo zoom sull'asse y (modifica a seconda del tuo intervallo)
zoom_ax.tick_params(axis='both', which='major', labelsize=7)
# Loop per plottare i dati anche nel riquadro di zoom
for k in range(temps_A.shape[0]):
    zoom_ax.plot(t, I_umid[k, :], '-', linewidth=2, color=cmap(k))

fig.tight_layout()
plt.savefig(output_folder + 'prova_test_temp_umid_f.pdf')
#plt.show()

