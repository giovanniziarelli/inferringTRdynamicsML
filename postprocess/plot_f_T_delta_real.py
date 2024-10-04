#!/usr/bin/env python3

# Forward evaluations of reconstructed model 

#%% Import modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import argparse
import os
import sys
import pickle
import random
import math
import shutil
from scipy.linalg import lstsq
from cycler import cycler
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Set the color cycle to the "Accent" palette
colors = plt.cm.tab20c.colors
# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')

b_ref = 0.25

Tmin = 5
Tmax = 17

Umin = 60
Umax = 85

def normalize_forw(v, v_min, v_max, axis = None):
    return (2.0*v - v_min - v_max) / (v_max - v_min)

output_folder = '' # ex '/home/giovanni/Desktop/LDNets/pprocess_images/forw_exp/' 
if os.path.exists(output_folder) == False:
    os.mkdir(output_folder)

folder = '' # ex '/home/giovanni/Desktop/LDNets/save_neurons_4_196_lay2_umidity_Tobs_50_trialsim0/'
mpl.rcParams["figure.constrained_layout.use"] = True
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
plt.rc('axes', prop_cycle=cycler(color=colors))

width_pixels  = 1500
height_pixels = 1000

# Desired DPI
dpi = 500

# Calculate figure size in inches
width_in_inches  = width_pixels / dpi
height_in_inches = height_pixels / dpi

plt.set_cmap('seismic')

NNdyn_load = tf.keras.models.load_model(folder + 'NNdyn')

delta = 0 # normalized inputs in the network
beta_vals = 50
beta_vec = np.linspace(0,1.5/b_ref,beta_vals) # normalized inputs in the network
n_temps = 10
X_temp = delta * np.zeros((beta_vals * n_temps, 4))
T = np.linspace(Tmin-5, Tmax+8, n_temps)
U = 72.5

for k in range(n_temps):
    X_temp[beta_vals * k:beta_vals*(k+1), 0] = beta_vec.copy()
    X_temp[beta_vals * k:beta_vals*(k+1), 1] = normalize_forw(T[k], Tmin, Tmax) 
    X_temp[beta_vals * k:beta_vals*(k+1), 2] = normalize_forw(U, Umin, Umax) 

predictions_temp = NNdyn_load.predict(X_temp)

plt.figure(figsize=(1.6*width_in_inches, 1.6*height_in_inches), dpi=dpi)
plt.set_cmap('seismic')
colors = ["paleturquoise", "tomato"]
cmap = LinearSegmentedColormap.from_list("paleturquoise_tomato", colors, N=n_temps)
for k in range(n_temps):
    plt.plot(beta_vec*b_ref,predictions_temp[k*beta_vals:(k+1)*beta_vals], color = cmap(k), linewidth=2.7)
plt.legend(frameon=False)
plt.xlim([0, 1.5])

plt.axhline(y=0, color='black', linestyle='--', label='x = 0', linewidth=2.5, alpha=0.5)
plt.xlabel(r'$\beta$', fontsize=12)
plt.ylabel(r'$f_{\mathrm{nn}}(\beta, T^*, \bar{U}, \bar{\delta})$', fontsize=12)
plt.savefig(os.path.join(output_folder, f'temps_va.png'), format='png')
plt.close()

delta = 0 # normalized inputs in the network
beta_vals = 50
beta_vec = np.linspace(0,1.5/b_ref,beta_vals) # normalized inputs in the network
n_temps = 20
X_temp = delta * np.ones((beta_vals * n_temps, 4))
T = 12 
U = np.linspace(Umin-10, Umax+10, n_temps)

for k in range(n_temps):
    X_temp[beta_vals * k:beta_vals*(k+1), 0] = beta_vec.copy()
    X_temp[beta_vals * k:beta_vals*(k+1), 1] = normalize_forw(T, Tmin, Tmax) 
    X_temp[beta_vals * k:beta_vals*(k+1), 2] = normalize_forw(U[k], Umin, Umax) 

predictions_temp = NNdyn_load.predict(X_temp)

plt.figure(figsize=(1.6*width_in_inches, 1.6*height_in_inches), dpi=dpi)
plt.set_cmap('seismic')
colors = ["goldenrod", "forestgreen"]
cmap = LinearSegmentedColormap.from_list("goldenrod_forestgreen", colors, N=n_temps)
for k in range(n_temps):
    plt.plot(beta_vec*b_ref,predictions_temp[k*beta_vals:(k+1)*beta_vals], color = cmap(k), linewidth=2.7)
plt.legend(frameon=False)
plt.axhline(y=0, color='black', linestyle='--', label='x = 0', linewidth=2.5, alpha=0.5)
plt.xlim([0, 1.5])
plt.xlabel(r'$\beta$', fontsize=12)
plt.ylabel(r'$f_{\mathrm{nn}}(\beta, \bar{T}, U^*, \bar{\delta})$', fontsize=12)
plt.savefig(os.path.join(output_folder, f'umids_va.png'), format='png')
plt.close()

delta = 0 # normalized inputs in the network
beta_vals = 50
beta_vec = np.linspace(0,1.5/b_ref,beta_vals) # normalized inputs in the network
n_temps = 20
X_temp = delta * np.ones((beta_vals * n_temps, 4))
T = 12
U = 82.5
delta_vec = np.linspace(-0.1, 0.1, n_temps)
for k in range(n_temps):
    X_temp[beta_vals * k:beta_vals*(k+1), 0] = beta_vec.copy()
    X_temp[beta_vals * k:beta_vals*(k+1), 1] = normalize_forw(T, Tmin, Tmax) 
    X_temp[beta_vals * k:beta_vals*(k+1), 2] = normalize_forw(U, Umin, Umax) 
    X_temp[beta_vals * k:beta_vals*(k+1), 3] = delta_vec[k]

predictions_temp = NNdyn_load.predict(X_temp)
b_ref_index = np.argmin((beta_vec -0.7*0.25 ))  # trova l'indice di b_ref in beta_vec
dy = predictions_temp[b_ref_index + 1] - predictions_temp[b_ref_index - 1]  # differenza vicini
dx = beta_vec[b_ref_index + 1] - beta_vec[b_ref_index - 1]
slope = dy / dx
plt.figure(figsize=(1.6*width_in_inches, 1.6*height_in_inches), dpi=dpi)
plt.set_cmap('seismic')
colors = ["lavender", "orchid"]
cmap = LinearSegmentedColormap.from_list("lavender_orchid", colors, N=n_temps)
for k in range(n_temps):
    plt.plot(beta_vec*b_ref,predictions_temp[k*beta_vals:(k+1)*beta_vals], color = cmap(k), linewidth=2.7)
plt.legend(frameon=False)
plt.axhline(y=0, color='black', linestyle='--', label='x = 0', linewidth=2.5, alpha=0.5)
plt.xlim([0, 1.5])
plt.xlabel(r'$\beta$', fontsize=12)
plt.ylabel(r'$f_{\mathrm{nn}}(\beta, \bar{T}, \bar{U}, \delta^*)$', fontsize=12)
plt.savefig(os.path.join(output_folder, f'delta_va.png'), format='png')
plt.close()
