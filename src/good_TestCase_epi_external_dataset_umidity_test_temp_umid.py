#!/usr/bin/env python3

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
import SirHelperFunctions as shf
from scipy.linalg import lstsq
from cycler import cycler
from scipy.optimize import curve_fit

# Set the color cycle to the "Accent" palette
colors = plt.cm.tab20c.colors
# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')

import utils
import optimization

# Class constraint for IC when estimated
class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)
#%% Set some hyperparameters
dt = 0.2
t_max = 12
b_ref = 0.25#0.5#0.025#0.7#0.77#0.1672#0.92851
tau = 0
layers = 2
err = 1e-1#1e-1
trial = 4

T_obs_vec = [29, 36, 43, 50, 57, 64, 71, 78, 85, 91]
T_obs = T_obs_vec[trial-1]#71#57#99 

#%% Model parameters
alpha = 1 / 1.5#5#1.5
gamma = 1 / 1.2#10#1.2

plot_training_set = 1
save_train = 1
save_testg = 1
various_IC = 0
new_cases_loss = 1
estim_IC = 0

neurons = 4 
T_fin = 196#70
N_weeks = 28 
dt_base = 1#T_fin 
trial_sim = 4 

# Creating saving folders
folder = '/home/giovanni/Desktop/LDNets/temp_prova_neurons_' + str(neurons) + '_' + str(T_fin) + '_lay' + str(layers) + '_umidity_Tobs_' + str(T_obs)+ '_trialsim'+ str(trial_sim)+ '_rd_2/'

if os.path.exists(folder) == False:
    os.mkdir(folder)
folder_train = folder + 'train/'
folder_testg = folder + 'testg/'

num_latent_states = 1
num_latent_params = 1

# Loading datasets
input_dataset_path_temp = '/home/giovanni/Desktop/LDNets/italian-temperatures/tmedia_national_length'+str(T_fin)+'.csv'#temperature_n.pkl'
input_dataset_extern_temp = np.loadtxt(input_dataset_path_temp)
input_dataset_path_umid = '/home/giovanni/Desktop/LDNets/italian-temperatures/umid_national_length'+str(T_fin)+'.csv'#temperature_n.pkl'
input_dataset_extern_umid = np.loadtxt(input_dataset_path_umid)
output_dataset_path = '/home/giovanni/Desktop/LDNets/influnet/data-aggregated/epidemiological_data/processed_output_new_cases_'+str(N_weeks)+'_weeks.csv'#temperature_n.pkl'
output_dataset_extern = np.loadtxt(output_dataset_path)

problem = {
    "input_parameters": [],
    "input_signals": [
        { "name": "T" },
        { "name": "U" }
    ],
    "output_fields": [
        { "name": "S" },
        { "name": "E" },
        { "name": "I" }
    ]
}
#%% Define problem
normalization = {
    'time': {
        'time_constant' : dt_base
    },
    'input_signals': {
        'T': {'min': 0, 'max' : 20},#{ 'min': 0, 'max': 4}
        #'T': {'min': 0, 'max' : 20},#{ 'min': 0, 'max': 4}
        'U': {'min': 40, 'max' : 100}#{ 'min': 0, 'max': 4}
        #'U': {'min': 50, 'max' : 100}#{ 'min': 0, 'max': 4}
        #'T': {'min': 1, 'max' : 3},#{ 'min': 0, 'max': 4}
        #'U': {'min': 4, 'max' : 4.5}#{ 'min': 0, 'max': 4}
    },
    'output_fields': {
        'S': { 'min': 0, "max": 1 },
        'E': { 'min': 0, "max": 1 },
        'I': { 'min': 0, "max": 1 }
    }
}

#%% Dataset parameters
n_size = input_dataset_extern_temp.shape[0]#6
T_mean = 16
dt = 1

length_period = 7
W_obs = int(T_obs / length_period)
weeks = int(T_fin / length_period)
t = np.arange(0, T_fin+dt, dt)[None,:]

dt_num = 0.5#0.1#0.05#0.2
t_num = np.arange(0, T_fin+dt_num, dt_num)[None, :]
print('inp temp shape')
print(input_dataset_extern_temp.shape)
training_var_numpy_orig = np.stack((input_dataset_extern_temp, input_dataset_extern_umid), axis = 2) #T_cos(T_mean, f, A, t)
undetection_mean_value  = 0.23
cases                   = output_dataset_extern / undetection_mean_value

def epiModel_rhs(state, beta): # state dim (samples, 3), beta dim (samples,1)

    dSdt = - beta * tf.expand_dims(state[:,0],axis=1) * tf.expand_dims(state[:,2],axis=1)
    dEdt = beta * tf.expand_dims(state[:,0],axis=1) * tf.expand_dims(state[:,2],axis=1) - alpha * tf.expand_dims(state[:,1],axis=1)
    dIdt = alpha * tf.expand_dims(state[:,1],axis=1) - gamma* tf.expand_dims(state[:,2],axis=1)

    return tf.concat([dSdt, dEdt, dIdt], axis = 1)

beta_guess = b_ref 
beta_guess_IC = b_ref * np.ones((n_size,1))

# Filtering input coeffiecients
def savgol_coeffs(window_length, polyorder):
    half_window = (window_length - 1) // 2
    # Precompute coefficients
    B = np.mat([[k**i for i in range(polyorder + 1)] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(B).A[0]
    return m

def savgol_filter_row(tensor_row, window_length, polyorder):
    if window_length % 2 == 0 or window_length <= polyorder:
        raise ValueError("window_length must be odd and greater than polyorder")
    
    # Get the Savitzky-Golay filter coefficients
    coeffs = savgol_coeffs(window_length, polyorder)
    coeffs = tf.constant(coeffs, dtype=tensor_row.dtype)
    
    # Apply convolution
    half_window = (window_length - 1) // 2
    tensor_padded = tf.pad(tensor_row, [[half_window, half_window]], mode='REFLECT')
    tensor_filtered = tf.nn.conv1d(tensor_padded[tf.newaxis, :, tf.newaxis], coeffs[:, tf.newaxis, tf.newaxis], stride=1, padding='VALID')
    
    return tensor_filtered[0, :, 0]

def savgol_filter(tensor, window_length, polyorder):
    def apply_filter(row):
        return savgol_filter_row(row, window_length, polyorder)
    
    filtered_rows = tf.map_fn(apply_filter, tensor, dtype=tensor.dtype)
    
    return filtered_rows

def smoothing(data):
    smoothed_data = savgol_filter(data, 197, 2)
    #smoothed_data = tf.math.log(smoothed_data + 1)#smoothed_data
    return smoothed_data

def smoothing_cases(data):
    smoothed_data = savgol_filter(data, 13, 2)
    #smoothed_data = tf.math.log(smoothed_data + 1)#smoothed_data
    return smoothed_data

def weekly_avg(T):
    print(T.shape)
    T_reshaped = np.reshape(T[:,:-1], (T.shape[0], T.shape[1]//7, 7))
    return np.mean(T_reshaped, axis = 2)

print(training_var_numpy_orig.shape)
# Deleting possible Nan values
mask = np.isnan(training_var_numpy_orig)
indexes = []
for ii in range(mask.shape[0]):
    if mask[ii].any() == True:
        indexes.append(ii)
        continue
#training_var_numpy = weekly_avg(training_var_numpy)
training_var_numpy_orig[:,:,0] = smoothing(training_var_numpy_orig[:,:,0])
training_var_numpy_orig[:,:,1] = smoothing(training_var_numpy_orig[:,:,1])
print(training_var_numpy_orig.shape)
print(cases.shape)
#cases = smoothing_cases(cases)
training_var_numpy_orig = np.delete(training_var_numpy_orig, indexes, 0)
print(training_var_numpy_orig.shape)
#training_var_numpy = np.delete(smoothing(training_var_numpy), indexes, 0)
training_var_numpy = np.zeros((training_var_numpy_orig.shape[0], t_num.shape[1], training_var_numpy_orig.shape[2]))
training_var_numpy[:,:,0] = np.array([np.interp(t_num.squeeze(), t.squeeze(), training_var_numpy_orig[k,:,0]) for k in range(n_size - len(indexes))])
training_var_numpy[:,:,1] = np.array([np.interp(t_num.squeeze(), t.squeeze(), training_var_numpy_orig[k,:,1]) for k in range(n_size - len(indexes))])
print(t_num.shape)
print(t.shape)
print(training_var_numpy.shape)
#training_var_numpy = np.array([np.interp(t_num[0,:-1].squeeze(), t[0,:-1].squeeze(), training_var_numpy[k,:]) for k in range(n_size - len(indexes))])
cases = np.delete(cases, indexes, 0)
n_size = n_size - len(indexes)

print(np.min(training_var_numpy, axis = 1), np.max(cases, axis = 1))

plot_temps = 1
if plot_temps:
    seasons = [r'2010-2011', r'2011-2012',r'2012-2013',r'2013-2014',r'2014-2015',r'2015-2016',r'2016-2017',r'2017-2018',r'2018-2019',r'2019-2020']
    seasons_r = ['2010-2011', '2011-2012','2012-2013','2013-2014','2014-2015','2015-2016','2016-2017','2017-2018','2018-2019','2019-2020']
    mpl.rcParams["figure.constrained_layout.use"] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
    plt.rc('axes', prop_cycle=cycler(color=colors))
    #plt.set_cmap('Accent')
    
    width_pixels  = 300#600#337
    height_pixels = 200#500#266

    # Desired DPI
    dpi = 100

    # Calculate figure size in inches
    width_in_inches  = width_pixels / dpi
    height_in_inches = height_pixels / dpi
    
    plt.figure(figsize=(2*width_in_inches, 2*height_in_inches), dpi=dpi)
    plt.set_cmap('Accent')

    for i, um in enumerate(training_var_numpy[:,:,1]):
        plt.plot(np.linspace(0, T_fin, len(um)), um, linewidth = 2, label=seasons_r[i])
    plt.legend(frameon=False)
    plt.ylabel(r'Relative Umidity [\%]')
    plt.xlabel(r'days')

    # Salvataggio del grafico come PDF
    plt.savefig(os.path.join(folder, f'umid_all.pdf'), format='pdf')
# Chiudere la figura per liberare memoria
    plt.close()

    plt.figure(figsize=(2*width_in_inches, 2*height_in_inches), dpi=dpi)
    plt.set_cmap('Accent')
    for i, tm in enumerate(training_var_numpy[:,:,0]):
        plt.plot(np.linspace(0, T_fin, len(tm)),tm, linewidth = 2, label=seasons_r[i])
    plt.legend(frameon=False)
    plt.xlabel(r'days')
    plt.ylabel(r'Temperature [°C]')
    # Salvataggio del grafico come PDF
    plt.savefig(os.path.join(folder, f'temp_all.pdf'), format='pdf')
# Chiudere la figura per liberare memoria
    plt.close()


#   plt.plot(training_var_numpy[0,:,0])
#   plt.plot(training_var_numpy[1,:,0])
#   plt.plot(training_var_numpy[2,:,0])
#   plt.plot(training_var_numpy[3,:,0])
#   plt.plot(training_var_numpy[4,:,0])
#   plt.plot(training_var_numpy[5,:,0])
#   plt.plot(training_var_numpy[6,:,0])
#   plt.plot(training_var_numpy[7,:,0])
#   plt.show()

#   plt.plot(training_var_numpy[0,:,1])
#   plt.plot(training_var_numpy[1,:,1])
#   plt.plot(training_var_numpy[2,:,1])
#   plt.plot(training_var_numpy[3,:,1])
#   plt.plot(training_var_numpy[4,:,1])
#   plt.plot(training_var_numpy[5,:,1])
#   plt.plot(training_var_numpy[6,:,1])
#   plt.plot(training_var_numpy[7,:,1])
#   plt.show()

#   plt.plot(cases[0,:])
#   plt.plot(cases[1,:])
#   plt.plot(cases[2,:])
#   plt.plot(cases[3,:])
#   plt.plot(cases[4,:])
#   plt.plot(cases[5,:])
#   plt.plot(cases[6,:])
#   plt.plot(cases[7,:])
#   plt.show()

#   plt.hist(cases.flatten())
#   plt.close()

# Defining Datasets

# dividing training and testing set
num_train = 9
print('SHAPESSSS')
print(cases.shape)
print(training_var_numpy.shape)
# seed 7 = 2014-15 testg
np.random.seed(7)
shuffle_indexes = np.arange(training_var_numpy.shape[0])
np.random.shuffle(shuffle_indexes)
#print(np.random.shuffle(ind))# shuffle_indexes)
print(shuffle_indexes)

#if shuffling
training_var_numpy = training_var_numpy[shuffle_indexes]
cases = cases[shuffle_indexes]
dataset_train = {
        'times' : t_num.T, # [num_times]
        'inp_parameters' : None, # [num_samples x num_par]
        'inp_signals' : training_var_numpy[:num_train,:,:], # [num_samples x num_times x num_signals]
        'target_cases' : cases[:num_train],
        'num_times' : T_fin+1,
        'beta_guess' : beta_guess, 
        'time_vec' : t.T,
        'weeks' : weeks,
        'frac' : int(dt/dt_num)
}
dataset_testg_full = {
        'times' : t_num.T, # [num_times]
        'inp_parameters' : None, # [num_samples x num_par]
        'inp_signals' : training_var_numpy[num_train:,:,:], # [num_samples x num_times x num_signals]
        'target_cases' : cases[num_train:],
        'num_times' : T_fin+1,
        'beta_guess' : beta_guess, 
        'time_vec' : t.T,
        'weeks' : weeks,
        'frac' : int(dt/dt_num)
}
dataset_testg = utils.cut_dataset_epi_real(dataset_testg_full, T_obs)
# For reproducibility (delete if you want to test other random initializations)
tf.random.set_seed(0)
# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset_epi_real(dataset_train, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_testg, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_testg_full, problem, normalization, dt = None, num_points_subsample = None)
print(dataset_testg)
print('END')
#%% Define model
constraint_IC = ClipConstraint(0, 1)
# dynamics network
# Initial conditions
InitialValue_train = np.zeros((dataset_train['num_samples'], 1))
InitialValue_testg = np.zeros((dataset_testg['num_samples'], 1))
InitialValue_train[:,0] = 10/1000000# * np.ones_like(cases[:,0])#cases[:,0]#10/50000000#0.01
InitialValue_testg[:,0] = 10/1000000# * np.ones_like(cases[:,0])#cases[:,0]#10 / 50000000
IC_train = tf.Variable(InitialValue_train, trainable=False, constraint = constraint_IC)
IC_testg = tf.Variable(InitialValue_testg, trainable=False, constraint = constraint_IC)
print('IC testg')
tf.print(IC_testg)
tf.print(dataset_train['target_cases'])
tf.print(dataset_testg['target_cases'])
initial_lat_state_train = tf.Variable(np.ones((dataset_train['num_samples'],1)) * beta_guess, trainable=True)
#print(dataset_train)
initial_lat_state_testg = tf.Variable(np.ones((dataset_testg['num_samples'],1)) * beta_guess, trainable=True)

if num_latent_params > 0:
    lat_param_train = tf.Variable(0.5 * np.ones((dataset_train['num_samples'], num_latent_params)), trainable=True)
    lat_param_testg = tf.Variable(0.5 * np.ones((dataset_testg['num_samples'], num_latent_params)), trainable=True)
else:
    lat_param_train = None
    lat_param_testg = None

if num_latent_params > 0:
    input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']) + num_latent_params,)
else:
    input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)

NNdyn = tf.keras.Sequential([
            tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, input_shape = input_shape, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.0001)),
            tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.0001)),
            #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
            #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
            #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
            #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
            tf.keras.layers.Dense(num_latent_states, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.0001)),
        ])
NNdyn.summary()


def evolve_dynamics(dataset, initial_lat_state, lat_param): ##initial_state (n_samples x n_latent_state), delta (n_samples x num_lat_param)
    # intial condition
    lat_state = initial_lat_state
    lat_state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    lat_state_history = lat_state_history.write(0, lat_state)
    dt_ref = normalization['time']['time_constant']
    
    # time integration
    for i in tf.range(dataset['num_times'] - 1):
        inputs = [lat_state / b_ref, dataset['inp_signals'][:,i,:]]
        #tf.print(inputs)
        if num_latent_params > 0:
            inputs.append(lat_param -1 )
        lat_state = tf.math.maximum(lat_state + dt_num/dt_ref * NNdyn(tf.concat(inputs, axis = -1)), tf.zeros_like(lat_state))
        lat_state_history = lat_state_history.write(i + 1, lat_state)

    return tf.transpose(lat_state_history.stack(), perm=(1,0,2))

# initialize epi model parameters

def evolve_model(dataset, lat_state, IC):
    IC_E = tf.reshape((dataset['target_cases'][:,0] / alpha /7 ), [IC.shape[0],1])
    IC_S = tf.reshape(1 - IC - IC_E, [IC.shape[0], 1])
    IC_all = tf.concat([IC_S, IC_E, IC], 1)
    #tf.print(IC_all)
    state = IC_all
    state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    state_history = state_history.write(0, state)

    # time integration
    for i in tf.range(dataset['num_times'] - 1):
        state = state + dt_num * epiModel_rhs(state, lat_state[:, i, :])
        state_history = state_history.write(i + 1, state)

    return tf.transpose(state_history.stack(), perm=(1,0,2))

def LDNet(dataset, initial_lat_state, lat_param, IC):
    lat_states = evolve_dynamics(dataset,initial_lat_state, lat_param)
    return evolve_model(dataset, lat_states, IC)

#%% Loss function

def loss_exp(dataset, initial_lat_state, lat_param, IC):
    state = LDNet(dataset, initial_lat_state, lat_param, IC)
    #E = tf.transpose(state[:,::dataset['frac'] * 7,1])
    E = state[:,::dataset['frac'] * length_period ,1]#frac * 7
    S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    #new_cases_summed_weekly = tf.transpose(tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])]))
    #MSE = tf.reduce_mean(tf.square(new_cases_summed_weekly - dataset['target_cases']) / tf.square( tf.square( dataset['target_cases']) + 1e-6 )) 
    C = 10#80 
    MSE = tf.reduce_mean(tf.square(C * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - C * dataset['target_cases']))# / tf.square( (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:])+ 1e-6 )) 
    
    #MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']) / tf.square( 100 * dataset['target_cases'] )) 
    return MSE

def loss_exp_beta_abs(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    #E = tf.transpose(state[:,::dataset['frac'] * 7,1])
    E = state[:,::dataset['frac'] * length_period ,1]#frac * 7
    S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    #new_cases_summed_weekly = tf.transpose(tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])]))
    #MSE = tf.reduce_mean(tf.square(new_cases_summed_weekly - dataset['target_cases']) / tf.square( tf.square( dataset['target_cases']) + 1e-6 )) 
    
    MSE = tf.reduce_mean(tf.square(10000 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 10000 * dataset['target_cases']))# / tf.square( 100 * dataset['target_cases'] )) 
    
    #MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']) / tf.square( 100 * dataset['target_cases'] )) 
    return MSE

def loss_s_inf(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    err   = tf.reduce_mean(tf.maximum(tf.pow(10 * (0.7 - state[:,-1,0]), 5), tf.zeros_like(state[:,-1,0]))) 
    
    #MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']) / tf.square( 100 * dataset['target_cases'] )) 
    return err

def loss_s_inf_testg(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    err   = tf.reduce_mean(tf.maximum(tf.pow(10 * (0.85 - state[:,-1,0]), 5), tf.zeros_like(state[:,-1,0]))) 
    
    #MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']) / tf.square( 100 * dataset['target_cases'] )) 
    return err

def loss_exp_beta(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    #E = tf.transpose(state[:,::dataset['frac'] * 7,1])
    E = state[:,::dataset['frac'] * length_period ,1]#frac * 7
    S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    #new_cases_summed_weekly = tf.transpose(tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])]))
    #MSE = tf.reduce_mean(tf.square(new_cases_summed_weekly - dataset['target_cases']) / tf.square( tf.square( dataset['target_cases']) + 1e-6 )) 
    
    MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']) / tf.square( 100 * dataset['target_cases'] )) 
    
    #MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']) / tf.square( 100 * dataset['target_cases'] )) 
    return MSE

def val_exp(dataset, initial_lat_state, lat_param, IC):
    state = LDNet(dataset, initial_lat_state, lat_param, IC)
    E = tf.transpose(state[:,::dataset['frac'],1])
    new_cases_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])
    return tf.reduce_mean(tf.square((new_cases_summed_weekly) - (tf.transpose(dataset['target_cases']))))

def loss_H1(dataset, initial_lat_state, lat_param, IC):
    state = LDNet(dataset, initial_lat_state, lat_param, IC)
    I = tf.transpose(state[:,::dataset['frac'],2])
    state_v = state[:,::dataset['frac'],:]
    
    I_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(I[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])

    beta = evolve_dynamics(dataset, initial_lat_state, lat_param)
    beta = beta[:,::dataset['frac'],:]
    rhs = tf.transpose(tf.convert_to_tensor([epiModel_rhs(state_v[:,i,:],beta[:,i,0])[:,2] for i in range(T_fin)]))
    rhs_weekly = tf.convert_to_tensor([tf.reduce_mean(rhs[:, length_period*k: length_period*(k+1)], axis = 1) for k in range(dataset['weeks'] - 1)])
    rhs = tf.reduce_mean(tf.square(rhs_weekly - (I_summed_weekly[1:,:] - I_summed_weekly[:-1,:])/(dt * length_period)), axis = 0)
    
    MSE = nu_h1 * tf.reduce_mean(rhs) 
    return MSE

#%% Loss weights
nu = 0.5e-2#1.1e-2#1.5e-2#3e-1#1.8e-1#1e-1#i1e0 good 1e0#1e-7#1e-2#-4#1e0#1e2
nu_h1 = 0#1e-1 
nu_r1  = 5e-3#1.4e-3#1e-3#1.4e-3#1.7e-3#e-5
nu_p = 0#5e-3#5e-2#5e-2#1e-2#6
nu_p2 = 5e-2#5e-2#1e-3#6
nu_p3 = 0#1e-3#1e0#5e-1#1e0#1e-3#6
nu_s_inf = 0#1e-3#1e-2
alpha_reg = 0#5e-3#5e-3#1e-3#1e#1e-5#1e-2#1e-2#1e-1# 1e-8#1e-8#1e1#1e-2#1e-3#5e-5#5e-5#5e-6#4.7e-3
alpha_reg_IC = 0#1e-8#1e-4#1e-3
#%% Training
trainable_variables_train = NNdyn.variables + [initial_lat_state_train] 
if num_latent_params > 0:
    trainable_variables_train += [lat_param_train]
if estim_IC > 0:
    trainable_variables_train += [IC_train]

def weights_reg(NN):
    return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in NN.layers])/len(NN.layers)

loss = loss_exp

def loss_train():
    beta = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
    beta_coarse = beta[:,1::2*7,:]
    l = nu *loss_exp_beta(dataset_train, beta, IC_train)\
            + alpha_reg * weights_reg(NNdyn)\
            + nu_r1 * tf.reduce_mean(tf.square(lat_param_train -1))\
            + nu_p * tf.reduce_mean(tf.maximum(tf.pow(10 * (beta - 1), 5), tf.zeros_like(beta)))\
            + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - beta[:,1,:]) / dt_num))\
            + nu_s_inf * loss_s_inf(dataset_train, beta, IC_train)\
            + nu_p3 * tf.reduce_mean(tf.reduce_max(tf.square((beta_coarse[:,1:] - beta_coarse[:,:-1])/(dt_num*2*7 )), axis = 1))
    return l 

def loss_train_mse2():
    beta = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
    beta_coarse = beta[:,1::2*7,:]
    l = nu *loss_exp_beta_abs(dataset_train, beta, IC_train)\
            + alpha_reg * weights_reg(NNdyn)\
            + nu_r1 * tf.reduce_mean(tf.square(lat_param_train-1))\
            + nu_p * tf.reduce_mean(tf.maximum(tf.pow((beta - 1), 5), tf.zeros_like(beta)))\
            + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - beta[:,1,:]) / dt_num))\
            + nu_s_inf * loss_s_inf(dataset_train, beta, IC_train)\
            + nu_p3 * tf.reduce_mean(tf.reduce_max(tf.square((beta_coarse[:,1:] - beta_coarse[:,:-1])/(dt_num*2*7 )), axis = 1))
            #+ nu_p3 * tf.reduce_mean(tf.square((beta_coarse[:,1:] - beta_coarse[:,:-1])/(dt_num)))
    return l 
####nu = 1.1e-2#3e-1#1.8e-1#1e-1#i1e0 good 1e0#1e-7#1e-2#-4#1e0#1e2
####nu_h1 = 0#1e-1 
####nu_r1  = 1.4e-3#e-5
####nu_p = 5e-2#1e-2#6
####nu_p2 = 5e-2#1e-3#6
####alpha_reg = 1e-3#1e-3#1e#1e-5#1e-2#1e-2#1e-1# 1e-8#1e-8#1e1#1e-2#1e-3#5e-5#5e-5#5e-6#4.7e-3
####alpha_reg_IC = 0#1e-8#1e-4#1e-3
#####%% Training
####trainable_variables_train = NNdyn.variables + [initial_lat_state_train] 
####if num_latent_params > 0:
####    trainable_variables_train += [lat_param_train]
####if estim_IC > 0:
####    trainable_variables_train += [IC_train]

####def weights_reg(NN):
####    return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in NN.layers])/len(NN.layers)

####loss = loss_exp

####def loss_train():
####    beta = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
####    l = nu *loss_exp_beta(dataset_train, beta, IC_train)\
####            + alpha_reg * weights_reg(NNdyn)\
####            + nu_r1 * tf.reduce_mean(tf.square(lat_param_train -1))\
####            + nu_p * tf.reduce_mean(tf.maximum(tf.pow(beta - 1, 5), tf.zeros_like(beta)))\
####            + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - beta[:,1,:]) / dt_num))
####    return l 

####def loss_train_mse2():
####    beta = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
####    l = nu *loss_exp_beta_abs(dataset_train, beta, IC_train)\
####            + alpha_reg * weights_reg(NNdyn)\
####            + nu_r1 * tf.reduce_mean(tf.square(lat_param_train -1))\
####            + nu_p * tf.reduce_mean(tf.maximum(tf.pow(beta - 1, 5), tf.zeros_like(beta)))\
####            + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - beta[:,1,:]) / dt_num))
####    return l 

def val_train():
    beta = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
    l = loss_exp_beta(dataset_train, beta, IC_train)
    return l 

##loss_train = lambda: nu * loss(dataset_train, initial_lat_state_train, lat_param_train, IC_train)\
##        + alpha_reg * weights_reg(NNdyn)\
##        + nu_r1 * tf.reduce_mean(tf.square(lat_param_train - 1))\
        #+ nu_p * tf.reduce_mean(tf.reduce_sum(tf.square(evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train) -0.6), axis = 0))\
##        + nu_p * tf.maximum(tf.reduce_sum(tf.square(evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train) -0.6), axis = 0))\
##        + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)[:,1,:]) / dt_num))

#val_train = lambda: val_exp(dataset_train, initial_lat_state_train, lat_param_train, IC_train)         
         
val_metric = val_train

opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train, val_metric)

num_epochs_Adam_train = 500#500 good
num_epochs_BFGS_train = 500#5000

print('training (Adam)...')
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=1e-3))#1e-3))#1e-2 good
print('training (BFGS)...')
opt_train.optimize_BFGS(num_epochs_BFGS_train)

#optimization 2

opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train_mse2, val_metric)

num_epochs_Adam_train = 500#1000#500 good
num_epochs_BFGS_train = 500#3000

#print('training (Adam)...')
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=1e-7))#1e-3))#1e-2 good
print('training (BFGS)...')
opt_train.optimize_BFGS(num_epochs_BFGS_train)



#%% Estimation

if num_latent_params>0:
    num_sample_test = dataset_testg['num_samples']
    #trainable_variables_testg += [IC_testg]
    
    #%% Constants
    nu_r2 = 0.2e0#6.2e0#1e-3#6.5e-3#5.6e0#3.8e0#5e-1#1.85e-1#5e-3#1e-5#quattro ordini in meno
    nu_loss_testg = 7e-2#1e-1#1e-2#8e-2#5e-3#1e-3#1.7e-1#prima era 1.7e-2
    nu_b0 = 5e-2#7e-4#5.6e-3#5.3e-3 #prima era e-3#quattro ordini in meno
    
    C_post    = tf.linalg.inv(tfp.stats.covariance(lat_param_train))
    C_post_b0 = tf.linalg.inv(tfp.stats.covariance(initial_lat_state_train))
    latent_train_mean = tf.reduce_mean(lat_param_train)
    #l_n = latent_train_mean.numpy()
    initial_lat_state_train_mean = tf.reduce_mean(initial_lat_state_train)
    #i_n = initial_lat_state_train_mean.numpy()

    
    trainable_variables_testg = [initial_lat_state_testg] 
    trainable_variables_testg += [lat_param_testg]
    
    #C_post_IC = tf.linalg.inv(tfp.stats.covariance(IC_train))
    #IC_train_mean = tf.repeat(tf.reshape(tf.reduce_mean(IC_train, axis = 0), [1,2]), num_sample_test, axis = 1)
    #IC_train_mean = tf.reduce_mean(IC_train)

    def loss_testg():
        beta = evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg)
        l = nu_loss_testg *loss_exp_beta(dataset_testg, beta, IC_testg)\
                + nu_r2 * tf.reduce_mean(tf.transpose(lat_param_testg - latent_train_mean ) * C_post * (lat_param_testg - latent_train_mean))\
                + nu_b0 * tf.reduce_mean(tf.transpose(initial_lat_state_testg - initial_lat_state_train_mean) * C_post_b0 * (initial_lat_state_testg - initial_lat_state_train_mean))\
                + 0*nu_s_inf * loss_s_inf_testg(dataset_testg, beta, IC_testg)\
                + nu_p * tf.reduce_mean(tf.maximum(tf.pow(beta - 1, 5), tf.zeros_like(beta)))\
                + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_testg - beta[:,1,:]) / dt_num))
        return l 
    def loss_testg_abs():
        beta = evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg)
        l = nu_loss_testg *loss_exp_beta_abs(dataset_testg, beta, IC_testg)\
                + nu_r2 * tf.reduce_mean(tf.transpose(lat_param_testg - latent_train_mean ) * C_post * (lat_param_testg - latent_train_mean))\
                + nu_b0 * tf.reduce_mean(tf.transpose(initial_lat_state_testg - initial_lat_state_train_mean) * C_post_b0 * (initial_lat_state_testg - initial_lat_state_train_mean))\
                + 0*nu_s_inf * loss_s_inf_testg(dataset_testg, beta, IC_testg)\
                + nu_p * tf.reduce_mean(tf.maximum(tf.pow(beta - 1, 5), tf.zeros_like(beta)))\
                + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_testg - beta[:,1,:]) / dt_num))
        return l 
########loss_testg = lambda: nu_loss_testg * loss(dataset_testg, initial_lat_state_testg, lat_param_testg, IC_testg)\
########        + nu_r2 * tf.reduce_mean(tf.transpose(lat_param_testg - latent_train_mean ) * C_post * (lat_param_testg - latent_train_mean))\
########        + nu_b0 * tf.reduce_mean(tf.transpose(initial_lat_state_testg - initial_lat_state_train_mean) * C_post_b0 * (initial_lat_state_testg - initial_lat_state_train_mean))\
########        + nu_p * tf.reduce_mean(tf.reduce_sum(tf.square(evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg) - 1), axis = 0))\
########        + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_testg - evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg)[:,1,:]) / dt_num))

            #+ nu_IC * tf.reduce_mean(tf.transpose(IC_testg - IC_train_mean) * C_post_IC * (IC_testg - IC_train_mean))

    opt_testg = optimization.OptimizationProblem(trainable_variables_testg, loss_testg_abs, loss_testg)

    num_epochs_Adam_testg = 5000#1000
    num_epochs_BFGS_testg = 1000#3000
    
    nu_loss_testg = 9e-2#1e-2#8e-2#5e-3#1e-3#1.7e-1#prima era 1.7e-2
    print('training (Adam)...')
    opt_testg.optimize_keras(num_epochs_Adam_testg, tf.keras.optimizers.Adam(learning_rate=1e-2))# tf.keras.optimizers.Adam(learning_rate=5e-3))
    print('training (BFGS)...')
    opt_testg.optimize_BFGS(num_epochs_BFGS_testg)

    #%% Prediction
#   opt_testg = optimization.OptimizationProblem(trainable_variables_testg, loss_testg_abs, loss_testg)

#   num_epochs_Adam_testg = 5000
#   num_epochs_BFGS_testg = 3000
#   
#   print('training (Adam)...')
#   opt_testg.optimize_keras(num_epochs_Adam_testg, tf.keras.optimizers.Adam(learning_rate=1e-6))# tf.keras.optimizers.Adam(learning_rate=5e-3))
#   print('training (BFGS)...')
#   opt_testg.optimize_BFGS(num_epochs_BFGS_testg)

    state_predict = LDNet(dataset_testg_full, initial_lat_state_testg, lat_param_testg, IC_testg)
    print('Cpost: ', C_post)
    #print('Cpost_IC: ', C_post_IC)
    
    if num_train == 9:
        if os.path.exists(folder_train) == False:
            os.mkdir(folder_train)

        if os.path.exists(folder_testg) == False:
            os.mkdir(folder_testg)

        num_test_time_series = 10
        initial_lat_state_testg_numpy = initial_lat_state_testg.numpy().squeeze() 
        lat_param_testg_numpy = lat_param_testg.numpy().squeeze() 
        IC_testg_numpy = IC_testg.numpy().squeeze() 

        initial_lat_state_testg_tests = tf.Variable(initial_lat_state_testg_numpy * np.ones((num_test_time_series,1)), trainable = True) 
        lat_param_testg_tests         = tf.Variable(lat_param_testg_numpy * np.ones((num_test_time_series,1)), trainable = True)
        
        InitialValue_testg_tests = np.zeros((num_test_time_series, 1))
        InitialValue_testg_tests[:,0] = 10/1000000# * np.ones_like(cases[:,0])#cases[:,0]#10 / 50000000
        IC_testg_tests = tf.Variable(InitialValue_testg_tests, trainable=False, constraint = constraint_IC)
       
        print(IC_testg_tests.shape)
        print(lat_param_testg_tests.shape)
        print(initial_lat_state_testg_tests.shape)
        # least square reconstruction
        def cosinusoid(t, A, B, C, D):
            return A * np.cos(B * t + C) + D

        initial_guess = [2, np.pi / T_fin, 0 , 10]
        print(training_var_numpy[num_train:,:,0].shape)
        print(t_num.shape)
        params, params_covariance = curve_fit(cosinusoid, t_num.squeeze(), training_var_numpy[num_train:,:,0].squeeze(), p0=initial_guess)
        A_fit, B_fit, C_fit, D_fit = params
        print(A_fit)
        print(B_fit)
        print(C_fit)
        print(D_fit)

        for j in range(4):
            temp_var_numpy = np.zeros((num_test_time_series,training_var_numpy.shape[1]))
            if  j== 0:
                for i in range(num_test_time_series):
                    temp_var_numpy[i, :] = cosinusoid(t_num, A_fit * 1.2 / num_test_time_series*i, B_fit, C_fit, D_fit)
            
            elif  j== 1:
                for i in range(num_test_time_series):
                    temp_var_numpy[i, :] = cosinusoid(t_num, A_fit , B_fit* 2 / num_test_time_series*i, C_fit, D_fit)
            
            elif  j== 2:
                for i in range(num_test_time_series):
                    temp_var_numpy[i, :] = cosinusoid(t_num, A_fit , B_fit, C_fit* 1.2 / num_test_time_series*i, D_fit)
            
            elif  j== 3:
                for i in range(num_test_time_series):
                    temp_var_numpy[i, :] = cosinusoid(t_num, A_fit, B_fit, C_fit, D_fit - 2 + 4 / num_test_time_series*i )
            

            print(temp_var_numpy.shape)
            print(training_var_numpy[9:,:,1].shape)
            print(np.repeat(training_var_numpy[num_train:, :, 1], num_test_time_series, axis = 0).shape)
            testg_var_numpy_temp = np.stack((temp_var_numpy, np.repeat(training_var_numpy[num_train:, :, 1], num_test_time_series, axis = 0)), axis = 2) #T_cos(T_mean, f, A, t)
            print(testg_var_numpy_temp.shape) 
            dataset_testg_temp = {
                    'times' : t_num.T, # [num_times]
                    'inp_parameters' : None, # [num_samples x num_par]
                    'inp_signals' : testg_var_numpy_temp, # [num_samples x num_times x num_signals]
                    'target_cases' : np.repeat(cases[num_train:], num_test_time_series, axis=0),
                    'num_times' : T_fin+1,
                    'beta_guess' : beta_guess, 
                    'time_vec' : t.T,
                    'weeks' : weeks,
                    'frac' : int(dt/dt_num)
            }

            width_pixels  = 200#600#337
            height_pixels = 200#500#266

            # Desired DPI
            dpi = 100

            # Calculate figure size in inches
            width_in_inches  = width_pixels / dpi
            height_in_inches = height_pixels / dpi

            state_predict_temp = LDNet(dataset_testg_temp, initial_lat_state_testg_tests, lat_param_testg_tests, IC_testg_tests).numpy()
            beta_predict_temp  = evolve_dynamics(dataset_testg_temp, initial_lat_state_testg_tests, lat_param_testg_tests).numpy()
            fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
            for k in range(num_test_time_series):    
                ax[0].plot(np.arange(1,state_predict_temp.shape[1]+1), state_predict_temp[k,:,2], '-',linewidth = 2, label = 'Real')
                ax[1].plot(np.arange(1,beta_predict_temp.shape[1]+1), beta_predict_temp[k,:], '-',linewidth = 2, label = 'Real')
                ax[2].plot(np.arange(1,testg_var_numpy_temp.shape[1]+1), testg_var_numpy_temp[k,:,0], '-',linewidth = 2, label = 'Real')
            
            #ax[0].set_title('Infected') 
            ax[0].set_ylabel(r'Total infected')
            ax[0].set_xlabel(r'days')
            #ax[1].set_title('Transmission Rate') 
            ax[1].set_ylabel(r'Transmission rate')
            ax[1].set_xlabel('days')
            #ax[2].set_title('Temperature')
            ax[2].set_ylabel(r'Temperature [°C]')
            ax[2].set_xlabel('days')
            fig.tight_layout()
            plt.savefig(folder + 'prova_' + str(j) + '_test_temp_amplitude.pdf')
            plt.show()
            
            np.savetxt(folder_testg + 'temps_'+str(j)+'.txt', testg_var_numpy_temp[:,:,0])
            np.savetxt(folder_testg + 'S_rec_testg_temp_'+str(j)+'.txt', state_predict_temp[:,:,0])
            np.savetxt(folder_testg + 'E_rec_testg_temp_'+str(j)+'.txt', state_predict_temp[:,:,1])
            np.savetxt(folder_testg + 'I_rec_testg_temp_'+str(j)+'.txt', state_predict_temp[:,:,2])
            np.savetxt(folder_testg + 'beta_rec_temp_testg_'+str(j)+'.txt', beta_predict_temp.squeeze())
        
        # least square reconstruction
        def parabolic(t, a, b, c):
            return a * t**2 + b * t + c
        
        M = np.array([[75**2, 75],[196**2, 196]])
        rhs = np.array([[77.5-82.5],[65-82.5]]) 

        ig_ab = np.linalg.solve(M, rhs).squeeze()
        print(ig_ab)
        initial_guess = [ig_ab[0].astype(np.float64), ig_ab[1].astype(np.float64), 82.5]
        print(training_var_numpy[num_train:,:,0].shape)
        print(t_num.shape)

        params, params_covariance = curve_fit(parabolic, t_num.squeeze(), training_var_numpy[num_train:,:,1].squeeze(), p0=initial_guess)
        a_fit, b_fit, c_fit = params
        print(a_fit)
        print(b_fit)
        print(c_fit)
        umid_var_numpy = np.zeros((num_test_time_series,training_var_numpy.shape[1]))
        for i in range(num_test_time_series):
            umid_var_numpy[i, :] = parabolic(t_num, a_fit, b_fit, c_fit -10 + 2 * i)

        testg_var_numpy_umid = np.stack((np.repeat(training_var_numpy[num_train:, :, 0], num_test_time_series, axis = 0), umid_var_numpy), axis = 2) #T_cos(T_mean, f, A, t)
        dataset_testg_umid = {
                'times' : t_num.T, # [num_times]
                'inp_parameters' : None, # [num_samples x num_par]
                'inp_signals' : testg_var_numpy_umid, # [num_samples x num_times x num_signals]
                'target_cases' : np.repeat(cases[num_train:], num_test_time_series, axis=0),
                'num_times' : T_fin+1,
                'beta_guess' : beta_guess, 
                'time_vec' : t.T,
                'weeks' : weeks,
                'frac' : int(dt/dt_num)
        }

        width_pixels  = 200#600#337
        height_pixels = 200#500#266

        # Desired DPI
        dpi = 100

        # Calculate figure size in inches
        width_in_inches  = width_pixels / dpi
        height_in_inches = height_pixels / dpi

        state_predict_umid = LDNet(dataset_testg_umid, initial_lat_state_testg_tests, lat_param_testg_tests, IC_testg_tests).numpy()
        beta_predict_umid  = evolve_dynamics(dataset_testg_umid, initial_lat_state_testg_tests, lat_param_testg_tests).numpy()
        fig,ax = plt.subplots(1,3, figsize=(5*width_in_inches, 1.5*height_in_inches), dpi=dpi)
        for k in range(num_test_time_series):    
            ax[0].plot(np.arange(1,state_predict_umid.shape[1]+1), state_predict_umid[k,:,2], '-',linewidth = 2, label = 'Real')
            ax[1].plot(np.arange(1,beta_predict_umid.shape[1]+1), beta_predict_umid[k,:], '-',linewidth = 2, label = 'Real')
            ax[2].plot(np.arange(1,testg_var_numpy_umid.shape[1]+1), testg_var_numpy_umid[k,:,1], '-',linewidth = 2, label = 'Real')
        
        #ax[0].set_title('Infected') 
        ax[0].set_ylabel(r'Total infected')
        ax[0].set_xlabel(r'days')
        #ax[1].set_title('Transmission Rate') 
        ax[1].set_ylabel(r'Transmission rate')
        ax[1].set_xlabel('days')
        #ax[2].set_title('Temperature')
        ax[2].set_ylabel(r'Relative Umidity [\%]')
        ax[2].set_xlabel('days')
        fig.tight_layout()
        plt.savefig(folder + 'prova_test_umid_amplitude.pdf')
        plt.show()
        
        np.savetxt(folder_testg + 'umids.txt', testg_var_numpy_umid[:,:,1])
        np.savetxt(folder_testg + 'S_rec_testg_umid.txt', state_predict_umid[:,:,0])
        np.savetxt(folder_testg + 'E_rec_testg_umid.txt', state_predict_umid[:,:,1])
        np.savetxt(folder_testg + 'I_rec_testg_umid.txt', state_predict_umid[:,:,2])
        np.savetxt(folder_testg + 'beta_rec_umid_testg.txt', beta_predict_umid.squeeze())
    #initial_lat_state_testg = tf.Variable(i_n.reshape((dataset_testg['num_samples'],1)), trainable=True)
    #lat_param_testg = tf.Variable(l_n.reshape((dataset_testg['num_samples'],1)), trainable=True)


if os.path.exists(folder_train) == False:
    os.mkdir(folder_train)

if os.path.exists(folder_testg) == False:
    os.mkdir(folder_testg)

if save_train:
    fig_loss, axs_loss = plt.subplots(1, 1)
    axs_loss.loglog(opt_train.iterations_history, opt_train.loss_train_history, 'o-', label = 'training loss')
    axs_loss.axvline(num_epochs_Adam_train)
    axs_loss.set_xlabel('epochs'), plt.ylabel('loss')
    axs_loss.legend()
    plt.savefig(folder_train + 'train_loss.png')
    plt.close()
    
    beta_r = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train).numpy()
    delta_train_r = lat_param_train.numpy().squeeze()
    state_r = LDNet(dataset_train, initial_lat_state_train, lat_param_train, IC_train).numpy()
    #cases_r = np.array([np.sum(alpha * state_r[:,length_period * k: (k+1) * length_period, 1], axis = 1) for k in range(dataset_train['weeks'])]).T
    #E = state_r[:,:-1:dataset_train['frac'],1]
    #reshaped_E = np.reshape(E, (E.shape[0], E.shape[1] // length_period, length_period))
    #new_cases_summed_weekly = np.sum(reshaped_E, axis = 2)
    E = state_r[:,::dataset_train['frac'] * length_period ,1]#frac * 7
    S = state_r[:,::dataset_train['frac'] * length_period ,0]#frac * 7
    #new_cases_summed_weekly = tf.transpose(tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])]))
    #MSE = tf.reduce_mean(tf.square(new_cases_summed_weekly - dataset['target_cases']) / tf.square( tf.square( dataset['target_cases']) + 1e-6 )) 
    #MSE = tf.reduce_mean(tf.square((S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - dataset['target_cases']) / tf.square(dataset['target_cases'] + 1e-6 )) 
    cases_r = S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]#alpha * state_r[:,:-1,1] 
    I_interp_r = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_r[k, :,2]) for k in range(state_r.shape[2])])
    t_num = t_num.squeeze()
    print(state_r[0,:,1]) 
    print(state_r[1,:,1]) 
    for k in range(int(dataset_train['num_samples'])):
        fig,ax = plt.subplots(1,4, figsize = (18,7))
        ax[0].plot(np.arange(1,len(cases[k,:])+1), cases[k,:], 'o--',linewidth = 2, label = 'Real')
        ax[0].plot(np.arange(1, len(cases_r[k,:])+1), cases_r[k,:],'o-',linewidth = 2,  label = 'NN')
        ax[0].set_xlabel('weeks')
        ax[0].set_ylabel('Weekly incidence')
        ax[0].set_title('New Cases') 
        ax[1].plot(t_num, beta_r[k,:],linewidth = 2)
        ax[1].set_xlabel('days')
        ax[1].set_title('Transmission Rate') 
        ax[2].plot(t_num, beta_r[k,:]/alpha,linewidth = 2)
        ax[2].set_title('Reproduction number') 
        ax[2].set_xlabel('days')
        ax[3].plot(t_num, training_var_numpy[k,:],linewidth = 2)
        ax[3].set_title('Temperature')
        ax[3].set_xlabel('days')
        fig.tight_layout()
        plt.savefig(folder + 'out_train_' + str(k) + '.pdf')
        plt.close()

    np.savetxt(folder_train + 'beta_rec_train.txt', beta_r.squeeze())
    np.savetxt(folder_train + 'cases_train.txt', cases.squeeze())
    np.savetxt(folder_train + 'delta_train.txt', delta_train_r.squeeze())
    np.savetxt(folder_train + 'S_rec_train.txt', state_r[:,:,0])
    np.savetxt(folder_train + 'E_rec_train.txt', state_r[:,:,1])
    np.savetxt(folder_train + 'I_rec_train.txt', state_r[:,:,2])
    np.savetxt(folder_train + 't_num.txt', t_num.squeeze())
    
    beta_train = beta_r.copy()

if save_testg:
   #####fig_delta, axs_delta = plt.subplots(1,1)
    delta_train_r = lat_param_train.numpy().squeeze()
    delta_testg_r = lat_param_testg.numpy().squeeze()
   #####axs_delta.scatter(delta_train_r, delta_testg_r)#, 'o')
   #####axs_delta.set_xlabel('latent param train')
   #####axs_delta.set_ylabel('latent param testg')
   #####plt.savefig(folder_train + 'train_testg.png')
   #####plt.show()
    
    beta_r = evolve_dynamics(dataset_testg_full, initial_lat_state_testg, lat_param_testg).numpy()
    state_r = state_predict.numpy()
    E = state_r[:,::dataset_train['frac'] * length_period ,1]#frac * 7
    S = state_r[:,::dataset_train['frac'] * length_period ,0]#frac * 7
    #new_cases_summed_weekly = tf.transpose(tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])]))
    #MSE = tf.reduce_mean(tf.square(new_cases_summed_weekly - dataset['target_cases']) / tf.square( tf.square( dataset['target_cases']) + 1e-6 )) 
    #MSE = tf.reduce_mean(tf.square((S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - dataset['target_cases']) / tf.square(dataset['target_cases'] + 1e-6 )) 
    cases_r = S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]#alpha * state_r[:,:-1,1] 
    highlight_color = 'lightgreen'
    for k in range(int(dataset_testg['num_samples'])):
        fig,ax = plt.subplots(1,4, figsize = (18,7))
        #print('delta_train: ', delta_train_r[k])
        #print('delta_testg: ', delta_testg_r[k])
        ax[0].plot(np.arange(1,len(cases[num_train+k,:])+1), cases[num_train+k,:], 'o--',linewidth = 2, label = 'Real')
        ax[0].plot(np.arange(1, len(cases_r[k,:])+1), cases_r[k,:],'o-',linewidth = 2,  label = 'NN')
        ax[0].set_xlabel('weeks')
        ax[0].set_ylabel('Weekly incidence')
        ax[0].set_title('New Cases') 
        ax[0].axvspan(0, W_obs, facecolor=highlight_color, alpha = 0.5)
        ax[1].plot(t_num, beta_r[k,:],linewidth = 2)
        #ax[1].plot(t_num, beta_train[num_train+k,:],'--',linewidth = 2)
        ax[1].set_xlabel('days')
        ax[1].set_title('Transmission Rate') 
        ax[1].axvspan(0, T_obs, facecolor=highlight_color, alpha = 0.5)
        ax[2].plot(t_num, beta_r[k,:]/alpha,linewidth = 2)
        #ax[2].plot(t_num, beta_train[num_train+k,:]/alpha,'--',linewidth = 2)
        ax[2].set_title('Reproduction number') 
        ax[2].set_xlabel('days')
        ax[2].axvspan(0, T_obs, facecolor=highlight_color, alpha = 0.5)
        ax[3].plot(t_num, training_var_numpy[num_train+k,:],linewidth = 2)
        ax[3].set_title('Temperature')
        ax[3].set_xlabel('days')
        ax[3].axvspan(0, T_obs, facecolor=highlight_color, alpha = 0.5)
        fig.tight_layout()
        plt.savefig(folder + 'out_testg_' + str(k) + '.pdf')
        plt.close()
    delta_testg_r = lat_param_testg.numpy().squeeze()
    #I_interp_r = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_r[k, :,2]) for k in range(state_r.shape[2])])
    
    np.savetxt(folder_testg + 'beta_rec_testg.txt', beta_r.squeeze())
    np.savetxt(folder_testg + 'delta_testg.txt', [[delta_testg_r]])
    np.savetxt(folder_testg + 'S_rec_testg.txt', state_r[:,:,0])
    np.savetxt(folder_testg + 'E_rec_testg.txt', state_r[:,:,1])
    np.savetxt(folder_testg + 'I_rec_testg.txt', state_r[:,:,2])
    np.savetxt(folder_testg + 't_num.txt', t_num.squeeze())
    print(delta_testg_r)
    print(delta_train_r)
os.system('cp /home/giovanni/Desktop/LDNets/src/TestCase_epi_external_dataset_umidity.py ' + folder)
