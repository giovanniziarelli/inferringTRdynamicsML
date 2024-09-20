#!/usr/bin/env python3

#%% Import modules
import numpy as np
import matplotlib
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
import time
import SirHelperFunctions as shf
from scipy.linalg import lstsq

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
b_ref = 0.061#0.77#0.5#0.025#0.7#0.77#0.1672#0.92851
tau = 0
layers = 2
err = 0#1e-1#1e-1#1e-1
err_name = '0'
#%% Model parameters
alpha = 1 / 1.5#5#1.5
gamma = 1 / 1.2#10#1.2

plot_training_set = 1
save_train = 1
save_testg = 1
various_IC = 0
new_cases_loss = 1
estim_IC = 0

# for sh
trial = 2
trial_sim = 2
trial_neu = 1 
trial_err = 1
trial_lay = 1

err_vec = [0, 1e-3, 1e-2, 1e-1]#1e-1#1e-1#1e-1
err_vec_name = ['0', '1e-3', '1e-2', '1e-1']
err_name = err_vec_name[trial_err-1]
err = err_vec[trial_err-1]

size = ['20', '50', '100', '500', '1000', '75', '150', '300', '200', '400', '750']
neurons_vec = [4,8,10,12]#4 

neurons = neurons_vec[trial_neu - 1]#4 

T_fin = 364#70
N_weeks = 52 
dt_base = 1#T_fin 

num_latent_states = 1
num_latent_params = 1

##### Loading datasets

problem = {
    "input_parameters": [],
    "input_signals": [
        { "name": "T" }
    ],
    "output_fields": [
        { "name": "I" }
    ]
}
#%% Define problem
normalization = {
    'time': {
        'time_constant' : dt_base
    },
    'input_signals': {
        'T': {'min': -10, 'max' : 30}#{ 'min': 0, 'max': 4 }
    },
    'output_fields': {
        'I': { 'min': 0, "max": 1 },
    }
}
#prova dataset pkl

dataset_path = 'datasets/temperature_'+size[trial-1]+'.pkl'
test_path    = 'datasets/temperature_50.pkl'

with open(dataset_path, 'rb') as file:
    dataset_extern, _, _, beta0_train, _, _, T_mean_train, amp_T_train, phase_T_train, f_T_train = pickle.load(file)

with open(test_path, 'rb') as file:
    _, val_extern, test_extern, _, beta0_val, beta0_test, T_mean_test, amp_T_test, phase_T_test, f_T_test = pickle.load(file)

#betas, deltas = shf.compute_beta_delta([dataset_extern, val_extern, test_extern], \
#        [beta0_train, beta0_val, beta0_test],\
#        t_max, tau, b_ref, [T_mean_train, T_mean_test, T_mean_test])
#beta_train, beta_val, beta_test = betas
#delta_train, delta_val, delta_test = deltas
betas, deltas = shf.compute_beta_delta([test_extern, val_extern, dataset_extern], \
        [beta0_test, beta0_val, beta0_train],\
        t_max, tau, b_ref, [T_mean_test, T_mean_test, T_mean_train])
beta_test, beta_val, beta_train = betas
delta_test, delta_val, delta_train = deltas
beta_train.reshape((beta_train.shape[0], beta_train.shape[1],1))

#%% Dataset parameters
n_size = beta_train.shape[0]#6
n_size_testg = beta_test.shape[0]#6
#T_mean = 16
dt = 1
length_period = 7
weeks = int(T_fin / length_period)
tau = 100 #useful for linear ode
t = np.arange(0, T_fin+dt, dt)[None,:]
nu = 1e0#1e0#1e2

dt_num = 0.5#0.1#0.05#0.2
t_num = np.arange(0, T_fin+dt_num, dt_num)[None, :]

training_var_numpy = dataset_extern #T_cos(T_mean, f, A, t)
training_var_numpy = np.array([np.interp(t_num.squeeze(), t.squeeze(), dataset_extern[k,:]) for k in range(n_size)])

testing_var_numpy = test_extern #T_cos(T_mean, f, A, t)
testing_var_numpy = np.array([np.interp(t_num.squeeze(), t.squeeze(), test_extern[k,:]) for k in range(n_size_testg)])

alpha = 1 / 10#1.5#5#1.5
gamma = 1 / 20#1.2#10#1.2

def epiModel_rhs(state, beta): # state dim (samples, 3), beta dim (samples,1)

    dSdt = - beta * tf.expand_dims(state[:,0],axis=1) * tf.expand_dims(state[:,2],axis=1)
    dEdt = beta * tf.expand_dims(state[:,0],axis=1) * tf.expand_dims(state[:,2],axis=1) - alpha * tf.expand_dims(state[:,1],axis=1)
    dIdt = alpha * tf.expand_dims(state[:,1],axis=1) - gamma* tf.expand_dims(state[:,2],axis=1)

    return tf.concat([dSdt, dEdt, dIdt], axis = 1)

def beta_evolution(beta, T, delta_, T_m):
    beta_rhs =4/365*(T/T_m - 1) * ((beta)**2 / b_ref / delta_ - (beta))#1/tau * (beta - 2 + (T - 278) * 0.02)
    return beta_rhs

beta_guess = b_ref
beta_guess_IC = b_ref * np.ones((n_size,1))

S0 = 0.97
E0 = 0.02
I0 = 0.01
IC = np.repeat(np.array([S0, E0, I0])[:,None],n_size, axis = 1).T# num_sample x num_variabili
IC_testg = np.repeat(np.array([S0, E0, I0])[:,None],n_size_testg, axis = 1).T# num_sample x num_variabili
def IC_fun(num_samples):
    IC = np.zeros((num_samples,3))
    IC[:,0] = np.random.uniform(0.9,1.0,num_samples)
    IC[:,1] = 1 - IC[:,0] #np.minimum(np.random.uniform(0,0.1,num_samples), 1 - IC[:,0])
    IC[:,2] = 0#1 - IC[:,0] - IC[:,2]
    return IC
if various_IC > 0:
    IC = IC_fun(n_size)
state_mod = np.zeros((n_size,t_num.shape[1], 3))
state_mod[:,0,:] = IC

state_mod_testg = np.zeros((n_size_testg,t_num.shape[1], 3))
state_mod_testg[:,0,:] = IC_testg

#initialize real beta
beta_real = np.zeros((n_size, t_num.shape[1], 1))
beta_real[:,0,:] = beta0_train[:,None]

beta_real_testg = np.zeros((n_size_testg, t_num.shape[1], 1))
beta_real_testg[:,0,:] = beta0_test[:,None]

plt.plot(training_var_numpy.T)
plt.close()

#solving the ODEs for retrieving beta and state
for i in range(t_num.shape[1]-1):
    beta_real[:, i+1, :] =beta_real[:, i, :] + dt_num * beta_evolution(beta_real[:,i,:], training_var_numpy[:,i][:,None], delta_train[:,None], T_mean_train[:,None])
    state_mod[:,i+1,:] = state_mod[:,i,:] + dt_num * epiModel_rhs(state_mod[:,i,:], beta_real[:,i,:])
    beta_real_testg[:, i+1, :] = beta_real_testg[:, i, :] + dt_num * beta_evolution(beta_real_testg[:,i,:], testing_var_numpy[:,i][:,None], delta_test[:,None], T_mean_test[:,None])
    state_mod_testg[:,i+1,:] = state_mod_testg[:,i,:] + dt_num * epiModel_rhs(state_mod_testg[:,i,:], beta_real_testg[:,i,:])
print('beta qui')
# state_mod[:,:,2] = np.maximum(state_mod[:,:,2] * (1 + err * np.random.randn(state_mod.shape[0], state_mod.shape[1])), 0.0)
# state_mod[:,:,0] = 1 - state_mod[:,:,2]
mask = np.isnan(state_mod[:,:,1])
mask_testg = np.isnan(state_mod_testg[:,:,1])

indexes = []
indexes_testg = []

for ii in range(mask.shape[0]):
    if mask[ii].any() == True:
        indexes.append(ii)
        continue

for ii in range(mask_testg.shape[0]):
    if mask_testg[ii].any() == True:
        indexes_testg.append(ii)
        continue

training_var_numpy = np.delete(training_var_numpy, indexes, 0)
beta_real          = np.delete(beta_real, indexes, 0)
state_mod          = np.delete(state_mod, indexes, 0)
delta_train        = np.delete(delta_train, indexes, 0)
IC                 = np.delete(IC, indexes, 0)

testing_var_numpy = np.delete(testing_var_numpy, indexes_testg, 0)
beta_real_testg   = np.delete(beta_real_testg, indexes_testg, 0)
state_mod_testg   = np.delete(state_mod_testg, indexes_testg, 0)
delta_test        = np.delete(delta_test, indexes_testg, 0)
IC_testg          = np.delete(IC_testg, indexes_testg, 0)

n_size = n_size - len(indexes)
n_size_testg = n_size_testg - len(indexes_testg)

E_interp = state_mod[:,::int(dt/dt_num), 1]# np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,1]) for k in range(n_size)])
S_interp = state_mod[:,::int(dt/dt_num), 0]# np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,0]) for k in range(n_size)])
print('ESHAPE', E_interp.shape)
E = E_interp[:,::length_period]
S = S_interp[:,::length_period]
print(E.shape)
plt.plot(S.T)
plt.close()
#cases = np.array([np.sum(alpha * E_interp[:,1 + length_period*k:length_period*(k+1)], axis = 1) for k in range(weeks)]).T#S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]
cases = np.maximum((S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) * (1 + err * np.random.randn(S.shape[0], S.shape[1]-1)), 0.0)

E_interp_testg = state_mod_testg[:,::int(dt/dt_num), 1]# np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,1]) for k in range(n_size)])
S_interp_testg = state_mod_testg[:,::int(dt/dt_num), 0]# np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,0]) for k in range(n_size)])
E_testg = E_interp_testg[:,::length_period]
S_testg = S_interp_testg[:,::length_period]
#cases = np.array([np.sum(alpha * E_interp[:,1 + length_period*k:length_period*(k+1)], axis = 1) for k in range(weeks)]).T#S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]
cases_testg = np.maximum((S_testg[:,:-1] + E_testg[:,:-1] - S_testg[:,1:] - E_testg[:,1:]) * (1 + err * np.random.randn(S_testg.shape[0], S_testg.shape[1]-1)), 0.0)

mask = (cases >= 0.15)
mask_testg = (cases_testg >= 0.15)

indexes = []
indexes_testg = []
for ii in range(mask.shape[0]):
    if mask[ii].any() == True:
        indexes.append(ii)
        continue
for ii in range(mask_testg.shape[0]):
    if mask_testg[ii].any() == True:
        indexes_testg.append(ii)
        continue


training_var_numpy = np.delete(training_var_numpy, indexes, 0)
cases              = np.delete(cases, indexes, 0)
beta_real          = np.delete(beta_real, indexes, 0)
state_mod          = np.delete(state_mod, indexes, 0)
delta_train        = np.delete(delta_train, indexes, 0)
IC                 = np.delete(IC, indexes, 0)

testing_var_numpy = np.delete(testing_var_numpy, indexes_testg, 0)
cases_testg       = np.delete(cases_testg, indexes_testg, 0)
beta_real_testg   = np.delete(beta_real_testg, indexes_testg, 0)
state_mod_testg   = np.delete(state_mod_testg, indexes_testg, 0)
delta_test        = np.delete(delta_test, indexes_testg, 0)
IC_testg          = np.delete(IC_testg, indexes_testg, 0)

n_size = n_size - len(indexes)
n_size_testg = n_size_testg - len(indexes_testg)

print(state_mod.shape)
Delta_loss = 0
print(np.arange(n_size))
shuffle_indexes = np.arange(training_var_numpy.shape[0])
np.random.shuffle(shuffle_indexes)
#print(np.random.shuffle(ind))# shuffle_indexes)
training_var_numpy = training_var_numpy[shuffle_indexes]
print(training_var_numpy.shape)
beta_real = beta_real[shuffle_indexes]
state_mod = state_mod[shuffle_indexes]
cases = cases[shuffle_indexes]
IC = IC[shuffle_indexes]
delta_train = delta_train[shuffle_indexes]
if Delta_loss:
    I_interp = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,2]) for k in range(n_size)])
    Delta_I = (I_interp[:, 1:] - I_interp[:, :-1])
    print('Delta_I sahpe')
    print(Delta_I.shape)
    plt.plot(Delta_I)
    plt.close()
    inf_weekly = np.array([np.sum(Delta_I[:,length_period*k:length_period*(k+1)], axis = 1) for k in range(weeks)]).T
    print(inf_weekly.shape)
else:
    I_interp = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,2]) for k in range(n_size)])
    inf_weekly = np.array([np.sum(I_interp[:,length_period*k:length_period*(k+1)], axis = 1) for k in range(weeks)]).T

E_interp = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,1]) for k in range(n_size)])
S_interp = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,0]) for k in range(n_size)])
E = E_interp[:,1::length_period]
S = S_interp[:,1::length_period]
#MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']))# / tf.square( (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:])+ 1e-6 )) 
print('qui')
print(S.shape)
print(S[:,0])
print(S[:,1])
print(E[:,0])
print(E[:,1])
#cases = np.array([np.sum(alpha * E_interp[:,1 + length_period*k:length_period*(k+1)], axis = 1) for k in range(weeks)]).T

#beta_guess = b_ref 
#beta_guess_IC = b_ref * np.ones((n_size,1))

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
    smoothed_data = savgol_filter(data, 31, 1)
    smoothed_data = smoothed_data #tf.math.log(smoothed_data + 1)
    return smoothed_data
def weekly_avg(T):
    print(T.shape)
    T_reshaped = np.reshape(T[:,:-1], (T.shape[0], T.shape[1]//7, 7))
    return np.mean(T_reshaped, axis = 2)

#cases = np.delete(cases, indexes, 0)
#n_size = n_size - len(indexes)

plot_temps = 1
if plot_temps:
    fig,ax = plt.subplots(1,4)
    for k in range(training_var_numpy.shape[0]):
        ax[0].plot(training_var_numpy[k,:])

        ax[1].plot(cases[k,:])
        
        ax[2].plot(beta_real[k,:])
        ax[3].plot(I_interp[k,:])
    
    ax[0].set_title('Temps')
    ax[1].set_title('Cases')
    ax[2].set_title('beta')
    ax[1].set_title('Inf')
    plt.close()
# Defining Datasets
print(training_var_numpy.shape)
print(cases.shape)
dataset_train = {
        'times' : t_num.T, # [num_times]
        'inp_parameters' : None, # [num_samples x num_par]
        'inp_signals' : training_var_numpy[:,:,None], # [num_samples x num_times x num_signals]
        'beta_state' : beta_real, #[n_samples x num_times x num_latent_staes]
        'inf_variables' : state_mod,
        'initial_state' : IC,
        'target_incidence' : inf_weekly,
        'target_cases' : cases,
        'num_times' : T_fin+1,
        'beta_guess' : beta_guess,
        'time_vec' : t.T,
        'weeks' : weeks,
        'frac' : int(dt/dt_num)
}

dataset_testg_full = {
        'times' : t_num.T, # [num_times]
        'inp_parameters' : None, # [num_samples x num_par]
        'inp_signals' : testing_var_numpy[:,:,None], # [num_samples x num_times x num_signals]
        'beta_state' : beta_real_testg, #[n_samples x num_times x num_latent_staes]
        'inf_variables' : state_mod_testg,
        'initial_state' : IC_testg,
        'target_incidence' : inf_weekly,
        'target_cases' : cases_testg,
        'num_times' : T_fin+1,
        'beta_guess' : beta_guess,
        'time_vec' : t.T,
        'weeks' : weeks,
        'frac' : int(dt/dt_num)
}
T_obs = 78
W_obs = int(T_obs / 7)
dataset_testg = utils.cut_dataset_epi(dataset_testg_full, T_obs)
# For reproducibility (delete if you want to test other random initializations)
np.random.seed(0)
tf.random.set_seed(0)
# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset_epi(dataset_train, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi(dataset_testg, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi(dataset_testg_full, problem, normalization, dt = None, num_points_subsample = None)


#%% Define model
constraint_IC = ClipConstraint(0, 1)
# dynamics network
# Initial conditions
InitialValue_train = np.zeros((dataset_train['num_samples'], 1))
InitialValue_testg = np.zeros((dataset_testg['num_samples'], 1))
InitialValue_train[:,0] = IC[:,2]# * np.ones_like(cases[:,0])#cases[:,0]#10/50000000#0.01
InitialValue_testg[:,0] = IC_testg[:,2]# * np.ones_like(cases[:,0])#cases[:,0]#10 / 50000000
IC_train = tf.Variable(InitialValue_train, trainable=False, constraint = constraint_IC)
IC_testg = tf.Variable(InitialValue_testg, trainable=False, constraint = constraint_IC)

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

if trial_lay == 1:
    NNdyn = tf.keras.Sequential([
                tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, input_shape = input_shape, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
                tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
                #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
                #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
                #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
                #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
                #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
                #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
                tf.keras.layers.Dense(num_latent_states, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
            ])
elif trial_lay == 2:
    NNdyn = tf.keras.Sequential([
                tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, input_shape = input_shape, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
                tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
                tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
                tf.keras.layers.Dense(num_latent_states, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
            ])
else: 
    NNdyn = tf.keras.Sequential([
                tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, input_shape = input_shape, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
                tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
                tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
                tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)),
                tf.keras.layers.Dense(num_latent_states, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
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
        inputs = [lat_state, dataset['inp_signals'][:,i,:]]
        #tf.print(inputs)
        if num_latent_params > 0:
            inputs.append(lat_param)
        lat_state = tf.math.maximum(lat_state + dt_num/dt_ref * NNdyn(tf.concat(inputs, axis = -1)), tf.zeros_like(lat_state))
        #lat_state = lat_state + dt_num/dt_ref * NNdyn(tf.concat(inputs, axis = -1))
        lat_state_history = lat_state_history.write(i + 1, lat_state)

    return tf.transpose(lat_state_history.stack(), perm=(1,0,2))

# initialize epi model parameters

def evolve_model(dataset, lat_state, IC_d):
    IC_E = 0.02 * tf.ones_like(IC_d)#reshape((IC ), [IC_d.shape[0],1])
    IC_S = tf.reshape(1 - IC_d - IC_E, [IC_d.shape[0], 1])
    IC_all = tf.concat([IC_S, IC_E, IC_d], 1)
    #tf.print(IC_all)
    state = IC_all
    state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    state_history = state_history.write(0, state)

    # time integration
    for i in tf.range(dataset['num_times'] - 1):
        state = state + dt_num * epiModel_rhs(state, lat_state[:, i, :])
        state_history = state_history.write(i + 1, state)

    return tf.transpose(state_history.stack(), perm=(1,0,2))

def LDNet(dataset, initial_lat_state, lat_param, IC_d):
    lat_states = evolve_dynamics(dataset,initial_lat_state, lat_param)
    return evolve_model(dataset, lat_states, IC_d)

#%% Loss function

def loss_exp(dataset, initial_lat_state, lat_param, IC_d):
    state = LDNet(dataset, initial_lat_state, lat_param, IC_d)
    #E = tf.transpose(state[:,::dataset['frac'] * 7,1])
    E = state[:,::dataset['frac'] * length_period ,1]#frac * 7
    S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    #new_cases_summed_weekly = tf.transpose(tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])]))
    #MSE = tf.reduce_mean(tf.square(new_cases_summed_weekly - dataset['target_cases']) / tf.square( tf.square( dataset['target_cases']) + 1e-6 )) 
    
    MSE = tf.reduce_mean(tf.square((S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - dataset['target_cases']))# / tf.square( dataset['target_cases']))#(S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:])+ 1e-6 )) 
    
    #MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']) / tf.square( 100 * dataset['target_cases'] )) 
    return MSE


def loss_exp_beta(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    #E = tf.transpose(state[:,::dataset['frac'] * 7,1])
    E = state[:,::dataset['frac'] * length_period ,1]#frac * 7
    S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    #new_cases_summed_weekly = tf.transpose(tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])]))
    #MSE = tf.reduce_mean(tf.square(new_cases_summed_weekly - dataset['target_cases']) / tf.square( tf.square( dataset['target_cases']) + 1e-6 )) 
    
    MSE = tf.reduce_mean(tf.square(100*(S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100* dataset['target_cases']) / tf.square(100* dataset['target_cases']))#(S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:])+ 1e-6 )) 
    
    #MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']) / tf.square( 100 * dataset['target_cases'] )) 
    return MSE

def loss_exp_beta_abs(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    #E = tf.transpose(state[:,::dataset['frac'] * 7,1])
    E = state[:,::dataset['frac'] * length_period ,1]#frac * 7
    S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    #new_cases_summed_weekly = tf.transpose(tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])]))
    #MSE = tf.reduce_mean(tf.square(new_cases_summed_weekly - dataset['target_cases']) / tf.square( tf.square( dataset['target_cases']) + 1e-6 )) 
    
    MSE = tf.reduce_mean(tf.square(100*(S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100* dataset['target_cases']))#(S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:])+ 1e-6 )) 
    
    #MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']) / tf.square( 100 * dataset['target_cases'] )) 
    return MSE

def loss_exp_beta_(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    #E = tf.transpose(state[:,::dataset['frac'] * 7,1])
    E = tf.transpose(state[:,::dataset['frac'] * length_period ,1])#frac * 7
    #S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    #new_cases_summed_weekly = tf.transpose(tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])]))
    #MSE = tf.reduce_mean(tf.square(new_cases_summed_weekly - dataset['target_cases']) / tf.square( tf.square( dataset['target_cases']) + 1e-6 )) 
    
    new_cases_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])
    #print(new_cases_summed_weekly.shape)
    return tf.reduce_mean(tf.square((new_cases_summed_weekly) - (tf.transpose(dataset['target_cases']))) / (tf.square(new_cases_summed_weekly + 1e-6)))

def val_exp(dataset, initial_lat_state, lat_param, IC_d):
    state = LDNet(dataset, initial_lat_state, lat_param, IC_d)
    E = tf.transpose(state[:,::dataset['frac'],1])
    new_cases_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])
    return tf.reduce_mean(tf.square((new_cases_summed_weekly) - (tf.transpose(dataset['target_cases']))))

def loss_H1(dataset, initial_lat_state, lat_param, IC_d):
    state = LDNet(dataset, initial_lat_state, lat_param, IC_d)
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
nu = 3e-2#1e-1#1e-4#5e-2#5e-2#5.1e0#1e-1#i1e0 good 1e0#1e-7#1e-2#-4#1e0#1e2
nu_h1 = 0#1e-3 
nu_r1  = 5e-4#1e-6#1e-2#5e-3#1e-5
nu_p = 1e-4#1e1#1e4#1e0#1e-1#6
nu_p2 = 1e-4#1e1#1e4#5e0#1e-1#1e-3#1e-3#6
alpha_reg = 1e-1#1e-1#1e-3#1e-6#1e-1#1e-2#1e#1e-5#1e-2#1e-2#1e-1# 1e-8#1e-8#1e1#1e-2#1e-3#5e-5#5e-5#5e-6#4.7e-3
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
    l = nu *loss_exp_beta(dataset_train, beta, IC_train)\
            + alpha_reg * weights_reg(NNdyn)\
            + nu_r1 * tf.reduce_mean(tf.square(lat_param_train - 1))\
            + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - beta[:,1])/dt_num))\
            + nu_p * tf.reduce_mean(tf.square((beta[:,1:] - beta[:,:-1])/dt_num))
    return l 

def loss_train_abs():
    beta = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
    l = nu *loss_exp_beta_abs(dataset_train, beta, IC_train)\
            + alpha_reg * weights_reg(NNdyn)\
            + nu_r1 * tf.reduce_mean(tf.square(lat_param_train -1))\
            + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - beta[:,1])/dt_num))\
            + nu_p * tf.reduce_mean(tf.square((beta[:,1:] - beta[:,:-1])/dt_num))
    return l 

def val_train():
    beta = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
    l = loss_exp_beta_abs(dataset_train, beta, IC_train)
    return l 

####loss_train = lambda: nu * loss(dataset_train, initial_lat_state_train, lat_param_train, IC_train)\
####        + alpha_reg * weights_reg(NNdyn)\
####        + nu_r1 * tf.reduce_mean(tf.square(lat_param_train - 1))\
####        + nu_p * tf.reduce_mean(tf.reduce_sum(tf.square(evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train) ), axis = 0)) + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)[:,1,:]) / dt_num))

#val_train = lambda: val_exp(dataset_train, initial_lat_state_train, lat_param_train, IC_train)         
         
val_metric = val_train

print(training_var_numpy.shape)
print(cases.shape)

opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train, val_metric)

num_epochs_Adam_train = 200# good
num_epochs_BFGS_train = 200

init_adam_time = time.time()
print('training (Adam)...')
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=5e-4))#1e-3))#1e-2 good
end_adam_time = time.time()

print('training (BFGS)...')
init_bfgs_time = time.time()
opt_train.optimize_BFGS(num_epochs_BFGS_train)
end_bfgs_time = time.time()

train_times = [end_adam_time - init_adam_time, end_bfgs_time - init_bfgs_time]

opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train_abs, val_metric)

num_epochs_Adam_train = 200# good
num_epochs_BFGS_train = 500

init_adam_time = time.time()
print('training (Adam)...')
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=1e-5))#1e-3))#1e-2 good
end_adam_time = time.time()

print('training (BFGS)...')
init_bfgs_time = time.time()
opt_train.optimize_BFGS(num_epochs_BFGS_train)
end_bfgs_time = time.time()

train_times.append(end_adam_time - init_adam_time)
train_times.append(end_bfgs_time - init_bfgs_time)



#%% Estimation

if num_latent_params>0:
    num_sample_test = dataset_testg['num_samples']
    trainable_variables_testg = [initial_lat_state_testg] 
    trainable_variables_testg += [lat_param_testg]
    trainable_variables_testg += [IC_testg]
    
    for T_obs in [15, 36, 57, 78, 99, 120, 141]:
        dataset_testg_full = {
                'times' : t_num.T, # [num_times]
                'inp_parameters' : None, # [num_samples x num_par]
                'inp_signals' : testing_var_numpy[:,:,None], # [num_samples x num_times x num_signals]
                'beta_state' : beta_real_testg, #[n_samples x num_times x num_latent_staes]
                'inf_variables' : state_mod_testg,
                'initial_state' : IC_testg,
                'target_incidence' : inf_weekly,
                'target_cases' : cases_testg,
                'num_times' : T_fin+1,
                'beta_guess' : beta_guess,
                'time_vec' : t.T,
                'weeks' : weeks,
                'frac' : int(dt/dt_num)
        }
        
        W_obs = int(T_obs / 7)

        T_fin = 364#70
        N_weeks = 52 
        dt_base = 1#T_fin 

        # Creating saving folders
        folder = 'neurons_' + str(neurons) + '_' + str(T_fin) + '_synthetic_lay' + str(trial_lay+1) + '_err_'+err_name+'_size_'+size[trial-1] + '_trialsim_' + str(trial_sim) +'_T_obs_' + str(T_obs)+'/'
        #folder = '/home/giovanni/Desktop/LDNets/target_err_3e_2_Testg/'

        if os.path.exists(folder) == False:
            os.mkdir(folder)
        folder_train = folder + 'train/'
        folder_testg = folder + 'testg/'

        dataset_testg = utils.cut_dataset_epi(dataset_testg_full, T_obs)
        # For reproducibility (delete if you want to test other random initializations)
        np.random.seed(0)
        tf.random.set_seed(0)
        # We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
        utils.process_dataset_epi(dataset_train, problem, normalization, dt = None, num_points_subsample = None)
        utils.process_dataset_epi(dataset_testg, problem, normalization, dt = None, num_points_subsample = None)
        dataset_testg = utils.cut_dataset_epi(dataset_testg_full, T_obs)
        
        #%% Constants
        nu_r2 = 1.2e0#3e1#2.4e1#1e-2#5e-3#1e-5
        nu_r3 = 5e-1#3e1#2.4e1#1e-2#5e-3#1e-5
        nu_loss_testg = 5e-2#5e-2#1e-1
        nu_IC = 0#1e-3
        nu_p_testg = 0#1e2
        nu_p2_testg = 0#1e0#1e2

        C_post    = tf.linalg.inv(tfp.stats.covariance(lat_param_train))
        latent_train_mean = tf.reduce_mean(lat_param_train)

        C_post_b0    = tf.linalg.inv(tfp.stats.covariance(initial_lat_state_train))
        initial_lat_state_train_mean = tf.reduce_mean(initial_lat_state_train)

        #C_post_IC = tf.linalg.inv(tfp.stats.covariance(IC_train))
        #IC_train_mean = tf.repeat(tf.reshape(tf.reduce_mean(IC_train, axis = 0), [1,2]), num_sample_test, axis = 1)
        #IC_train_mean = tf.reduce_mean(IC_train)

        def loss_testg():
            beta = evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg)
            l = nu_loss_testg *loss_exp_beta_abs(dataset_testg, beta, IC_testg)\
                    + nu_r2 * tf.reduce_mean(tf.transpose(lat_param_testg - latent_train_mean ) * C_post * (lat_param_testg - latent_train_mean))\
                    + nu_r3 * tf.reduce_mean(tf.transpose(initial_lat_state_testg - initial_lat_state_train_mean ) * C_post_b0 * (initial_lat_state_testg - initial_lat_state_train_mean ))\
                    + nu_p2_testg * tf.reduce_mean(tf.square((initial_lat_state_testg - beta[:,1])/dt_num))\
                    + nu_p_testg * tf.reduce_mean(tf.square((beta[:,1:] - beta[:,:-1])/dt_num))
            return l 
    ####loss_testg = lambda: nu_loss_testg * loss(dataset_testg, initial_lat_state_testg, lat_param_testg, IC_testg)\
    ####        + nu_r2 * tf.reduce_mean(tf.transpose(lat_param_testg - latent_train_mean ) * C_post * (lat_param_testg - latent_train_mean))#\
    ####        #+ nu_IC * tf.reduce_mean(tf.transpose(IC_testg - IC_train_mean) * C_post_IC * (IC_testg - IC_train_mean))

        opt_testg = optimization.OptimizationProblem(trainable_variables_testg, loss_testg, loss_testg)

        num_epochs_Adam_testg = 500
        num_epochs_BFGS_testg = 500
        
        print('training (Adam)...')
        opt_testg.optimize_keras(num_epochs_Adam_testg, tf.keras.optimizers.Adam(learning_rate=1e-3))
        print('training (BFGS)...')
        opt_testg.optimize_BFGS(num_epochs_BFGS_testg)

        #%% Prediction

        state_predict = LDNet(dataset_testg_full, initial_lat_state_testg, lat_param_testg, IC_testg)
        print('Cpost: ', C_post)
        #print('Cpost_IC: ', C_post_IC)

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
        #plt.savefig(folder_train + 'train_loss.png')
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
        I_interp_r = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_r[k, :,2]) for k in range(n_size)])
        t_num = t_num.squeeze()
        print(state_r[0,:,1]) 
        print(state_r[1,:,1])
        #beta_real = np.delete(beta_real, indexes)
    #   for k in range(int(dataset_train['num_samples'])):
    #       fig,ax = plt.subplots(1,4, figsize = (18,7))
    #       ax[0].plot(np.arange(1,len(cases[k,:])+1), cases[k,:], 'o--',linewidth = 2, label = 'Real')
    #       ax[0].plot(np.arange(1, len(cases_r[k,:])+1), cases_r[k,:],'o-',linewidth = 2,  label = 'NN')
    #       ax[0].set_xlabel('weeks')
    #       ax[0].set_ylabel('Weekly incidence')
    #       ax[0].set_title('New Cases')
    #       ax[1].plot(t_num, beta_r[k,:],linewidth = 2)
    #       ax[1].plot(t_num, beta_real[k,:],'--',linewidth = 2)
    #       ax[1].set_xlabel('days')
    #       ax[1].set_title('Transmission Rate')
    #       ax[2].plot(t_num, beta_r[k,:]/alpha,linewidth = 2)
    #       ax[2].set_title('Reproduction number')
    #       ax[2].set_xlabel('days')
    #       ax[3].plot(t_num, training_var_numpy[k,:],linewidth = 2)
    #       ax[3].set_title('Temperature')
    #       ax[3].set_xlabel('days')
    #       fig.tight_layout()
    #       #plt.savefig(folder + 'out_train_' + str(k) + '.pdf')
    #       plt.close()
    #       #plt.close()
        
        beta_train = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
        l_train = loss_exp_beta(dataset_train, beta_train, IC_train)
        beta_testg = evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg)
        l_testg = loss_exp_beta(dataset_testg, beta_testg, IC_testg)

        np.savetxt(folder_train + 'testg_error.txt', np.array(l_testg).reshape((1,1))) 
        np.savetxt(folder_train + 'train_error.txt', np.array(l_train).reshape((1,1))) 
        np.savetxt(folder_train + 'testg_train_error.txt', np.array(l_testg / l_train).reshape((1,1))) 
        np.savetxt(folder_train + 'train_times.txt', train_times) 
        np.savetxt(folder_train + 'beta_rec_train.txt', beta_r.squeeze())
        np.savetxt(folder_train + 'cases_train.txt', cases_r.squeeze())
        np.savetxt(folder_train + 'delta_train.txt', delta_train_r.squeeze())
        np.savetxt(folder_train + 'S_rec_train.txt', state_r[:,:,0])
        np.savetxt(folder_train + 'E_rec_train.txt', state_r[:,:,1])
        np.savetxt(folder_train + 'I_rec_train.txt', state_r[:,:,2])
        np.savetxt(folder_train + 't_num.txt', t_num.squeeze())

    if save_testg:
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
            ax[0].plot(np.arange(1,len(cases_testg[k,:])+1), cases_testg[k,:], 'o--',linewidth = 2, label = 'Real')
            ax[0].plot(np.arange(1, len(cases_r[k,:])+1), cases_r[k,:],'o-',linewidth = 2,  label = 'NN')
            ax[0].set_xlabel('weeks')
            ax[0].set_ylabel('Weekly incidence')
            ax[0].set_title('New Cases')
            ax[0].axvspan(0, W_obs, facecolor=highlight_color, alpha = 0.5)
            ax[1].plot(t_num, beta_r[k,:],linewidth = 2)
            ax[1].plot(t_num, beta_real_testg[k,:],'--',linewidth = 2)
            ax[1].set_xlabel('days')
            ax[1].set_title('Transmission Rate')
            ax[1].axvspan(0, T_obs, facecolor=highlight_color, alpha = 0.5)
            ax[2].plot(t_num, beta_r[k,:]/alpha,linewidth = 2)
            ax[2].set_title('Reproduction number')
            ax[2].set_xlabel('days')
            ax[2].axvspan(0, T_obs, facecolor=highlight_color, alpha = 0.5)
            ax[3].plot(t_num, testing_var_numpy[k,:],linewidth = 2)
            ax[3].set_title('Temperature')
            ax[3].set_xlabel('days')
            ax[3].axvspan(0, T_obs, facecolor=highlight_color, alpha = 0.5)
            fig.tight_layout()
            plt.savefig(folder + 'out_testg_' + str(k) + '.png')
            plt.close()
        delta_testg_r = lat_param_testg.numpy().squeeze()
        #I_interp_r = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_r[k, :,2]) for k in range(n_size)])

        np.savetxt(folder_testg + 'beta_rec_testg.txt', beta_r.squeeze())
        np.savetxt(folder_testg + 'delta_testg.txt', delta_testg_r.squeeze())
        np.savetxt(folder_testg + 'S_rec_testg.txt', state_r[:,:,0])
        np.savetxt(folder_testg + 'E_rec_testg.txt', state_r[:,:,1])
        np.savetxt(folder_testg + 'I_rec_testg.txt', state_r[:,:,2])
        np.savetxt(folder_testg + 't_num.txt', t_num.squeeze())
        np.savetxt(folder_train + 'S_real.txt', state_mod[:,:,0])
        np.savetxt(folder_train + 'E_real.txt', state_mod[:,:,1])
        np.savetxt(folder_train + 'I_real.txt', state_mod[:,:,2])
        np.savetxt(folder_train + 'cases_real.txt', cases)
        np.savetxt(folder_train + 'beta_real.txt', beta_real.squeeze())
        np.savetxt(folder_testg + 'S_real_testg.txt', state_mod_testg[:,:,0])
        np.savetxt(folder_testg + 'E_real_testg.txt', state_mod_testg[:,:,1])
        np.savetxt(folder_testg + 'I_real_testg.txt', state_mod_testg[:,:,2])
        np.savetxt(folder_testg + 'cases_real_testg.txt', cases_testg)
        np.savetxt(folder_testg + 'beta_real_testg.txt', beta_real_testg.squeeze())
    os.system('cp src/TestCase_epi_external_dataset_synthetic_scenario_test.py ' + folder)
    #time.sleep(35000)
