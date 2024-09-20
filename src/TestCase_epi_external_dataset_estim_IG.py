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
b_ref = 0.025#0.5#0.025#0.7#0.77#0.1672#0.92851
tau = 0
layers = 2
err = 1e-1#1e-1

#%% Model parameters
alpha = 1 / 1.5#5#1.5
gamma = 1 / 1.2#10#1.2

plot_training_set = 1
save_train = 1
save_testg = 1
various_IC = 0
new_cases_loss = 1
estim_IC = 1

neurons = 4 
T_fin = 196#70
N_weeks = 28 
dt_base = 1#T_fin 

# Creating saving folders
folder = '/home/giovanni/Desktop/LDNets/neurons_' + str(neurons) + '_' + str(T_fin) + '_ITALY_lay' + str(layers) + '_MSE_loss_estim_IG_1/'

if os.path.exists(folder) == False:
    os.mkdir(folder)
folder_train = folder + 'train/'
folder_testg = folder + 'testg/'

num_latent_states = 1
num_latent_params = 1

# Loading datasets
input_dataset_path = '/home/giovanni/Desktop/LDNets/italian-temperatures/tmedia_national_length'+str(T_fin)+'.csv'#temperature_n.pkl'
input_dataset_extern = np.loadtxt(input_dataset_path)
output_dataset_path = '/home/giovanni/Desktop/LDNets/influnet/data-aggregated/epidemiological_data/processed_output_new_cases_'+str(N_weeks)+'_weeks.csv'#temperature_n.pkl'
output_dataset_extern = np.loadtxt(output_dataset_path)

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
        'T': {'min': 0, 'max' : 20}#{ 'min': 0, 'max': 20 }
    },
    'output_fields': {
        'I': { 'min': 0, "max": 1 },
    }
}

#%% Dataset parameters
n_size = input_dataset_extern.shape[0]#6
T_mean = 16
dt = 1
T_obs = 99 
length_period = 7
W_obs = int(T_obs / length_period)
weeks = int(T_fin / length_period)
t = np.arange(0, T_fin+dt, dt)[None,:]

dt_num = 0.5#0.1#0.05#0.2
t_num = np.arange(0, T_fin+dt_num, dt_num)[None, :]

training_var_numpy = input_dataset_extern #T_cos(T_mean, f, A, t)
cases              = 5 * output_dataset_extern # 20 % undetection, pugliese lunelli understanding

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
    smoothed_data = savgol_filter(data, 31, 5)
    #smoothed_data = tf.math.log(smoothed_data + 1)
    return smoothed_data

def weekly_avg(T):
    print(T.shape)
    T_reshaped = np.reshape(T[:,:-1], (T.shape[0], T.shape[1]//7, 7))
    return np.mean(T_reshaped, axis = 2)


# Deleting possible Nan values
mask = np.isnan(training_var_numpy)
indexes = []
for ii in range(mask.shape[0]):
    if mask[ii].any() == True:
        indexes.append(ii)
        continue
#training_var_numpy = weekly_avg(training_var_numpy)
training_var_numpy = np.delete(smoothing(training_var_numpy), indexes, 0)
#training_var_numpy = np.delete(smoothing(training_var_numpy), indexes, 0)
training_var_numpy = np.array([np.interp(t_num.squeeze(), t.squeeze(), training_var_numpy[k,:]) for k in range(n_size - len(indexes))])
cases = np.delete(cases, indexes, 0)
n_size = n_size - len(indexes)

plot_temps = 0
if plot_temps:
    plt.plot(training_var_numpy[0,:])
    plt.plot(training_var_numpy[1,:])
    plt.plot(training_var_numpy[2,:])
    plt.plot(training_var_numpy[3,:])
    plt.plot(training_var_numpy[4,:])
    plt.plot(training_var_numpy[5,:])
    plt.plot(training_var_numpy[6,:])
    plt.plot(training_var_numpy[7,:])
    plt.show()

    plt.plot(cases[0,:])
    plt.plot(cases[1,:])
    plt.plot(cases[2,:])
    plt.plot(cases[3,:])
    plt.plot(cases[4,:])
    plt.plot(cases[5,:])
    plt.plot(cases[6,:])
    plt.plot(cases[7,:])
    plt.show()
# Defining Datasets
dataset_train = {
        'times' : t_num.T, # [num_times]
        'inp_parameters' : None, # [num_samples x num_par]
        'inp_signals' : training_var_numpy[:,:,None], # [num_samples x num_times x num_signals]
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
        'inp_signals' : training_var_numpy[:,:,None], # [num_samples x num_times x num_signals]
        'target_cases' : cases,
        'num_times' : T_fin+1,
        'beta_guess' : beta_guess, 
        'time_vec' : t.T,
        'weeks' : weeks,
        'frac' : int(dt/dt_num)
}
dataset_testg = utils.cut_dataset_epi_real(dataset_testg_full, T_obs)
# For reproducibility (delete if you want to test other random initializations)
np.random.seed(0)
tf.random.set_seed(0)
# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset_epi_real(dataset_train, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_testg, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_testg_full, problem, normalization, dt = None, num_points_subsample = None)

#%% Define model
constraint_IC = ClipConstraint(0, 1)
# dynamics network
# Initial conditions
InitialValue_train = 0.00001 * np.ones((dataset_train['num_samples'], 1))
InitialValue_testg = 0.00001 * np.ones((dataset_testg['num_samples'], 1))
#InitialValue_train[:,1] = 10/1000000# * np.ones_like(cases[:,0])#cases[:,0]#10/50000000#0.01
#InitialValue_train = cases[:,0] / 7 / alpha# * np.ones_like(cases[:,0])#cases[:,0]#10/50000000#0.01
#InitialValue_testg[:,1] = 10/1000000# * np.ones_like(cases[:,0])#cases[:,0]#10 / 50000000
#InitialValue_testg = cases[:,0] / 7 / alpha# * np.ones_like(cases[:,0])#cases[:,0]#10/50000000#0.01
if estim_IC > 0:
    train_IC = True
else:
    train_IC = False

IC_train = tf.Variable(InitialValue_train.reshape((InitialValue_train.shape[0],1)), trainable=train_IC, constraint = constraint_IC)
IC_testg = tf.Variable(InitialValue_testg.reshape((InitialValue_testg.shape[0],1)), trainable=train_IC, constraint = constraint_IC)

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
            tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, input_shape = input_shape, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
            tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
            #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
            #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
            #tf.keras.layers.Dense(neurons, activation = tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)),
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
        #inputs = [lat_state / b_ref, dataset['inp_signals'][:,i,:]]
        inputs = [lat_state / b_ref, dataset['inp_signals'][:,i,:]]
        #tf.print(inputs)
        if num_latent_params > 0:
            inputs.append(lat_param)
        lat_state = tf.math.maximum(lat_state + dt_num/dt_ref * NNdyn(tf.concat(inputs, axis = -1)), tf.zeros_like(lat_state))
        lat_state_history = lat_state_history.write(i + 1, lat_state)

    return tf.transpose(lat_state_history.stack(), perm=(1,0,2))

# initialize epi model parameters

def evolve_model(dataset, lat_state, IC):
    IC_S = tf.reshape(1 - IC, [IC.shape[0], 1])
    IC_all = tf.concat([IC_S, IC, tf.zeros_like(IC)], 1)
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
    E = state[:,::dataset['frac'] * length_period ,1]#frac * 7
    S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    
    MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']))# / tf.square( (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:])+ 1e-6 )) 
    
    return MSE

def loss_exp_beta(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    #E = tf.transpose(state[:,::dataset['frac'] * 7,1])
    E = state[:,::dataset['frac'] * length_period ,1]#frac * 7
    S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    #new_cases_summed_weekly = tf.transpose(tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])]))
    #MSE = tf.reduce_mean(tf.square(new_cases_summed_weekly - dataset['target_cases']) / tf.square( tf.square( dataset['target_cases']) + 1e-6 )) 
    
    MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']))# / tf.square( (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:])+ 1e-6 )) 
    
    #MSE = tf.reduce_mean(tf.square(100 * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - 100 * dataset['target_cases']) / tf.square( 100 * dataset['target_cases'] )) 
    return MSE

def val_exp(dataset, initial_lat_state, lat_param, IC):
    state = LDNet(dataset, initial_lat_state, lat_param, IC)
    E = tf.transpose(state[:,::dataset['frac'],1])
    new_cases_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])
    return tf.reduce_mean(tf.square((new_cases_summed_weekly) - (tf.transpose(dataset['target_cases']))))

#%% Loss weights
nu             = 1e-3#1e-3#1e0#1e2#1.75e-3#1e-1#1e-1#i1e0 good 1e0#1e-7#1e-2#-4#1e0#1e2
nu_h1          = 0#1e-1
nu_r1          = 0#1e0#1e-4#1e-2#1e-4#1e-5
nu_p           = 1e2#1e2#1e0#1e4#1e1#1e-1#1e-8#1e-2#6
nu_p2          = 0#1e2#1e0#1e-1#1e-8#1e-3#1e-3#6
alpha_reg      = 1e-5#1e#1e-5#1e-2#1e-2#1e-1# 1e-8#1e-8#1e1#1e-2#1e-3#5e-5#5e-5#5e-6#4.7e-3
alpha_reg_IC_I = 0#1e0#1e-8#1e-4#1e-3
alpha_reg_IC_E = 1e7#1e4#1e6#1e0#1e4#1e0#1e-8#1e-4#1e-3
alpha_reg_IC_E_pos = 1e7#1e4#1e6#1e0#1e4#1e0#1e-8#1e-4#1e-3

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
    l = nu *loss_exp_beta(dataset_train, beta, IC_train) + alpha_reg * weights_reg(NNdyn) + nu_r1 * tf.reduce_mean(tf.square(lat_param_train -1)) + nu_p * tf.reduce_mean(tf.maximum(tf.pow(beta - 1.2, 7), tf.zeros_like(beta))) + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - beta[:,1,:]) / dt_num)) + alpha_reg_IC_E * tf.reduce_mean(tf.maximum(tf.pow(IC_train - 0.001, 5), tf.zeros_like(IC_train))) - alpha_reg_IC_E_pos * tf.reduce_mean(tf.minimum(tf.pow(IC_train-0.000001, 5), tf.zeros_like(IC_train)))#dataset_train['target_cases'][:,0] / 7/ alpha))
            #+ alpha_reg_IC_E * tf.reduce_mean(tf.square(IC_train - dataset_train['target_cases'][:,0] / 7/ alpha))
    return l 

####loss_train = lambda: nu  * loss(dataset_train, initial_lat_state_train, lat_param_train, IC_train)\
####        + alpha_reg      * weights_reg(NNdyn)\
####        + nu_r1          * tf.reduce_mean(tf.square(lat_param_train - 1))\
####        + nu_p           * tf.reduce_mean(tf.reduce_sum(tf.square(evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train) - 1), axis = 0))\
####        + nu_p2          * tf.reduce_mean(tf.square((initial_lat_state_train - evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)[:,1,:]) / dt_num))\
####        + alpha_reg_IC_E * tf.reduce_mean(tf.square(IC_train - dataset_train['target_cases'][:,0] / 7/ alpha))
####        #+ alpha_reg_IC_I * tf.reduce_mean(tf.square(IC_train[:,1] - 10/10000000))

val_train = lambda: val_exp(dataset_train, initial_lat_state_train, lat_param_train, IC_train)         
         
val_metric = loss_train

opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train, val_metric)

num_epochs_Adam_train = 500#500 good
num_epochs_BFGS_train = 10000

print('training (Adam)...')
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=1e-4))#1e-3))#1e-2 good
print('training (BFGS)...')
opt_train.optimize_BFGS(num_epochs_BFGS_train)

#%% Estimation

if num_latent_params>0:
    num_sample_test = dataset_testg['num_samples']
    trainable_variables_testg = [initial_lat_state_testg] 
    trainable_variables_testg += [lat_param_testg]
    if estim_IC > 0:
        trainable_variables_testg += [IC_testg]
    
    #%% Constants
    nu_r2         = 1.0e0#1e-2#5e-3#1e-5
    nu_loss_testg = 6e-1#1e-1
    nu_b0         = 1e-4
    nu_IC_I       = 1e-4
    nu_IC_E       = 1e-2

    C_post      = tf.linalg.inv(tfp.stats.covariance(lat_param_train))
    C_post_b0   = tf.linalg.inv(tfp.stats.covariance(initial_lat_state_train))
    C_post_IC_E = tf.linalg.inv(tfp.stats.covariance(tf.reshape(IC_train, [IC_train.shape[0],1])))
    #C_post_IC_I = tf.linalg.inv(tfp.stats.covariance(tf.reshape(IC_train[:,1], [IC_train.shape[0],1])))
    
    latent_train_mean            = tf.reduce_mean(lat_param_train)
    initial_lat_state_train_mean = tf.reduce_mean(initial_lat_state_train)
    IC_E_train_mean              = tf.reduce_mean(IC_train)
    #IC_I_train_mean              = tf.reduce_mean(IC_train[:,1])

####loss_testg = lambda: nu_loss_testg * loss(dataset_testg, initial_lat_state_testg, lat_param_testg, IC_testg)\
####        + nu_r2                    * tf.reduce_mean(tf.transpose(lat_param_testg - latent_train_mean ) * C_post * (lat_param_testg - latent_train_mean))\
####        + nu_b0                    * tf.reduce_mean(tf.transpose(initial_lat_state_testg - initial_lat_state_train_mean) * C_post_b0 * (initial_lat_state_testg - initial_lat_state_train_mean))\
####        + nu_p                     * tf.reduce_mean(tf.reduce_sum(tf.square(evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg) - 1), axis = 0))\
####        + nu_p2                    * tf.reduce_mean(tf.square((initial_lat_state_testg - evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg)[:,1,:]) / dt_num))\
####        + nu_IC_E                  * tf.reduce_mean(tf.square(tf.transpose(IC_testg - IC_E_train_mean) * C_post_IC_E * (IC_testg - IC_E_train_mean)))\
####        #+ nu_IC_I                  * tf.reduce_mean(tf.square(tf.transpose(IC_testg[:,1] - IC_I_train_mean) * C_post_IC_I * (IC_testg[:,1] - IC_I_train_mean)))

            #+ nu_IC * tf.reduce_mean(tf.transpose(IC_testg - IC_train_mean) * C_post_IC * (IC_testg - IC_train_mean))
    def loss_testg():
        beta = evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg)
        l = nu_loss_testg *loss_exp_beta(dataset_testg, beta, IC_testg)\
                + nu_r2 * tf.reduce_mean(tf.transpose(lat_param_testg - latent_train_mean ) * C_post * (lat_param_testg - latent_train_mean))\
                + nu_b0 * tf.reduce_mean(tf.transpose(initial_lat_state_testg - initial_lat_state_train_mean) * C_post_b0 * (initial_lat_state_testg - initial_lat_state_train_mean))\
                + nu_p * tf.reduce_mean(tf.maximum(tf.pow(beta - 1.2, 6), tf.zeros_like(beta)))\
                + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_testg - beta[:,1,:]) / dt_num))\
                + nu_IC_E * tf.reduce_mean(tf.square(tf.transpose(IC_testg - IC_E_train_mean) * C_post_IC_E * (IC_testg - IC_E_train_mean)))
        return l 

    opt_testg = optimization.OptimizationProblem(trainable_variables_testg, loss_testg, loss_testg)

    num_epochs_Adam_testg = 500
    num_epochs_BFGS_testg = 5000
    
    print('training (Adam)...')
    opt_testg.optimize_keras(num_epochs_Adam_testg, tf.keras.optimizers.Adam(learning_rate=1e-4))
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
    plt.savefig(folder_train + 'train_loss.png')
    plt.show()
    
    beta_r = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train).numpy()
    delta_train_r = lat_param_train.numpy().squeeze()
    state_r = LDNet(dataset_train, initial_lat_state_train, lat_param_train, IC_train).numpy()
    E = state_r[:,::dataset_train['frac'] * length_period ,1]#frac * 7
    S = state_r[:,::dataset_train['frac'] * length_period ,0]#frac * 7
    cases_r = S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]#alpha * state_r[:,:-1,1] 
    I_interp_r = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_r[k, :,2]) for k in range(n_size)])
    t_num = t_num.squeeze()
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
        plt.show()

    np.savetxt(folder_train + 'beta_rec_train.txt', beta_r.squeeze())
    np.savetxt(folder_train + 'cases_train.txt', cases.squeeze())
    np.savetxt(folder_train + 'delta_train.txt', delta_train_r.squeeze())
    np.savetxt(folder_train + 'S_rec_train.txt', state_r[:,:,0])
    np.savetxt(folder_train + 'E_rec_train.txt', state_r[:,:,1])
    np.savetxt(folder_train + 'I_rec_train.txt', state_r[:,:,2])
    np.savetxt(folder_train + 't_num.txt', t_num.squeeze())

if save_testg:
    fig_delta, axs_delta = plt.subplots(1,1)
    delta_train_r = lat_param_train.numpy().squeeze()
    delta_testg_r = lat_param_testg.numpy().squeeze()
    axs_delta.scatter(delta_train_r, delta_testg_r)#, 'o')
    axs_delta.set_xlabel('latent param train')
    axs_delta.set_ylabel('latent param testg')
    plt.savefig(folder_train + 'train_testg.png')
    plt.show()
    
    beta_r = evolve_dynamics(dataset_testg_full, initial_lat_state_testg, lat_param_testg).numpy()
    state_r = state_predict.numpy()
    E = state_r[:,::dataset_train['frac'] * length_period ,1]#frac * 7
    S = state_r[:,::dataset_train['frac'] * length_period ,0]#frac * 7
    cases_r = S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]#alpha * state_r[:,:-1,1] 
    highlight_color = 'lightgreen'
    for k in range(int(dataset_testg['num_samples'])):
        fig,ax = plt.subplots(1,4, figsize = (18,7))
        ax[0].plot(np.arange(1,len(cases[k,:])+1), cases[k,:], 'o--',linewidth = 2, label = 'Real')
        ax[0].plot(np.arange(1, len(cases_r[k,:])+1), cases_r[k,:],'o-',linewidth = 2,  label = 'NN')
        ax[0].set_xlabel('weeks')
        ax[0].set_ylabel('Weekly incidence')
        ax[0].set_title('New Cases') 
        ax[0].axvspan(0, W_obs, facecolor=highlight_color, alpha = 0.5)
        ax[1].plot(t_num, beta_r[k,:],linewidth = 2)
        ax[1].set_xlabel('days')
        ax[1].set_title('Transmission Rate') 
        ax[1].axvspan(0, T_obs, facecolor=highlight_color, alpha = 0.5)
        ax[2].plot(t_num, beta_r[k,:]/alpha,linewidth = 2)
        ax[2].set_title('Reproduction number') 
        ax[2].set_xlabel('days')
        ax[2].axvspan(0, T_obs, facecolor=highlight_color, alpha = 0.5)
        ax[3].plot(t_num, training_var_numpy[k,:],linewidth = 2)
        ax[3].set_title('Temperature')
        ax[3].set_xlabel('days')
        ax[3].axvspan(0, T_obs, facecolor=highlight_color, alpha = 0.5)
        fig.tight_layout()
        plt.savefig(folder + 'out_testg_' + str(k) + '.pdf')
        plt.show()
    delta_testg_r = lat_param_testg.numpy().squeeze()
    I_interp_r = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_r[k, :,2]) for k in range(n_size)])
    
    np.savetxt(folder_testg + 'beta_rec_testg.txt', beta_r.squeeze())
    np.savetxt(folder_testg + 'delta_testg.txt', delta_testg_r.squeeze())
    np.savetxt(folder_testg + 'S_rec_testg.txt', state_r[:,:,0])
    np.savetxt(folder_testg + 'E_rec_testg.txt', state_r[:,:,1])
    np.savetxt(folder_testg + 'I_rec_testg.txt', state_r[:,:,2])
    np.savetxt(folder_testg + 't_num.txt', t_num.squeeze())
os.system('cp /home/giovanni/Desktop/LDNets/src/TestCase_epi_external_dataset_estim_IG.py ' + folder)
