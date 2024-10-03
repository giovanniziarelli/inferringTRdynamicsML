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
trial             = 10   # defining observation width 
trial_sim         = 4    # varying this parameter for different initializations
plot_vars         = 1    # plot input temperatures and humidities
save_train        = 1    # save training dataset
save_testg        = 1    # save testing
various_IC        = 0    # for the synthetic case
estim_IC          = 0    # estimating online IC: WIP


#%% Set numerical parameters
dt            = 1                      # days
layers        = 1                      # number of hidden layers
neurons       = 4                      # number of neurons of each layer 
N_weeks       = 28                     # weeks of simulation 
dt_base       = 1                      # rescaling factor 
variance_init = 0.0001                 # initial variance of 
length_period = 7                      # weekly collected influence data
W_obs = int(T_obs / length_period)     # number of observation weeks
weeks = int(T_fin / length_period)     # number of weeks
t = np.arange(0, T_fin+dt, dt)[None,:] # numerical times

dt_num = 0.5#0.1#0.05#0.2
t_num = np.arange(0, T_fin+dt_num, dt_num)[None, :]

num_latent_states = 1    # dimension of the latent space
num_latent_params = 1    # number of unknown parameters to be estimated online
num_input_var     = 2    # number of input variables

#%% Model parameters
b_ref                   = 0.25                                     # initial guess for beta0
T_fin                   = 196                                      # days of available data from Influnet
T_obs_vec               = [29, 36, 43, 50, 57, 64, 71, 78, 85, 92]
T_obs                   = T_obs_vec[trial-1]  
alpha                   = 1 / 1.5 
gamma                   = 1 / 1.2
undetection_mean_value  = 0.23                                     # undetection rate for influenza, derived from Trentini et al.

#%% Preprocessing functions 

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
    return smoothed_data


#%% Generating folders
folder = '/home/giovanni/Desktop/inferringTRdynamicsML/TestCase_epi_real_temperature_humidity_neurons_' + str(neurons) + '_hlayers_' + str(layers) + '_observation__width_' + str(T_obs)+ '_random_init_'+ str(trial_sim)+ '/'

if os.path.exists(folder) == False:
    os.mkdir(folder)
folder_train = folder + 'train/'
folder_testg = folder + 'testg/'

# Loading datasets
input_dataset_path_temp    = '/home/giovanni/Desktop/inferringTRdynamicsML/italian-temperatures/tmedia_national_length'+str(T_fin)+'.csv'                                       # input dataset for Temperature
input_dataset_extern_temp  = np.loadtxt(input_dataset_path_temp)
input_dataset_path_umid    = '/home/giovanni/Desktop/inferringTRdynamicsML/italian-temperatures/umid_national_length'+str(T_fin)+'.csv'                                         # input dataset for Humidity
input_dataset_extern_umid  = np.loadtxt(input_dataset_path_umid)
output_dataset_path        = '/home/giovanni/Desktop/inferringTRdynamicsML/influnet/data-aggregated/epidemiological_data/processed_output_new_cases_'+str(N_weeks)+'_weeks.csv' # output dataset
output_dataset_extern      = np.loadtxt(output_dataset_path)

n_size = input_dataset_extern_temp.shape[0]
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
        'T': {'min': 5,  'max' : 17},
        'U': {'min': 60, 'max' : 85}
    },
    'output_fields': {
        'S': { 'min': 0, "max": 1 },
        'E': { 'min': 0, "max": 1 },
        'I': { 'min': 0, "max": 1 }
    }
}

#%% Dataset parameters
n_size = input_dataset_extern_temp.shape[0]

training_var_numpy_orig = np.stack((input_dataset_extern_temp, input_dataset_extern_umid), axis = 2)
cases                   = output_dataset_extern / undetection_mean_value

def epiModel_rhs(state, beta): # state dim (samples, 3), beta dim (samples,1)

    dSdt = - beta * tf.expand_dims(state[:,0],axis=1) * tf.expand_dims(state[:,2],axis=1)
    dEdt = beta * tf.expand_dims(state[:,0],axis=1) * tf.expand_dims(state[:,2],axis=1) - alpha * tf.expand_dims(state[:,1],axis=1)
    dIdt = alpha * tf.expand_dims(state[:,1],axis=1) - gamma* tf.expand_dims(state[:,2],axis=1)

    return tf.concat([dSdt, dEdt, dIdt], axis = 1)

beta_guess = b_ref 
beta_guess_IC = b_ref * np.ones((n_size,1))

# Deleting possible Nan values in input data
mask = np.isnan(training_var_numpy_orig)
indexes = []
for ii in range(mask.shape[0]):
    if mask[ii].any() == True:
        indexes.append(ii)
        continue

training_var_numpy_orig[:,:,0] = smoothing(training_var_numpy_orig[:,:,0])
training_var_numpy_orig[:,:,1] = smoothing(training_var_numpy_orig[:,:,1])
training_var_numpy_orig = np.delete(training_var_numpy_orig, indexes, 0)

# Interpolating data
training_var_numpy = np.zeros((n_size, t_num.shape[1], num_input_var))
training_var_numpy[:,:,0] = np.array([np.interp(t_num.squeeze(), t.squeeze(), training_var_numpy_orig[k,:,0]) for k in range(n_size - len(indexes))])
training_var_numpy[:,:,1] = np.array([np.interp(t_num.squeeze(), t.squeeze(), training_var_numpy_orig[k,:,1]) for k in range(n_size - len(indexes))])

cases = np.delete(cases, indexes, 0)
n_size = n_size - len(indexes)

if plot_vars: # specific for temperature and humidity
    seasons = [r'2010-2011', r'2011-2012',r'2012-2013',r'2013-2014',r'2014-2015',r'2015-2016',r'2016-2017',r'2017-2018',r'2018-2019',r'2019-2020']
    seasons_r = ['2010-2011', '2011-2012','2012-2013','2013-2014','2014-2015','2015-2016','2016-2017','2017-2018','2018-2019','2019-2020']
    mpl.rcParams["figure.constrained_layout.use"] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
    plt.rc('axes', prop_cycle=cycler(color=colors))
    
    width_pixels  = 300
    height_pixels = 200

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

    # Saving pdf 
    plt.savefig(os.path.join(folder, f'umid_all.pdf'), format='pdf')
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
    plt.close()


# dividing training and testing set
num_train = 9

#np.random.seed(8)# 2014/15 leave one out
np.random.seed(2)# 2018/19 leave one out
#np.random.seed(4)# 2017/28 leave one out

shuffle_indexes = np.arange(training_var_numpy.shape[0])
np.random.shuffle(shuffle_indexes)

training_var_numpy = training_var_numpy[shuffle_indexes]
cases = cases[shuffle_indexes]

dataset_train = {
        'times' : t_num.T, # [num_times]
        'inp_parameters' : None, # [num_samples x num_par]
        'inp_signals' : training_var_numpy[:num_train,:,:], # [num_samples x num_times x num_signals]
        'target_cases' : cases[:num_train], # [num_samples x num_times_week]
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

tf.random.set_seed(0)

# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset_epi_real(dataset_train, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_testg, problem, normalization, dt = None, num_points_subsample = None)
utils.process_dataset_epi_real(dataset_testg_full, problem, normalization, dt = None, num_points_subsample = None)

#%% Define model
constraint_IC = ClipConstraint(0, 1)

# Initial conditions
InitialValue_train = np.zeros((dataset_train['num_samples'], 1))
InitialValue_testg = np.zeros((dataset_testg['num_samples'], 1))

InitialValue_train[:,0] = 10/1000000 # initial seed
InitialValue_testg[:,0] = 10/1000000 # initial seed

IC_train = tf.Variable(InitialValue_train, trainable=False, constraint = constraint_IC)
IC_testg = tf.Variable(InitialValue_testg, trainable=False, constraint = constraint_IC)

initial_lat_state_train = tf.Variable(np.ones((dataset_train['num_samples'],1)) * beta_guess, trainable=True)
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

#%% Defining the neural network model
NNdyn = tf.keras.Sequential()
NNdyn.add(tf.keras.layers.Dense(num_latent_states, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=variance_init)))
for _ in range(layers - 1):
    NNdyn.add(tf.keras.layers.Dense(neurons, activation=tf.nn.tanh, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=variance_init)))

NNdyn.add(tf.keras.layers.Dense(num_latent_states, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=variance_init)))
NNdyn.summary()

#%% Defining the forward evolution model for the latent transmission rate
def evolve_dynamics(dataset, initial_lat_state, lat_param): #initial_state (n_samples x n_latent_state), delta (n_samples x num_lat_param)
    
    lat_state = initial_lat_state
    lat_state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    lat_state_history = lat_state_history.write(0, lat_state)
    dt_ref = normalization['time']['time_constant']
    
    # time integration
    for i in tf.range(dataset['num_times'] - 1):
        inputs = [lat_state / b_ref, dataset['inp_signals'][:,i,:]]
        if num_latent_params > 0:
            inputs.append(lat_param -1 )
        lat_state = tf.math.maximum(lat_state + dt_num/dt_ref * NNdyn(tf.concat(inputs, axis = -1)), tf.zeros_like(lat_state))
        lat_state_history = lat_state_history.write(i + 1, lat_state)

    return tf.transpose(lat_state_history.stack(), perm=(1,0,2))

#%% Defining the forward compartmental model
def evolve_model(dataset, lat_state, IC):
    IC_E = tf.reshape((dataset['target_cases'][:,0] / alpha /7 ), [IC.shape[0],1])
    IC_S = tf.reshape(1 - IC - IC_E, [IC.shape[0], 1])
    IC_all = tf.concat([IC_S, IC_E, IC], 1)
    
    state = IC_all
    state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    state_history = state_history.write(0, state)

    # time integration
    for i in tf.range(dataset['num_times'] - 1):
        state = state + dt_num * epiModel_rhs(state, lat_state[:, i, :])
        state_history = state_history.write(i + 1, state)

    return tf.transpose(state_history.stack(), perm=(1,0,2))

#%% Defining the ward compartmental model
def epiTRDNet(dataset, initial_lat_state, lat_param, IC):
    lat_states = evolve_dynamics(dataset,initial_lat_state, lat_param)
    return evolve_model(dataset, lat_states, IC)

#%% Loss functions
def loss_exp_beta(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    E = state[:,::dataset['frac'] * length_period ,1]#frac * 7
    S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    C = 100
    MSE = tf.reduce_mean(tf.square(C * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - C * dataset['target_cases']) / tf.square(C * dataset['target_cases'] )) 
    return MSE

def loss_exp_beta_abs(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    E = state[:,::dataset['frac'] * length_period ,1]#frac * 7
    S = state[:,::dataset['frac'] * length_period ,0]#frac * 7
    C = 10000
    MSE = tf.reduce_mean(tf.square(C * (S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]) - C * dataset['target_cases'])) 
    return MSE

def loss_s_inf(dataset, lat_states, IC):
    state = evolve_model(dataset, lat_states, IC)
    C = 10
    err   = tf.reduce_mean(tf.maximum(tf.pow(C * (0.7 - state[:,-1,0]), 5), tf.zeros_like(state[:,-1,0]))) 
    return err

def loss_H1(dataset, initial_lat_state, lat_param, IC):
    state = epiTRDNet(dataset, initial_lat_state, lat_param, IC)
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

def val_exp(dataset, initial_lat_state, lat_param, IC):
    state = epiTRDNet(dataset, initial_lat_state, lat_param, IC)
    E = tf.transpose(state[:,::dataset['frac'],1])
    new_cases_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])
    return tf.reduce_mean(tf.square((new_cases_summed_weekly) - (tf.transpose(dataset['target_cases']))))

def weights_reg(NN):
    return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in NN.layers])/len(NN.layers)

#%% Loss weights
nu_loss_train = 5e-1 # weight MSE metric
nu_r1         = 5e-3 # weight regularization latent param
nu_p          = 1e-7 # weight \beta_threshold
nu_p2         = 5e-2 # weight H1 norm on \beta (1st point FD)
nu_p3         = 1e-6 # weight H1 norm on \beta
nu_s_inf      = 1e-7 # penalization of final susceptibles
alpha_reg     = 1e-5 # regularization of trainable variables

#%% Training
trainable_variables_train = NNdyn.variables + [initial_lat_state_train] 
if num_latent_params > 0:
    trainable_variables_train += [lat_param_train]
if estim_IC > 0:
    trainable_variables_train += [IC_train]

loss = loss_exp
n_fact = int(dt/dt_num)

def loss_train():
    beta = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
    beta_coarse = beta[:,1::n_fact*7,:]
    l = nu_loss_train *loss_exp_beta(dataset_train, beta, IC_train)\
            + alpha_reg * weights_reg(NNdyn)\
            + nu_r1 * tf.reduce_mean(tf.square(lat_param_train -1))\
            + nu_p * tf.reduce_mean(tf.maximum(tf.pow((beta - 1.5), 5), tf.zeros_like(beta)))\
            + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - beta[:,1,:]) / dt_num))\
            + nu_s_inf * loss_s_inf(dataset_train, beta, IC_train)\
            + nu_p3 * tf.reduce_mean(tf.reduce_max(tf.square((beta_coarse[:,1:] - beta_coarse[:,:-1])/(dt_num*n_fact*7 )), axis = 1))
    return l 

def loss_train_abs():
    beta = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
    beta_coarse = beta[:,1::n_fact*7,:]
    l = nu_loss_train *loss_exp_beta_abs(dataset_train, beta, IC_train)\
            + alpha_reg * weights_reg(NNdyn)\
            + nu_r1 * tf.reduce_mean(tf.square(lat_param_train-1))\
            + nu_p * tf.reduce_mean(tf.maximum(tf.pow((beta - 1.5), 5), tf.zeros_like(beta)))\
            + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_train - beta[:,1,:]) / dt_num))\
            + nu_s_inf * loss_s_inf(dataset_train, beta, IC_train)\
            + nu_p3 * tf.reduce_mean(tf.reduce_max(tf.square((beta_coarse[:,1:] - beta_coarse[:,:-1])/(dt_num*n_fact*7 )), axis = 1))
    return l 

# Validation metric
def val_train():
    beta = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train)
    l = loss_exp_beta(dataset_train, beta, IC_train)
    return l 

val_metric = val_train

#%% Training (Routine step 1) 
opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train, val_metric)

num_epochs_Adam_train = 1000
num_epochs_BFGS_train = 5000

print('training (Adam)...')
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=5e-3))
print('training (BFGS)...')
opt_train.optimize_BFGS(num_epochs_BFGS_train)


#%% Training (Routine step 2) 
opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train_abs, val_metric)

num_epochs_Adam_train = 2000
num_epochs_BFGS_train = 3000

print('training (Adam)...')
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=1e-6))
print('training (BFGS)...')
opt_train.optimize_BFGS(num_epochs_BFGS_train)


#%% Saving the rhs-NN (NNdyn) 
NNdyn.save(folder + 'NNdyn')


#%% Testing 
if num_latent_params>0:
    num_sample_test = dataset_testg['num_samples']
    
    #%% Constants
    nu_r2_vec = [5e0, 1e0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    nu_r2 = nu_r2_vec[2] # penalization of the latent prior
    
    nu_loss_testg_vec = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]
    nu_loss_testg = nu_loss_testg_vec[9] # weight of the discrepancy error on testg 
    
    nu_b0_vec = [7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2]
    nu_b0 = nu_b0_vec[7]# penalization of the \beta0 prior 
    
    nu_p          = 1e-7 # weight \beta_threshold
    nu_p2         = 5e-2 # weight H1 norm on \beta (1st point FD)
    nu_s_inf      = 1e-7 # penalization of final susceptibles
    alpha_reg     = 1e-5 # regularization of trainable variables

    C_post    = tf.linalg.inv(tfp.stats.covariance(lat_param_train))
    C_post_b0 = tf.linalg.inv(tfp.stats.covariance(initial_lat_state_train))
    latent_train_mean = tf.reduce_mean(lat_param_train)
    initial_lat_state_train_mean = tf.reduce_mean(initial_lat_state_train)
    
    trainable_variables_testg = [initial_lat_state_testg] 
    trainable_variables_testg += [lat_param_testg]
    
    def loss_testg():
        beta = evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg)
        l = nu_loss_testg *loss_exp_beta(dataset_testg, beta, IC_testg)\
                + nu_r2 * tf.reduce_mean(tf.transpose(lat_param_testg - latent_train_mean ) * C_post * (lat_param_testg - latent_train_mean))\
                + nu_b0 * tf.reduce_mean(tf.transpose(initial_lat_state_testg - initial_lat_state_train_mean) * C_post_b0 * (initial_lat_state_testg - initial_lat_state_train_mean))\
                + nu_s_inf * loss_s_inf_testg(dataset_testg, beta, IC_testg)\
                + nu_p * tf.reduce_mean(tf.maximum(tf.pow(beta - 1.5, 5), tf.zeros_like(beta)))\
                + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_testg - beta[:,1,:]) / dt_num))
        return l

    def loss_testg_abs():
        beta = evolve_dynamics(dataset_testg, initial_lat_state_testg, lat_param_testg)
        l = nu_loss_testg *loss_exp_beta_abs(dataset_testg, beta, IC_testg)\
                + nu_r2 * tf.reduce_mean(tf.transpose(lat_param_testg - latent_train_mean ) * C_post * (lat_param_testg - latent_train_mean))\
                + nu_b0 * tf.reduce_mean(tf.transpose(initial_lat_state_testg - initial_lat_state_train_mean) * C_post_b0 * (initial_lat_state_testg - initial_lat_state_train_mean))\
                + nu_s_inf * loss_s_inf_testg(dataset_testg, beta, IC_testg)\
                + nu_p * tf.reduce_mean(tf.maximum(tf.pow(beta - 1.5, 5), tf.zeros_like(beta)))\
                + nu_p2 * tf.reduce_mean(tf.square((initial_lat_state_testg - beta[:,1,:]) / dt_num))
        return l 
    
    #%% Estimation 
    opt_testg = optimization.OptimizationProblem(trainable_variables_testg, loss_testg_abs, loss_testg)

    num_epochs_Adam_testg = 5000
    num_epochs_BFGS_testg = 3000
    
    print('training (Adam)...')
    opt_testg.optimize_keras(num_epochs_Adam_testg, tf.keras.optimizers.Adam(learning_rate=1e-3))# tf.keras.optimizers.Adam(learning_rate=5e-3))
    print('training (BFGS)...')
    opt_testg.optimize_BFGS(num_epochs_BFGS_testg)

    #%% Prediction 
    state_predict = epiTRDNet(dataset_testg_full, initial_lat_state_testg, lat_param_testg, IC_testg)

if os.path.exists(folder_train) == False:
    os.mkdir(folder_train)

if os.path.exists(folder_testg) == False:
    os.mkdir(folder_testg)

if save_train:
    
    beta_r = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train).numpy()
    delta_train_r = lat_param_train.numpy().squeeze()
    state_r = epiTRDNet(dataset_train, initial_lat_state_train, lat_param_train, IC_train).numpy()
    E = state_r[:,::dataset_train['frac'] * length_period,1]#frac * 7
    S = state_r[:,::dataset_train['frac'] * length_period,0]#frac * 7
    cases_r = S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]#alpha * state_r[:,:-1,1] 
    I_interp_r = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_r[k, :,2]) for k in range(state_r.shape[2])])
    t_num = t_num.squeeze()
    
    np.savetxt(folder_train + 'beta_rec_train.txt', beta_r.squeeze())
    np.savetxt(folder_train + 'cases_train.txt', cases.squeeze())
    np.savetxt(folder_train + 'delta_train.txt', delta_train_r.squeeze())
    np.savetxt(folder_train + 'S_rec_train.txt', state_r[:,:,0])
    np.savetxt(folder_train + 'E_rec_train.txt', state_r[:,:,1])
    np.savetxt(folder_train + 'I_rec_train.txt', state_r[:,:,2])
    np.savetxt(folder_train + 't_num.txt', t_num.squeeze())
    
    beta_train = beta_r.copy()

if save_testg:
    delta_train_r = lat_param_train.numpy().squeeze()
    delta_testg_r = lat_param_testg.numpy().squeeze()
    
    beta_r = evolve_dynamics(dataset_testg_full, initial_lat_state_testg, lat_param_testg).numpy()
    state_r = state_predict.numpy()
    E = state_r[:,::dataset_train['frac'] * length_period ,1]
    S = state_r[:,::dataset_train['frac'] * length_period ,0]
    cases_r = S[:,:-1] + E[:,:-1] - S[:,1:] - E[:,1:]
    delta_testg_r = lat_param_testg.numpy().squeeze()
    
    np.savetxt(folder_testg + 'beta_rec_testg.txt', beta_r.squeeze())
    np.savetxt(folder_testg + 'delta_testg.txt', [[delta_testg_r]])
    np.savetxt(folder_testg + 'S_rec_testg.txt', state_r[:,:,0])
    np.savetxt(folder_testg + 'E_rec_testg.txt', state_r[:,:,1])
    np.savetxt(folder_testg + 'I_rec_testg.txt', state_r[:,:,2])
    np.savetxt(folder_testg + 't_num.txt', t_num.squeeze())

os.system('cp /home/giovanni/Desktop/inferringTRdynamicsML/src/TestCase_epi_humidity.py ' + folder)
