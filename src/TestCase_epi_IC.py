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

# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')

import utils
import optimization
class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

#%% Set some hyperparameters
dt = 0.2
t_max = 12
b_ref = 0.77#0.1672#0.92851
tau = 0
layers = 2
err = 0#1e-1#1e-1

plot_train = 1
plot_testg = 1
various_IC = 1
Delta_loss = 0
new_cases_loss = 1

neurons = 4 
T_fin = 56#70
dt_base = 1#T_fin 

folder = '/home/giovanni/Desktop/LDNets/neurons_' + str(neurons) + '_' + str(T_fin) + '_days_random_20_lay' + str(layers) +'_IC/'
#folder = '/home/giovanni/Desktop/LDNets/prova_noise/'
if os.path.exists(folder) == False:
    os.mkdir(folder)
folder_train = folder + 'train/'
folder_testg = folder + 'testg/'

num_latent_states = 1
num_latent_params = 1
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
        'T': { 'min': 0, 'max': 30 }
    },
    'output_fields': {
        'I': { 'min': 0, "max": 1 },
    }
}

#prova dataset pkl
#dataset_path = '/home/giovanni/Desktop/LDNets/datasets/temperature_n_2.pkl'
dataset_path = '/home/giovanni/Desktop/LDNets/datasets/temperature_not_modified_new_b_ref.pkl'#temperature_n.pkl'
dataset_path = '/home/giovanni/Desktop/LDNets/datasets/temperature_new_50_new.pkl'#temperature_not_modified_new_b_ref_40.pkl'#temperature_n.pkl'
dataset_path = '/home/giovanni/Desktop/LDNets/datasets/temperature_new_20_new.pkl'#temperature_not_modified_new_b_ref_40.pkl'#temperature_n.pkl'
with open(dataset_path, 'rb') as file:
    dataset_extern, val_extern, test_extern, beta0_train, beta0_val, beta0_test = pickle.load(file)
betas, deltas = shf.compute_beta_delta([dataset_extern, val_extern, test_extern], \
        [beta0_train, beta0_val, beta0_test],\
        t_max, tau, b_ref)
beta_train, beta_val, beta_test = betas
delta_train, delta_val, delta_test = deltas
beta_train.reshape((beta_train.shape[0], beta_train.shape[1],1))

#%% Dataset parameters
n_size = beta_train.shape[0]#6
T_mean = 16
dt = 1
T_obs = 14 
length_period = 7
weeks = int(T_fin / length_period)
tau = 100 #useful for linear ode
t = np.arange(0, T_fin+dt, dt)[None,:]
nu = 1e-2#1e0#1e2

dt_num = 0.5#0.1#0.05#0.2
t_num = np.arange(0, T_fin+dt_num, dt_num)[None, :]

training_var_numpy = dataset_extern #T_cos(T_mean, f, A, t)
training_var_numpy = np.array([np.interp(t_num.squeeze(), t.squeeze(), dataset_extern[k,:]) for k in range(n_size)])

alpha = 1 / 1.5#5#1.5
gamma = 1 / 1.2#10#1.2

def epiModel_rhs(state, beta): # state dim (samples, 3), beta dim (samples,1)

    dSdt = - beta * tf.expand_dims(state[:,0],axis=1) * tf.expand_dims(state[:,2],axis=1)
    dEdt = beta * tf.expand_dims(state[:,0],axis=1) * tf.expand_dims(state[:,2],axis=1) - alpha * tf.expand_dims(state[:,1],axis=1)
    dIdt = alpha * tf.expand_dims(state[:,1],axis=1) - gamma* tf.expand_dims(state[:,2],axis=1)

    return tf.concat([dSdt, dEdt, dIdt], axis = 1)

def beta_evolution(beta, T, delta_):
    beta_rhs = (delta_) * (T/T_mean - 1) * ((beta)**2 / b_ref - (beta))#1/tau * (beta - 2 + (T - 278) * 0.02)
    return beta_rhs

beta_guess = b_ref 
beta_guess_IC = b_ref * np.ones((n_size,1))

S0 = 0.99
E0 = 0
I0 = 0.01
IC = np.repeat(np.array([S0, E0, I0])[:,None],n_size, axis = 1).T# num_sample x num_variabili
def IC_fun(num_samples):
    IC = np.zeros((num_samples,3))
    IC[:,0] = np.random.uniform(0.95,1.0,num_samples)
    IC[:,2] = 1 - IC[:,0] #np.minimum(np.random.uniform(0,0.1,num_samples), 1 - IC[:,0])
    IC[:,1] = 0#1 - IC[:,0] - IC[:,2]
    return IC
if various_IC > 0:
    IC = IC_fun(n_size)
state_mod = np.zeros((n_size,t_num.shape[1], 3))
state_mod[:,0,:] = IC

#initialize real beta
beta_real = np.zeros((n_size, t_num.shape[1], 1))
beta_real[:,0,:] = beta0_train[:,None]

plt.plot(training_var_numpy.T)
plt.show()

#solving the ODEs for retrieving beta and state
for i in range(t_num.shape[1]-1):
    beta_real[:, i+1, :] = beta_real[:, i, :] + dt_num * beta_evolution(beta_real[:,i,:], training_var_numpy[:,i][:,None], delta_train[:,None])
    state_mod[:,i+1,:] = state_mod[:,i,:] + dt_num * epiModel_rhs(state_mod[:,i,:], beta_real[:,i+1,:])
print('beta qui')
state_mod[:,:,2] = np.maximum(state_mod[:,:,2] * (1 + err * np.random.randn(state_mod.shape[0], state_mod.shape[1])), 0.0)
state_mod[:,:,0] = 1 - state_mod[:,:,2]
#print(beta_real[np.ma.fix_invalid(beta_real).mask.any(axis=1)])
mask = np.isnan(beta_real)
indexes = []
for ii in range(mask.shape[0]):
    if mask[ii].any() == True:
        indexes.append(ii)
        continue

training_var_numpy = np.delete(training_var_numpy, indexes, 0)
beta_real          = np.delete(beta_real, indexes, 0)
state_mod          = np.delete(state_mod, indexes, 0)
delta_train        = np.delete(delta_train, indexes, 0)
IC                 = np.delete(IC, indexes, 0)

n_size = n_size - len(indexes)
print(state_mod.shape)
#print(beta_real[indexes[0], :, :])
#print(delta_train[indexes[0]])
#print(training_var_numpy[indexes[0], :] / T_mean - 1)


if Delta_loss:
    I_interp = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,2]) for k in range(n_size)])
    Delta_I = (I_interp[:, 1:] - I_interp[:, :-1])
    print('Delta_I sahpe')
    print(Delta_I.shape)
    plt.plot(Delta_I)
    plt.show()
    inf_weekly = np.array([np.sum(Delta_I[:,length_period*k:length_period*(k+1)], axis = 1) for k in range(weeks)]).T
    print(inf_weekly.shape)
else:
    I_interp = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,2]) for k in range(n_size)])
    inf_weekly = np.array([np.sum(I_interp[:,length_period*k:length_period*(k+1)], axis = 1) for k in range(weeks)]).T

E_interp = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_mod[k, :,1]) for k in range(n_size)])
cases    = np.array([np.sum(alpha * E_interp[:,length_period*k:length_period*(k+1)], axis = 1) for k in range(weeks)]).T
cases_mean = np.mean(cases)
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
InitialValue_train = np.zeros((dataset_train['num_samples'], 1))
InitialValue_testg = np.zeros((dataset_testg['num_samples'], 1))
#InitialValue_train[:,0] = 0.01
#InitialValue_testg[:,0] = 0.01
##InitialValue_train[:,1] = 0.01#10/50000000#0.01
##InitialValue_testg[:,1] = 0.01#10 / 50000000
#tf.print(InitialValue_train)
#tf.print(InitialValue_testg)
IC_train = tf.Variable(InitialValue_train, trainable=True, constraint = constraint_IC)
IC_testg = tf.Variable(InitialValue_testg, trainable=True, constraint = constraint_IC)

# dynamics network
initial_lat_state_train = tf.Variable(np.ones((dataset_train['num_samples'],1)) * beta_guess, trainable=True)
print(dataset_train)
initial_lat_state_testg = tf.Variable(np.ones((dataset_testg['num_samples'],1)) * beta_guess, trainable=True)
if num_latent_params > 0:
    lat_param_train = tf.Variable(np.zeros((dataset_train['num_samples'], num_latent_params)), trainable=True)
    lat_param_testg = tf.Variable(np.zeros((dataset_testg['num_samples'], num_latent_params)), trainable=True)
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
            tf.keras.layers.Dense(num_latent_states, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
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
        if num_latent_params > 0:
            inputs.append(lat_param)
        lat_state = lat_state + dt_num/dt_ref * NNdyn(tf.concat(inputs, axis = -1))
        lat_state_history = lat_state_history.write(i + 1, lat_state)

    return tf.transpose(lat_state_history.stack(), perm=(1,0,2))

# initialize epi model parameters

def evolve_model(dataset, lat_state, IC):
    
    IC_S = tf.reshape(1 - IC[:,0], [IC.shape[0], 1])
    IC_all = tf.concat([IC_S, tf.zeros_like(IC), IC], 1)
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
def loss_inf(dataset, initial_lat_state, lat_param):
    state = LDNet(dataset, initial_lat_state, lat_param)
    I = tf.transpose(state[:,::dataset['frac'],2])
    I_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(I[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])
    MSE = nu * tf.reduce_mean(tf.square(I_summed_weekly - tf.transpose(dataset['target_incidence'])) / (tf.square(tf.transpose(dataset['target_incidence']) + 1e-10))) 
    return MSE

def loss_exp(dataset, initial_lat_state, lat_param, IC):
    state = LDNet(dataset, initial_lat_state, lat_param, IC)
    E = tf.transpose(state[:,::dataset['frac'],1])
    new_cases_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(alpha * E[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])
    MSE = nu * tf.reduce_mean(tf.square(new_cases_summed_weekly - tf.transpose(dataset['target_cases'])) / (tf.square(tf.transpose(dataset['target_cases']) + 1e-10))) 
    return MSE



def loss_delta(dataset, initial_lat_state, lat_param):
    state = LDNet(dataset, initial_lat_state, lat_param)
    I = tf.transpose(state[:,::dataset['frac'],2])
    Delta_I = (I[1:,:] - I[:-1, :]) 
    print('Delta_I shape')
    print(Delta_I.shape)
    #I_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(I[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])
    I_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(Delta_I[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])
    target = tf.transpose(dataset['target_incidence'])#-1 * tf.math.log(tf.transpose(dataset['target_incidence']))
    #MSE = nu * tf.reduce_mean(tf.square(I_summed_weekly - target ) / (tf.square(I_summed_weekly) + 1e-10)) 
    MSE = nu * tf.reduce_mean(tf.square(I_summed_weekly - target ) /(tf.square(target) + 1e-10) )# / (tf.square(I_summed_weekly) + 1e-10)) 
    #MSE = nu * tf.reduce_mean(tf.math.log(1 + tf.square(I_summed_weekly - target ) / (tf.square(I_summed_weekly) + 1e-10))) 
    #MSE = nu * tf.reduce_mean(tf.square(I_summed_weekly - tf.transpose(dataset['target_incidence'])) / (tf.square(tf.transpose(dataset['target_incidence']) + 1e-10))) 
    return MSE

nu_h1 = 1e0 
def loss_H1(dataset, initial_lat_state, lat_param):
    state = LDNet(dataset, initial_lat_state, lat_param)
    I = tf.transpose(state[:,::dataset['frac'],2])
    state_v = state[:,::dataset['frac'],:]
    
    I_summed_weekly = tf.convert_to_tensor([tf.reduce_sum(I[length_period*k:length_period*(k+1),:], axis = 0) for k in range(dataset['weeks'])])

    beta = evolve_dynamics(dataset, initial_lat_state, lat_param)
    beta = beta[:,::dataset['frac'],:]
    rhs = tf.transpose(tf.convert_to_tensor([epiModel_rhs(state_v[:,i,:],beta[:,i,0])[:,2] for i in range(T_fin)]))
    #print(rhs)
    rhs_weekly = tf.convert_to_tensor([tf.reduce_mean(rhs[:, length_period*k: length_period*(k+1)], axis = 1) for k in range(dataset['weeks'] - 1)])
    rhs = tf.reduce_mean(tf.square(rhs_weekly - (I_summed_weekly[1:,:] - I_summed_weekly[:-1,:])/(dt * length_period)), axis = 0)
    
    MSE = nu_h1 * tf.reduce_mean(rhs) 
    return MSE

#%% Training
trainable_variables_train = NNdyn.variables + [initial_lat_state_train] 
if num_latent_params > 0:
    trainable_variables_train += [lat_param_train]
trainable_variables_train += [IC_train]
nu_r1  = 1e-2#1e-3
alpha_IC = 1e-3
def weights_reg(NN):
    return sum([tf.reduce_mean(tf.square(lay.kernel)) for lay in NN.layers])/len(NN.layers)

alpha_reg = 1e-5# 1e-8#1e-8#1e1#1e-2#1e-3#5e-5#5e-5#5e-6#4.7e-3

if Delta_loss:
    loss = loss_delta
else:
    loss = loss_inf

if new_cases_loss:
    loss = loss_exp

loss_train = lambda: loss(dataset_train, initial_lat_state_train, lat_param_train, IC_train)  + alpha_reg * weights_reg(NNdyn)+ nu_r1 * tf.reduce_mean(tf.square(lat_param_train)) + alpha_IC * tf.reduce_mean(tf.square(IC_train - cases_mean))# + alpha_reg * weights_reg(NNdyn) 
val_metric = lambda: tf.reduce_mean(tf.square(evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train) - tf.convert_to_tensor(beta_real)))
opt_train = optimization.OptimizationProblem(trainable_variables_train, loss_train, val_metric)

num_epochs_Adam_train = 2000
num_epochs_BFGS_train = 10000


print('training (Adam)...')
opt_train.optimize_keras(num_epochs_Adam_train, tf.keras.optimizers.Adam(learning_rate=1e-4))
print('training (BFGS)...')
opt_train.optimize_BFGS(num_epochs_BFGS_train)

#%% Estimation

if num_latent_params>0:
    trainable_variables_testg = [initial_lat_state_testg] 
    trainable_variables_testg += [lat_param_testg]
    trainable_variables_testg += [IC_testg]
# TO DO: regularization on lat_param_testg
    C_post = tf.linalg.inv(tfp.stats.covariance(lat_param_train))
    nu_r2 = 1e-1#1e-2#5e-3#1e-5
    nu_loss_testg = 1e1#1e-1
    loss_testg = lambda: nu_loss_testg * loss(dataset_testg, initial_lat_state_testg, lat_param_testg, IC_testg) + nu_r2 * tf.reduce_mean(tf.transpose((lat_param_testg - tf.reduce_mean(lat_param_train))) * C_post * (lat_param_testg - tf.reduce_mean(lat_param_train)))

    opt_testg = optimization.OptimizationProblem(trainable_variables_testg, loss_testg, loss_testg)

    num_epochs_Adam_testg = 600
    num_epochs_BFGS_testg = 5000


    print('training (Adam)...')
    opt_testg.optimize_keras(num_epochs_Adam_testg, tf.keras.optimizers.Adam(learning_rate=1e-3))
    print('training (BFGS)...')
    opt_testg.optimize_BFGS(num_epochs_BFGS_testg)

    #%% Prediction

    state_predict = LDNet(dataset_testg_full, initial_lat_state_testg, lat_param_testg, IC_testg)
    print('Cpost: ', C_post)

if os.path.exists(folder_train) == False:
    os.mkdir(folder_train)

if os.path.exists(folder_testg) == False:
    os.mkdir(folder_testg)

if plot_train:
    fig_loss, axs_loss = plt.subplots(1, 1)
    axs_loss.loglog(opt_train.iterations_history, opt_train.loss_train_history, 'o-', label = 'training loss')
    axs_loss.axvline(num_epochs_Adam_train)
    axs_loss.set_xlabel('epochs'), plt.ylabel('loss')
    axs_loss.legend()
    plt.savefig(folder_train + 'train_loss.png')
    plt.show()
       
    if num_latent_params > 0:   
        fig_delta, axs_delta = plt.subplots(1,1)
        delta_train_r = lat_param_train.numpy().squeeze()
        axs_delta.scatter(delta_train, delta_train_r)#, 'o')
        axs_delta.set_ylabel('latent param reconstructed')
        axs_delta.set_xlabel('latent param true')
        plt.savefig(folder_train + 'train_latent.png')
        plt.show()

    beta_r = evolve_dynamics(dataset_train, initial_lat_state_train, lat_param_train).numpy()
    state_r = LDNet(dataset_train, initial_lat_state_train, lat_param_train, IC_train).numpy()
    I_interp_r = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_r[k, :,2]) for k in range(n_size)])
    I_r_weekly = np.array([np.sum(I_interp_r[:,length_period*k:length_period*(k+1)], axis = 1) for k in range(weeks)]).T
    t_num = t_num.squeeze()
    
    np.savetxt(folder_train + 'beta_rec_train.txt', beta_r.squeeze())
    np.savetxt(folder_train + 'beta_real.txt', beta_real.squeeze())
    np.savetxt(folder_train + 'delta_train.txt', delta_train_r.squeeze())
    np.savetxt(folder_train + 'delta_true.txt', delta_train.squeeze())
    np.savetxt(folder_train + 'S_rec_train.txt', state_r[:,:,0])
    np.savetxt(folder_train + 'I_rec_train.txt', state_r[:,:,1])
    np.savetxt(folder_train + 'R_rec_train.txt', state_r[:,:,2])
    np.savetxt(folder_train + 'S_real.txt', state_mod[:,:,0])
    np.savetxt(folder_train + 'I_real.txt', state_mod[:,:,1])
    np.savetxt(folder_train + 'R_real.txt', state_mod[:,:,2])
    np.savetxt(folder_train + 'Incidence_rec_train.txt', I_r_weekly.squeeze())
    np.savetxt(folder_train + 'Incidence_real.txt', inf_weekly.squeeze())
    np.savetxt(folder_train + 't_num.txt', t_num.squeeze())

    for i in range(int(n_size/10)):    
        fig_inf, axs_inf = plt.subplots(3, 10, figsize = (30, 8))#n_size)
        for j  in range(10):
            axs_inf[0,j].plot(t_num,beta_r[j + 10*i,:,0], '--', linewidth=4)
            axs_inf[0,j].plot(t_num,beta_real[j + 10*i,:,0], linewidth=4, zorder=1)
            axs_inf[0,j].set_ylim(0.5,2.5)
            axs_inf[0,j].set_ylabel('Transmission rate')
            #axs_inf[0,j].set_title('Latent true = ' + "{:.2f}".format(delta_train[10*i + j]))
        
            axs_inf[1,j].set_ylabel('Weekly incidence')
            axs_inf[1,j].plot(np.arange(1,weeks+1),I_r_weekly[j + 10*i], '--o', linewidth=2)
            axs_inf[1,j].plot(np.arange(1,weeks+1),inf_weekly[j + 10*i], '-o', linewidth=2, zorder=1)
            axs_inf[1,j].set_ylim(0,0.2)
            #axs_inf[1,j].set_title('Latent rec = ' + "{:.2f}".format(delta_train_r[10*i + j]))

            axs_inf[2,j].set_ylabel('Infected')
            axs_inf[2,j].plot(t_num,state_r[j + 10*i,:,2], '--', linewidth=4)
            axs_inf[2,j].plot(t_num,state_mod[j + 10*i,:,2], linewidth=4, zorder=1)
            axs_inf[2,j].set_ylim(0,0.03)
            #axs_inf[2,j].set_title('Latent ratio = ' + "{:.2f}".format(delta_train[10*i + j] / delta_train_r[10*i + j]))
        
        fig_inf.tight_layout()
        namefig_inf = 'figure'+str(i)+'.png'
        plt.savefig(folder_train + namefig_inf)
        plt.show()

if plot_testg:
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
    print(state_r[0,:,2].shape)
    print(t_num.squeeze().shape)
    I_interp_r = np.array([np.interp(t.squeeze(), t_num.squeeze(), state_r[k, :,2]) for k in range(n_size)])
    I_r_weekly = np.array([np.sum(I_interp_r[:,length_period*k:length_period*(k+1)], axis = 1) for k in range(weeks)]).T

    for i in range(int(n_size/10)):    
        fig_inf, axs_inf = plt.subplots(3, 10, figsize = (30, 8))#n_size)
        for j  in range(10):
            axs_inf[0,j].plot(t_num,beta_r[j + 10*i,:,0], '--', linewidth=4)
            axs_inf[0,j].plot(t_num,beta_real[j + 10*i,:,0], linewidth=4, zorder=1)
            #axs_inf[0,j].set_ylim(0,0.5)
            axs_inf[0,j].set_ylim(0.5,2.5)
            axs_inf[0,j].set_ylabel('Transmission rate')
            highlight_color = 'lightgreen'  # Specify your desired color
            axs_inf[0,j].axvspan(0, T_obs, facecolor=highlight_color, alpha=0.5)
            #axs_inf[0,j].set_title('Latent train = ' + "{:.2f}".format(delta_train_r[10*i + j]))
            
            axs_inf[1,j].set_ylabel('Weekly incidence')
            axs_inf[1,j].plot(np.arange(1,weeks+1),I_r_weekly[j + 10*i], '--o', linewidth=2)
            axs_inf[1,j].plot(np.arange(1,weeks+1),inf_weekly[j + 10*i], '-o', linewidth=2, zorder=1)
            axs_inf[1,j].axvspan(0, int(T_obs/length_period), facecolor=highlight_color, alpha=0.5)
            axs_inf[1,j].set_ylim(0,0.2)
            #axs_inf[1,j].set_title('Latent testg = ' + "{:.2f}".format(delta_testg_r[10*i + j]))

            axs_inf[2,j].set_ylabel('Infected')
            axs_inf[2,j].plot(t_num,state_r[j + 10*i,:,2], '--', linewidth=4)
            axs_inf[2,j].plot(t_num,state_mod[j + 10*i,:,2], linewidth=4, zorder=1)
            axs_inf[2,j].axvspan(0, T_obs, facecolor=highlight_color, alpha=0.5)
            axs_inf[2,j].set_ylim(0,0.03)
            #axs_inf[2,j].set_title('Latent ratio = ' + "{:.2f}".format(delta_testg_r[10*i + j] / delta_train_r[10*i + j]))
        
        fig_inf.tight_layout()
        namefig_inf = 'figure'+str(i)+'.png'
        plt.savefig(folder_testg + namefig_inf)
        plt.show()
    
    np.savetxt(folder_testg + 'beta_rec_testg.txt', beta_r.squeeze())
    np.savetxt(folder_testg + 'beta_real.txt', beta_real.squeeze())
    np.savetxt(folder_testg + 'delta_testg.txt', delta_testg_r.squeeze())
    np.savetxt(folder_testg + 'S_rec_testg.txt', state_r[:,:,0])
    np.savetxt(folder_testg + 'I_rec_testg.txt', state_r[:,:,1])
    np.savetxt(folder_testg + 'R_rec_testg.txt', state_r[:,:,2])
    np.savetxt(folder_testg + 'S_real.txt', state_mod[:,:,0])
    np.savetxt(folder_testg + 'I_real.txt', state_mod[:,:,1])
    np.savetxt(folder_testg + 'R_real.txt', state_mod[:,:,2])
    np.savetxt(folder_testg + 'Incidence_rec_testg.txt', I_r_weekly.squeeze())
    np.savetxt(folder_testg + 'Incidence_real.txt', inf_weekly.squeeze())
    np.savetxt(folder_testg + 't_num.txt', t_num.squeeze())

