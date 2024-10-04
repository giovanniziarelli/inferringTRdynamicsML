import math
import random
import sys
sys.path.append('utilities')
import MyTemperatureGenerator as tg
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

data_dict = {}

with open('utilities/data-for-GenerateDataset.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            field, value = line.split(':')
            data_dict[field.strip()] = value.strip()

###############################
# Extracting data from file 
###############################

N              = int(data_dict['N'])             # days
K              = int(data_dict['K'])             # amount of training samples
K_val          = int(data_dict['K_val'])         # amount of validation samples
K_test         = int(data_dict['K_test'])        # amount of test samples
train_fun_type = 'style1'                        # function time for generating training data
val_fun_type   = 'style1'                        # function time for generating validation data
test_fun_type  = 'style1'                        # function time for generating testing data
S0             = float(data_dict['S0'])          # initial susceptible value
S_inf          = float(data_dict['S_inf'])       # target final level of susceptibles
file_name      = data_dict['file_name']          # file name 
beta0_inf      = float(data_dict['beta0_inf'])   # lb for transmission rate
beta0_sup      = float(data_dict['beta0_sup'])   # ub for transmission rate
t_max          = float(data_dict['t_max'])       # final time
alpha          = float(data_dict['alpha'])       # incubation rate
gamma          = float(data_dict['gamma'])       # recovery rate
n_variables    = int(data_dict['n_variables'])   # number of variables

# computing reference \beta
b_ref = gamma * math.log(S0/S_inf) / (1-S_inf)
T_fin = N 
t = np.linspace(0, T_fin, N)

if n_variables > 1:
    dataset = np.zeros([K, N, n_variables])
else:
    dataset = np.zeros([K, N])

# intial training \beta0
beta0_train = np.random.uniform(beta0_inf, beta0_sup, (K))

# generating training inputs
for k in range(K):
    
    phase_T    = np.random.uniform(np.pi / 365 / 6, 3 * np.pi / 365, size = (K))  # phase of the sinusoidal wave of temperature
    f_T        = 1/365                                                            # frequency of the sinusoidal wave of temperature
    amp_T      = np.random.uniform(5, 15, size = (K))                             # amplitude of the sinusoidal wave of temperature
    
    T_mean  = np.random.uniform(10,15, size = (K))                                # mean temperature
    
    if train_fun_type == 'style0':
        T_new, Betaeqnew = tg.generate_temp_style0(b_ref)
    elif train_fun_type == 'style1':# and (K == 20):
        T_new, Betaeqnew = tg.generate_temp_style1(b_ref, phase_T[k], f_T, amp_T[k], T_mean[k])
    else:
        T_new = tg.generate_temperature(val_fun_type)
    
    lockdown_function = tg.generate_lockdown_function()
    if n_variables > 1:
        for i in range(N):
            dataset[k, i, 0] = T_new(t[i])
            dataset[k, i, 1] = lockdown_function(t[i]/T_fin) #WIP
    else:
        for i in range(N):
            dataset[k, i] = T_new(t[i])
    del T_new, lockdown_function

# initial validation \beta0
beta0_val = np.random.uniform(beta0_inf, beta0_sup, (K_val))

if n_variables > 1:
    val_set = np.zeros([K_val, N, 2])
else:
    val_set = np.zeros([K_val, N])

# generating validation inputs
for k in range(K_val):
    if train_fun_type == 'style0':
        T_new, Betaeqnew = tg.generate_temp_style0(b_ref)
    elif train_fun_type == 'style1':
        T_new, Betaeqnew = tg.generate_temp_style1(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
    
    lockdown_function = tg.generate_lockdown_function()
    if n_variables > 1:
        for i in range(N):
            val_set[k, i, 0] = T_new(t[i])
            val_set[k, i, 1] = lockdown_function(t[i]/T_fin) #WIP
    else:
        for i in range(N):
            val_set[k, i] = T_new(t[i])
    del T_new, lockdown_function

# initial testing \beta0
beta0_test = np.random.uniform(beta0_inf, beta0_sup, (K_test))

if n_variables > 1:
    test_set = np.zeros([K_test, N, 2])
else:
    test_set = np.zeros([K_test, N])

# generating testing inputs
for k in range(K_test):
    
    phase_T    = np.random.uniform(np.pi / 365 / 6, 3 * np.pi / 365, size = (K))  # phase of the sinusoidal wave of temperature
    f_T        = 1/365                                                            # frequency of the sinusoidal wave of temperature
    amp_T      = np.random.uniform(5, 15, size = (K))                             # amplitude of the sinusoidal wave of temperature
    
    T_mean  = np.random.uniform(10,15, size = (K))                                # mean temperature
    
    if test_fun_type == 'style0':
        T_new, Betaeqnew = tg.generate_temp_style0(b_ref)
    elif test_fun_type == 'style1':
        T_new, Betaeqnew = tg.generate_temp_style1(b_ref, phase_T[k], f_T, amp_T[k], T_mean[k])
    else:
        T_new = tg.generate_temperature(val_fun_type)
    
    lockdown_function = tg.generate_lockdown_function()
    if n_variables > 1:
        for i in range(N):
            test_set[k, i, 0] = T_new(t[i])
            test_set[k, i, 1] = lockdown_function(t[i]/T_fin) # WIP
    else:
        for i in range(N):
            test_set[k, i] = T_new(t[i])
    del T_new, lockdown_function


with open('datasets/'+file_name, 'wb') as file:
    pickle.dump((dataset, val_set, test_set, beta0_train, beta0_val, beta0_test, T_mean, amp_T, phase_T, f_T), file)
    
