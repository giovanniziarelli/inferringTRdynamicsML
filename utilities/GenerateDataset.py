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
# VENGONO PRESI I DATI DAL FILE
###############################

N = int(data_dict['N'])
K = int(data_dict['K'])
K_val = int(data_dict['K_val'])
K_test = int(data_dict['K_test'])
train_fun_type = 'giovanni-style'
val_fun_type =  'giovanni-style'
test_fun_type =  'giovanni-style'
S0 = float(data_dict['S0'])
S_inf = float(data_dict['S_inf'])
nome_file = data_dict['nome_file']
beta0_inf = float(data_dict['beta0_inf'])
beta0_sup = float(data_dict['beta0_sup'])
t_max = float(data_dict['t_max'])
alpha = float(data_dict['alpha'])
gamma = float(data_dict['gamma'])
n_variabili = int(data_dict['n_variabili'])

b_ref = gamma*math.log(S0/S_inf)/(1-S_inf)
print(b_ref)
T_fin = N #/ 365 * 12
t = np.linspace(0, T_fin, N)

if n_variabili > 1:
    dataset = np.zeros([K, N, n_variabili])
else:
    dataset = np.zeros([K, N])


beta0_train = np.random.uniform(beta0_inf, beta0_sup, (K))
if K == 20:
    beta0_train = np.array([1,1,1,1.33,2,2,1.19, 1.65, 1.29, 1.29, 1.13,\
            0.91, 1.70, 1.7,1.7,1.7,1.7,2,1.77,1.7])
for k in range(K):
    # works only with K == 20 
    phase_T    = np.random.uniform(np.pi / 365 / 6, 3 * np.pi / 365, size = (K))#np.array([2.65, 3.86, 2.72, 2.81, 3.81, 3.81, 2.81, 2.82, 3.84, 3.84, 3.84,\
            #2.65, 3.75, 3.75, 3.75, 3.75, 3.75, 3.84, 3.7, 3.87]) #random.uniform(2*np.pi/12, 2*np.pi)
    f_T        = 1/365#np.random.uniform(0, 1/365, size = (K))#np.array([0.025, 0.018, 0.022, 0.02, 0.028, 0.037, 0.04, 0.042, 0.039, 0.035, 0.05,\
            #0.05, 0.05, 0.035, 0.035, 0.035, 0.025, 0.02, 0.033, 0.05]) 
    amp_T      = np.random.uniform(5, 15, size = (K))#random.uniform(0, 5)#random.uniform(0, 14)V 
    
    T_mean  = np.random.uniform(10,15, size = (K))#16
    
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    elif train_fun_type == 'giovanni-style':# and (K == 20):
        T_new, Betaeqnew = tg.generate_temp_by_gio(b_ref, phase_T[k], f_T, amp_T[k], T_mean[k])
    #elif train_fun_type == 'giovanni-style' and ((K == 20) == False):
    #    T_new, Betaeqnew = tg.generate_temp_by_gio_old(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
    
    lockdown_function = tg.generate_lockdown_function()
    if n_variabili > 1:
        for i in range(N):
            dataset[k, i, 0] = T_new(t[i])
            #dataset[k, i, 1] = lockdown_function(t[i])
            dataset[k, i, 1] = lockdown_function(t[i]/T_fin)#quando uso Giovanni dopo aver aggiornato la temp
    else:
        for i in range(N):
            #print(T_new(t[i]))
            dataset[k, i] = T_new(t[i])
    del T_new, lockdown_function

beta0_val = np.random.uniform(beta0_inf, beta0_sup, (K_val))
if n_variabili > 1:
    val_set = np.zeros([K_val, N, 2])
else:
    val_set = np.zeros([K_val, N])
for k in range(K_val):
    if train_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    elif train_fun_type == 'giovanni-style':
        T_new, Betaeqnew = tg.generate_temp_by_gio_old(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
    
    lockdown_function = tg.generate_lockdown_function()
    if n_variabili > 1:
        for i in range(N):
            val_set[k, i, 0] = T_new(t[i])
            #val_set[k, i, 1] = lockdown_function(t[i])
            val_set[k, i, 1] = lockdown_function(t[i]/T_fin)#quando uso Giovanni dopo aver aggiornato la temp
        plt.plot(val_set[k,:,1])
    else:
        for i in range(N):
            val_set[k, i] = T_new(t[i])
    del T_new, lockdown_function
plt.show()

beta0_test = np.random.uniform(beta0_inf, beta0_sup, (K_test))
if n_variabili > 1:
    test_set = np.zeros([K_test, N, 2])
else:
    test_set = np.zeros([K_test, N])

if K_test == 20:
    beta0_test = np.array([1,1,1,1.33,2,2,1.19, 1.65, 1.29, 1.29, 1.13,\
            0.91, 1.70, 1.7,1.7,1.7,1.7,2,1.77,1.7])
for k in range(K_test):
    # works only with K_test == 20 
    phase_T    = np.random.uniform(np.pi / 365 / 6, 3 * np.pi / 365, size = (K_test))#np.array([2.65, 3.86, 2.72, 2.81, 3.81, 3.81, 2.81, 2.82, 3.84, 3.84, 3.84,\
            #2.65, 3.75, 3.75, 3.75, 3.75, 3.75, 3.84, 3.7, 3.87]) #random.uniform(2*np.pi/12, 2*np.pi)
    f_T        = 1/365#np.random.uniform(0, 1/365, size = (K_test))#np.array([0.025, 0.018, 0.022, 0.02, 0.028, 0.037, 0.04, 0.042, 0.039, 0.035, 0.05,\
            #0.05, 0.05, 0.035, 0.035, 0.035, 0.025, 0.02, 0.033, 0.05]) 
    amp_T      = np.random.uniform(5, 15, size = (K_test))#random.uniform(0, 5)#random.uniform(0, 14)V 
    
    T_mean  = np.random.uniform(10,15, size = (K_test))#16
    
    if test_fun_type == 'adriano-style':
        T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
    elif test_fun_type == 'giovanni-style':# and (K_test == 20):
        T_new, Betaeqnew = tg.generate_temp_by_gio(b_ref, phase_T[k], f_T, amp_T[k], T_mean[k])
    #elif test_fun_type == 'giovanni-style' and ((K_test == 20) == False):
    #    T_new, Betaeqnew = tg.generate_temp_by_gio_old(b_ref)
    else:
        T_new = tg.generate_temperature(val_fun_type)
    
    lockdown_function = tg.generate_lockdown_function()
    if n_variabili > 1:
        for i in range(N):
            test_set[k, i, 0] = T_new(t[i])
            #test_set[k, i, 1] = lockdown_function(t[i])
            test_set[k, i, 1] = lockdown_function(t[i]/T_fin)#quando uso Giovanni dopo aver aggiornato la temp
    else:
        for i in range(N):
            #print(T_new(t[i]))
            test_set[k, i] = T_new(t[i])
    del T_new, lockdown_function

### beta0_test = np.random.uniform(beta0_inf, beta0_sup, (K_test))
### for k in range(K_test):
###     if train_fun_type == 'adriano-style':
###         T_new, Betaeqnew = tg.generate_temp_by_adri(b_ref)
###     elif train_fun_type == 'giovanni-style':
###         T_new, Betaeqnew = tg.generate_temp_by_gio_old(b_ref)
###     else:
###         T_new = tg.generate_temperature(val_fun_type)
###     
###     lockdown_function = tg.generate_lockdown_function()

###     if n_variabili > 1:
###         for i in range(N):
###             test_set[k, i, 0] = T_new(t[i])
###             #test_set[k, i, 1] = lockdown_function(t[i])
###             test_set[k, i, 1] = lockdown_function(t[i]/T_fin)#quando uso Giovanni dopo aver aggiornato la temp
###     else:
###         for i in range(N):
###             test_set[k, i] = T_new(t[i])
###     del T_new

with open('datasets/'+nome_file, 'wb') as file:
    pickle.dump((dataset, val_set, test_set, beta0_train, beta0_val, beta0_test, T_mean, amp_T, phase_T, f_T), file)
    
"""
n_input = 2
n_hidden = 15
n_output = 1
weights = {
'h1': tf.Variable(tf.random.normal([n_input, n_hidden], dtype='float64'), dtype='float64'),
'out': tf.Variable(tf.random.normal([n_hidden, n_output], dtype='float64'), dtype='float64')
}
biases = {
'b1': tf.Variable(tf.random.normal([n_hidden], dtype='float64'), dtype='float64'),
'out': tf.Variable(tf.random.normal([n_output], dtype='float64'), dtype='float64')
}

def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    output = tf.matmul(layer_1, weights['out']) +biases['out']
    return output

def g(y, v):
    v.shape = (v.shape[0], n_input-1)
    tv = tf.constant(v, dtype='float64')
    x = tf.concat([y, tv], 1)
    return multilayer_perceptron(x)

 
"""
