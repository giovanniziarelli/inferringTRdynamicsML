from srcs import generate_beta_arrays as gba
from srcs import generate_temp_arrays as gta
from srcs import generate_zone_arrays as gza
import numpy as np
import pickle
import math

# DATI DA SCEGLIERE
"""
. n_timesteps = 
. nome_file = 
.
"""

n_timesteps = 100
path_i = "../dati-regioni"
regione_beta = ['Lombardia']
K = 40
n_giorni = 15
overlap = 3
path_t = "../prova-estrazione-temp/Temperature"
regione_temp = ['Milano']
nome_file = "Prova_Milano_zone_nuove_fun.pkl"
K_train = math.floor(K*0.8) # il resto è messo nel validation set

use_zone = True

# vengono estratti i dati
beta, infetti, date = gba.generate_beta_arrays_2(path_i, K, n_giorni, overlap, regions = regione_beta)
temp = gta.generate_temp_arrays_2(path_t, K, n_giorni, overlap, regions = regione_temp)

print(beta.shape)
print(temp.shape)

temp_train = temp[0:K_train, :]
beta_train = beta[0:K_train, :]
infetti_train = infetti[0:K_train, :]
date_train = date[0:K_train]

K_val = temp.shape[0] - K_train
temp_val = temp[K_train:, :]
beta_val = beta[K_train:, :]
infetti_val = infetti[K_train:, :]
date_val = date[K_train:]
print(infetti_val.shape)

if not use_zone:
    with open('datasets/'+nome_file, 'wb') as file:
        pickle.dump((infetti_train, infetti_val, temp_train, temp_val, beta_train, beta_val, n_giorni, date_train, date_val), file)
else:
    zone = gza.generate_zone_arrays_2(K, n_giorni, overlap, regions = regione_beta)
    zone_train = np.expand_dims(zone[0:K_train, :], axis = -1)
    zone_val = np.expand_dims(zone[K_train:, :], axis = -1)
    temp_train = np.expand_dims(temp_train, axis = -1)
    temp_val = np.expand_dims(temp_val, axis = -1)
    train_set = np.concatenate((temp_train, zone_train), axis = -1)
    val_set = np.concatenate((temp_val, zone_val), axis = -1)
    with open('datasets/'+nome_file, 'wb') as file:
        pickle.dump((infetti_train, infetti_val, train_set, val_set, beta_train, beta_val, n_giorni, date_train, date_val), file)
