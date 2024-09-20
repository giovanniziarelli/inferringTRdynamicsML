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
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# We configure TensorFlow to work in double precision 
tf.keras.backend.set_floatx('float64')

import utils
import optimization

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
n_timesteps = 197
n_features =2 

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
trial_sim = 0 

num_latent_states = 1
num_latent_params = 1

# Loading datasets
input_dataset_path_temp = '/home/giovanni/Desktop/LDNets/italian-temperatures/tmedia_national_length'+str(T_fin)+'.csv'#temperature_n.pkl'
input_dataset_extern_temp = np.loadtxt(input_dataset_path_temp)
input_dataset_path_umid = '/home/giovanni/Desktop/LDNets/italian-temperatures/umid_national_length'+str(T_fin)+'.csv'#temperature_n.pkl'
input_dataset_extern_umid = np.loadtxt(input_dataset_path_umid)
output_dataset_path = '/home/giovanni/Desktop/LDNets/influnet/data-aggregated/epidemiological_data/processed_output_new_cases_'+str(N_weeks)+'_weeks.csv'#temperature_n.pkl'
output_dataset_extern = np.loadtxt(output_dataset_path)
y = output_dataset_extern.copy()

training_var_numpy_orig = np.stack((input_dataset_extern_temp, input_dataset_extern_umid), axis = 2) #T_cos(T_mean, f, A, t)
undetection_mean_value  = 0.23
cases                   = output_dataset_extern / undetection_mean_value

X_reshaped = training_var_numpy_orig.reshape(training_var_numpy_orig.shape[0], -1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
print(X_scaled.shape)
print(y_scaled.shape)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0, random_state=42)
X_train = X_scaled
y_train = y_scaled
# Fit LASSO model
lasso = MultiTaskLasso(alpha=0.1)
lasso.fit(X_train, y_train)
# Get the coefficients
coefficients = lasso.coef_
print(coefficients.shape)
# Reshape coefficients back to the original (n_timesteps, n_features) shape
#coefficients_reshaped = coefficients.reshape(n_timesteps, n_features)
n_timesteps_y = 28
coefficients_reshaped = coefficients.reshape(n_timesteps_y, n_features, n_timesteps)
# Identify selected features (those with non-zero coefficients)
selected_features = np.where(coefficients_reshaped != 0)

print("Selected features indices (time, feature):", list(zip(*selected_features)))
print("Coefficients of selected features:", coefficients_reshaped[selected_features])
sample_importance = np.sum(np.abs(coefficients), axis=1)

# Rank samples by importance
sorted_indices = np.argsort(sample_importance)[::-1]  # Sort in descending order

# Select the most informative samples
top_n = 10  # Number of top informative samples to select
informative_samples = sorted_indices#[:top_n]

print("Most informative sample indices:", informative_samples)
y_pred_train = lasso.predict(X_train)

# Calculate residuals (difference between actual and predicted values)
residuals = np.sum((y_train - y_pred_train) ** 2, axis=1)  # Sum over the timesteps

# Rank samples by residuals (low residuals indicate well-fitted samples)
sorted_indices = np.argsort(residuals)  # Sort in ascending order (low residuals first)

# Select the most informative samples (i.e., those with the lowest residuals)
top_n = 10  # Number of top informative samples to select
informative_samples = sorted_indices[:top_n]

print("Most informative sample indices:", informative_samples)
print("Corresponding residuals:", residuals[sorted_indices])
