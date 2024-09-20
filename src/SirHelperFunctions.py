# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:30:45 2023

@author: alean
"""
import tensorflow as tf
import numpy as np

def forward_euler_step(curr_S_nn, curr_I_nn, curr_beta, dt, a):
    next_S_nn = curr_S_nn - dt*tf.matmul(curr_beta, tf.matmul(curr_S_nn, curr_I_nn))
    next_I_nn = curr_I_nn + dt*(tf.matmul(curr_beta, tf.matmul(curr_S_nn, curr_I_nn))) - dt*a*curr_I_nn
    return next_S_nn, next_I_nn
"""
next_S_nn = curr_S_nn - dt*tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))
next_I_nn = curr_I_nn + dt*(tf.matmul(curr_y, tf.matmul(curr_S_nn, curr_I_nn))) - dt*a*curr_I_nn
"""

def runge_kutta_step(curr_s, curr_i, curr_beta, next_beta, dt, a):
    t=0.
    s = curr_s
    i = curr_i
    m1 = -tf.matmul(curr_beta, tf.matmul(s, i))
    k1 = tf.matmul(curr_beta, tf.matmul(s, i)) - a*i
    l1 = a*i
    #def m2,k2,l2
    ft2 = t + (dt/2.)
    fs2 =  s + (dt/2.)*m1
    fi2 =  i + (dt/2.)*k1
    b = 0.5*( tf.add(curr_beta, next_beta) )
    m2 = -tf.matmul(b, tf.matmul(fs2, fi2))
    k2 = tf.matmul(b, tf.matmul(fs2, fi2)) - a*fi2
    l2 = a*fi2
    #def m3,k3,l3
    ft3 = t + (dt/2.)
    fs3 =  tf.add(s, (dt/2.)*m2)
    fi3 =  tf.add(i, (dt/2.)*k2)
    b = 0.5*(tf.add(curr_beta, next_beta))
    m3 = -tf.matmul(b, tf.matmul(fs3, fi3)) # -b*fs3*fi3
    k3 =  tf.matmul(b, tf.matmul(fs3, fi3)) - a*fi3 # b*fs3*fi3 - a*fi3
    l3 = a*fi3
    #def m4,k4,l4
    ft4 = t + dt
    fs4 =  tf.add(s, dt*m3)
    fi4 =  tf.add(i, dt*k3)
    b = next_beta
    m4 = -tf.matmul(b, tf.matmul(fs4, fi4))# -b*fs4*fi4
    k4 =  tf.matmul(b, tf.matmul(fs4, fi4)) - a*fi4 # b*fs4*fi4 - a*fi4
    l4 = a*fi4
    t = t + dt
    next_s = s + (dt/6.)*(m1 + 2.*m2 +2.*m3 + m4)
    next_i = i + (dt/6.)*(k1 + 2.*k2 +2.*k3 + k4)
    return next_s, next_i

def compute_I(betas, t_max, alpha, sir_0):
    Is = []
    S0, I0, R0 = sir_0
    for beta_vec in betas:
        N = beta_vec.shape[1]
        K = beta_vec.shape[0]
        I = np.zeros([K, N])
        I[:, 0] = I0
        S_old = np.zeros(K) 
        S_new = np.zeros(K)
        S_old[:] = S0
        dt = 1.0/N
        for i in range(N-1):
            S_new = S_old - t_max * dt * beta_vec[:, i] * S_old * I[:, i]
            I[:, i+1] = I[:, i] + t_max*dt*( beta_vec[:, i]*S_old*I[:, i] - alpha*I[:, i] )
            S_old = S_new
        Is.append(I.copy())
    return Is
    
def compute_beta_napde(datasets, beta0s, t_max, tau, b_ref):
    if len(datasets[0].shape) == 2:
        def f(b, T):
            return (1.0/tau)*(2*b_ref - T - b)
        real_beta = []
        J = len(datasets)
        for j in range(J):
            temps = datasets[j]
            beta0 = beta0s[j]
            beta0.shape = (beta0.shape[0])
            N = temps.shape[1]
            K = temps.shape[0]
            beta = np.zeros([K, N])
            dt = 1.0/N
            beta[:, 0] = beta0
            for i in range(N-1):
                # beta[:, i+1] = beta[:, i] + dt*t_max*f(beta[:, i], temps[:, i])
                
                k1 = t_max*dt * f(beta[:, i], temps[:, i])
                k2 = t_max*dt * f( beta[:, i] + k1/2, 0.5*(temps[:, i]+temps[:, i+1]) )
                k3 = t_max*dt * f(beta[:, i] + k2/2, 0.5*(temps[:, i]+temps[:, i+1]) )
                k4 = t_max*dt * f(beta[:, i] + k3, temps[:, i+1])
                beta[:, i+1] = beta[:, i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
                
                
            real_beta.append(beta.copy())
        return real_beta
    
    if len(datasets[0].shape) == 3 and datasets[0].shape[2] == 2:
        # caso con in input temperatura e livello di isolamento
        def f(b, T, iso):
            return (1.0/tau)*( (2*b_ref - T)*(1-0.75*iso) - b )
        real_beta = []
        J = len(datasets)
        for j in range(J):
            temps = datasets[j][:, :, 0]
            isos = datasets[j][:, :, 1]
            beta0 = beta0s[j]
            beta0.shape = (beta0.shape[0])
            N = temps.shape[1]
            K = temps.shape[0]
            beta = np.zeros([K, N])
            dt = 1.0/N
            beta[:, 0] = beta0
            for i in range(N-1):
                beta[:, i+1] = beta[:, i] + dt*t_max*f(beta[:, i], temps[:, i], isos[:, i])
            real_beta.append(beta.copy())
        return real_beta

def compute_beta(datasets, beta0s, t_max, tau, b_ref):
    T_mean = 16 # in utilities/MyTemp
    if len(datasets[0].shape) == 2:
        def f(b, T):
            return 0.05 * (T - T_mean) * (b**2 / b_ref - b)
        real_beta = []
        J = len(datasets)
        for j in range(J):
            temps = datasets[j]
            beta0 = beta0s[j]
            beta0.shape = (beta0.shape[0])
            N = temps.shape[1]
            K = temps.shape[0]
            beta = np.zeros([K, N])
            dt = 1.0/N
            beta[:, 0] = beta0
            for i in range(N-1):
                #beta[:, i+1] = beta[:, i] + dt*t_max*f(beta[:, i], temps[:, i])
                
                k1 = t_max*dt * f(beta[:, i], temps[:, i])
                k2 = t_max*dt * f( beta[:, i] + k1/2, 0.5*(temps[:, i]+temps[:, i+1]) )
                k3 = t_max*dt * f(beta[:, i] + k2/2, 0.5*(temps[:, i]+temps[:, i+1]) )
                k4 = t_max*dt * f(beta[:, i] + k3, temps[:, i+1])
                beta[:, i+1] = beta[:, i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
                
                
            real_beta.append(beta.copy())
        return real_beta
    if len(datasets[0].shape) == 3 and datasets[0].shape[2] == 2:
        # caso con in input temperatura e livello di isolamento
        def f(b, T, iso):
            return 0.5 * (1-0.5*iso)* (T - T_mean) * (b**2 / b_ref - b)
        real_beta = []
        J = len(datasets)
        for j in range(J):
            temps = datasets[j][:, :, 0]
            isos = datasets[j][:, :, 1]
            beta0 = beta0s[j]
            beta0.shape = (beta0.shape[0])
            N = temps.shape[1]
            K = temps.shape[0]
            beta = np.zeros([K, N])
            dt = 1.0/N
            beta[:, 0] = beta0
            for i in range(N-1):
                beta[:, i+1] = beta[:, i] + dt*t_max*f(beta[:, i], temps[:, i], isos[:, i])
            print(real_beta)
            real_beta.append(beta.copy())
        return real_beta
    
def compute_beta_delta(datasets, beta0s, t_max, tau, b_ref, T_mean_arr):
    #T_mean = 16 # in utilities/MyTemp

    if len(datasets[0].shape) == 2:
        def f(b, T, delta, T_m):
            return (T/T_m - 1) * ((b)**2 / b_ref/delta - (b))
        real_beta = []
        delta     = []
        np.random.seed(0)
        J = len(datasets)
        for j in range(J):
            temps = datasets[j]
            beta0 = beta0s[j]
            T_mean = T_mean_arr[j]
            beta0.shape = (beta0.shape[0])
            #delta_j = 0.05 + 0.01 * np.random.randint(10, size = beta0.shape)#np.random.uniform(0.01, 0.01, beta0.shape)
            if len(beta0) != 20:
                delta_j = np.random.uniform(0.1, 3, beta0.shape)#np.linspace(0.5, 2, beta0.shape[0])#np.random.uniform(0.1, 3, beta0.shape) #np.random.uniform(0.05, 0.1, beta0.shape)
            else:
                delta_j = np.array([2.7, 2.75, 2.8, 2.36, 3.6, 3.7, 4.07, 4.26, 2.6, 2.65, 0.35,\
                        0.1, 0.55, 0.65, 0.31, 0.4, 0.5, 0.54, 2.61, 1.06])
            N = temps.shape[1]
            K = temps.shape[0]
            beta = np.zeros([K, N])
            dt = 1.0/N
            beta[:, 0] = beta0
            t_max = N
            for i in range(N-1):
                #beta[:, i+1] = beta[:, i] + dt*t_max*f(beta[:, i], temps[:, i])
                
                k1 = t_max*dt * f(beta[:, i], temps[:, i], delta_j, T_mean)
                k2 = t_max*dt * f( beta[:, i] + k1/2, 0.5*(temps[:, i]+temps[:, i+1]), delta_j, T_mean)
                k3 = t_max*dt * f(beta[:, i] + k2/2, 0.5*(temps[:, i]+temps[:, i+1]), delta_j, T_mean)
                k4 = t_max*dt * f(beta[:, i] + k3, temps[:, i+1], delta_j, T_mean)
                beta[:, i+1] = beta[:, i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
                
                
            real_beta.append(beta.copy())
            delta.append(delta_j.copy())
        return real_beta, delta
    if len(datasets[0].shape) == 3 and datasets[0].shape[2] == 2:
        # caso con in input temperatura e livello di isolamento
        def f(b, T, iso):
            return 0.5 * (1-0.5*iso)* (T - T_mean) * (b**2 / b_ref - b)
        real_beta = []
        J = len(datasets)
        for j in range(J):
            temps = datasets[j][:, :, 0]
            isos = datasets[j][:, :, 1]
            beta0 = beta0s[j]
            beta0.shape = (beta0.shape[0])
            N = temps.shape[1]
            K = temps.shape[0]
            beta = np.zeros([K, N])
            dt = 1.0/N
            beta[:, 0] = beta0
            for i in range(N-1):
                beta[:, i+1] = beta[:, i] + dt*t_max*f(beta[:, i], temps[:, i], isos[:, i])
            print(real_beta)
            real_beta.append(beta.copy())
        return real_beta
    
    
    
    
    
    
