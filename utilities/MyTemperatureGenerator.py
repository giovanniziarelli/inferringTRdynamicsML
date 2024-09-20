# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:40:50 2023

@author: alean
"""
import random
import math
import matplotlib.pyplot as plt
import numpy as np
"""
data_dict = {}

with open('data.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            field, value = line.split(':')
            data_dict[field.strip()] = value.strip()
t_max = data_dict['t_max']
"""
def generate_temperature(fun_type, t_max=1.0, T_max=1.0):
    if fun_type == "step":
        t1 = random.uniform(0., t_max)
        def T(t):
            d = min(t1, t_max-t1)
            d=d/5
            if t<t1:
                return T_max
            else:
                return math.exp(-( (t-t1)/d )**2)
        return T
    if fun_type == "exp":
        mean = random.uniform(0, t_max)
        width = t_max/2 + random.uniform(-t_max/4, t_max/4)
        def T(t):
            return math.exp(-( (t-mean)/width )**2)
        return T
    if fun_type == "decr-exp":
        tau = random.gauss(0.5*t_max, 0.3*t_max)
        def T(t):
            return math.exp( -t/tau )
        return T
    if fun_type == "sin":
        periodo = 2*math.pi + random.gauss(0.0, t_max)
        def T(t):
            return math.sin(periodo*t)*T_max
        return T
    if fun_type == "const":
        c = random.uniform(0.0, T_max)
        def T(t):
            return c
        return T
    if fun_type == "boy":
        def rumore(t):
            return math.cos(2*math.pi/0.2*t)*0.1 - 0.1
        def gradino(t):
            return  (math.atan(15*(t-0.3))+math.pi/2)/(math.pi)
        shift = random.uniform(-1.5, 1.5)
        ampiezza = random.uniform(0.7, 1)
        sign_r = random.randint(-1, 1)
        sign_gradino = random.randint(0, 1)
        if sign_r == 0:
            sign_rumore = 0.0
        if sign_r == 1:
            sign_rumore = 1.0
        if sign_r == -1:
            sign_rumore = -1.0
        if sign_gradino == 0:
            def T(t):
                return  ampiezza*(1.2+math.cos(2*math.pi*(shift+7/12-t)))/2.2+sign_rumore*rumore(t)
        if sign_gradino == 1:
            def T(t):
                return  0.5*(ampiezza*(1.2+math.cos(2*math.pi*(shift+7/12-t)))/2.2+sign_rumore*rumore(t)+gradino(t))
        return T
    
    if fun_type == "mixed":
        type_fun = random.randint(0, 11)
        if type_fun == 0:
            return generate_temperature("step", t_max, T_max)
        if type_fun == 1:
            return generate_temperature("exp", t_max, T_max)
        if type_fun == 2:
            return generate_temperature("decr-exp", t_max, T_max)
        if type_fun == 3:
            return generate_temperature("sin", t_max, T_max)
        if type_fun == 4:
            return generate_temperature("const", t_max, T_max)
        if type_fun >= 5:
            return generate_temperature("boy", t_max, T_max)

def generate_temp_by_adri(beta_ref):
    # temperature generate con valori in [temp_min, temp_max]
    # e con tempo in [0, 1]
    tau_noise = random.uniform(10, 15)
    ampiezza_noise = 1 +0.5*random.uniform(0, 1.5)# 0.7 +1/2*random.uniform(0, 1.5)
    mese = random.randint(0, 12)
    ampiezza = random.uniform(0.95, 1.05)
    altezza = random.uniform(-3, 3)
    Tamp = 22
    Tmean = 22*0.93
    def Tcos(t):
        return altezza + ampiezza*(np.cos((t-9-mese)*2*np.pi/6))*Tamp + Tmean + ampiezza_noise*np.cos(10*tau_noise*t)
    def T_return(t):
        #return beta_ref - Tcos(t*12)/6.0 - 3.0
        return (Tcos(t*12) - Tmean)/(Tamp + 0.7)*0.3*beta_ref*2 + beta_ref
    def Betaeq_return(t):
        return (Tmean - Tcos(t*12))/(Tamp + 0.7)*0.3*beta_ref + 0.45*beta_ref
    t = np.linspace(0,12,100)
    #fig,ax = plt.subplots(1,3)
    #ax[0].plot(t, Tcos(t))
    #ax[1].plot(t, T_return(t))
    #ax[2].plot(t, Betaeq_return(t))
    #plt.show()
    return T_return, Betaeq_return

def generate_temp_by_gio(beta_ref, phase_T, f_T, amp_T, T_mean):
    """
    T(t) = 273 cos(f t + \phi)
    \dot{T}(t) = 273 f sin(f t + \phi)
    beta(t) = beta0 + \Delta beta sin(f t + \phi)
    \dot{beta}(t) = - \Delta beta cos(f t + \phi) = - \Delta beta f / 273 T(t)
    A questa aggiungo un'altra periodicità di beta
    \dot{beta}(t) = - sqrt{1 - beta^2(t)}
    """
    #phase_T    = random.uniform(2*np.pi/12, 3*np.pi)#random.uniform(2*np.pi/12, 2*np.pi)
    #f_T        = random.uniform(0, 8*np.pi/22)#random.uniform(6*np.pi/6, 6*np.pi/22)
    #amp_T      = random.uniform(7, 15)#random.uniform(0, 14)
 
    #phase_T    = random.uniform(2.65, 4.20) #random.uniform(2*np.pi/12, 2*np.pi)
    #f_T        = random.uniform(0.02, 0.03) #random.uniform(6*np.pi/20, 2*np.pi)
    #amp_T      = 7#random.uniform(0, 5)#random.uniform(0, 14)
 

    def T(t):
        return T_mean - amp_T * np.sin(2 * np.pi * f_T * t + phase_T)#T_mean + amp_T * np.cos(f_T * t + phase_T)
    
    def trapIntT(t):
        N = 100
        t_nodes = np.linspace(0,t,N+1)
        return t/(N*2) * ((T(0) - T_mean) + (T(t) - T_mean) + 2 * np.sum((T(t_nodes[1:-1])-T_mean)))

    def beta(t):
        return beta_ref /( (1 + np.exp(0.05*trapIntT(t)))) 
    #t = np.linspace(0,12,100)
    #fig,ax = plt.subplots(1,3)
    #ax[0].plot(t, T(t))
    #ax[1].plot(t, beta(t))
    #ax[2].plot(t, trapIntT(t))
    #plt.show()
    return T, beta

def generate_temp_by_gio_old(beta_ref):
    """
    T(t) = 273 cos(f t + \phi)
    \dot{T}(t) = 273 f sin(f t + \phi)
    beta(t) = beta0 + \Delta beta sin(f t + \phi)
    \dot{beta}(t) = - \Delta beta cos(f t + \phi) = - \Delta beta f / 273 T(t)
    A questa aggiungo un'altra periodicità di beta
    \dot{beta}(t) = - sqrt{1 - beta^2(t)}
    """
    #phase_T    = random.uniform(2*np.pi/12, 3*np.pi)#random.uniform(2*np.pi/12, 2*np.pi)
    #f_T        = random.uniform(0, 8*np.pi/22)#random.uniform(6*np.pi/6, 6*np.pi/22)
    #amp_T      = random.uniform(7, 15)#random.uniform(0, 14)
 
    phase_T    = random.uniform(2.65, 4.20) #random.uniform(2*np.pi/12, 2*np.pi)
    f_T        = random.uniform(0.02, 0.03) #random.uniform(6*np.pi/20, 2*np.pi)
    amp_T      = 7#random.uniform(0, 5)#random.uniform(0, 14)
 
    T_mean  = 16

    def T(t):
        return T_mean + amp_T * np.sin(2 * np.pi * f_T * t + phase_T)#T_mean + amp_T * np.cos(f_T * t + phase_T)
    
    def trapIntT(t):
        N = 100
        t_nodes = np.linspace(0,t,N+1)
        return t/(N*2) * ((T(0) - T_mean) + (T(t) - T_mean) + 2 * np.sum((T(t_nodes[1:-1])-T_mean)))

    def beta(t):
        return beta_ref /( (1 + np.exp(0.05*trapIntT(t)))) 
    #t = np.linspace(0,12,100)
    #fig,ax = plt.subplots(1,3)
    #ax[0].plot(t, T(t))
    #ax[1].plot(t, beta(t))
    #ax[2].plot(t, trapIntT(t))
    #plt.show()
    return T, beta



def generate_lockdown_function():
    num_tratti = random.randint(20, 30)
    points = sorted([random.random() for _ in range(num_tratti-1)])
    intervals = [0] + points + [1]
    values = [random.randint(0, 7)/7 for _ in range(num_tratti)]
    
    def f_return(x):
        for i in range(len(intervals) - 1):
            if intervals[i] <= x < intervals[i + 1]:
                return values[i]
    return f_return

    
        
    
