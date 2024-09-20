# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 22:33:37 2023

@author: alean
"""


def compute_jacobian(sys,x, h):
    import numpy as np
    n=len(x)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_plus_h=x.copy()
            x_plus_h[j]=x[j]+h
            x_minus_h=x.copy()
            x_minus_h[j]=x[j]-h
            J[i, j] = 0.5*( sys(x_plus_h) - sys(x_minus_h) )[i]/h
    del np
    return J



def compute_sys(sys, x, t):
    import numpy as np
    y=np.zeros(len(x))
    for i in range(len(x)):
        y[i]=sys[i](x, t)
    del np 
    return y

def my_newton(sys, x0, max_it=15, toll=1e-6, h=1e-2, step=1e-2):
    import numpy as np
    J = compute_jacobian(sys, x0, step)
    current_iter=0
    current_x=x0.copy()
    next_x=x0.copy()
    while current_iter<max_it and np.linalg.norm(sys(current_x))>toll:
        b=sys(current_x)
        y=np.linalg.solve(J, b)
        next_x=current_x-y
        current_x=next_x.copy()
        current_iter=current_iter+1
    del np
    return next_x.copy()
    
    
    
    
