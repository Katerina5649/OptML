import numpy as np
from numpy import array  
from numpy.linalg import norm
import matplotlib.pyplot as plt

def accuracy_check(function_type, x, f_list, grad_f_list, x_star, f_star, mu, W):
#for function type, 0 stands for non-convex, 1 stands for convex and 2 stands for strongly convex
#if function is not stronly convex, input -1 for mu
    (maxiter, params, nodes) = x.shape
    T = maxiter - 1
    x_bar = np.zeros((maxiter, params, nodes))
    for t in range(maxiter):
        for i in range (params):
            x_bar[t][i] = x[t][i].mean(axis = 0)
    
    if function_type == 0:
        temp = [0] * T;
        for t in range(0, T):
            temp2 = 0
            expf = 0
            for grad_f in grad_f_list:
                expf = expf + np.power(norm(grad_f(x_bar[t]), 2), 2)
            expf = expf / len(f_list)
            temp[t] = (t > 0)*temp[t-1]+ expf
            temp[t] = temp[t] / (t + 1)
        print(temp[1])
        
        m = np.linspace(0, T, num = T)
        plt.plot(temp, m)
        plt.show()
            
    elif function_type == 1:
        temp = [0] * T;
        for t in range(0, T):
            temp2 = 0
            expf = 0
            for f in f_list:
                expf = expf + (f(x_bar[t]) - f_star)
            expf = expf / len(f_list)
            temp[t] = (t > 0)*temp[t-1]+ expf
            temp[t] = temp[t] / (t + 1)
        print(temp[1])
        
        m = np.linspace(0, T, num = T)
        plt.plot(temp, m)
        plt.show()
            
    elif function_type == 2:
        temp = [0] * T;
        for t in range(0, T):
            temp2 = 0
            expf = 0
            for f in f_list:
                #(W[t] > 0) * W[t] / W[T] *
                expf = expf + (f(x_bar[t]) - f_star)
            expf = expf / len(f_list)
            temp[t] = (t > 0)*temp[t-1]+ expf + mu * np.power(norm(x_bar[T] - x_star, 2), 2)
            temp[t] = temp[t] / (t + 1)
        
        m = np.linspace(0, T, num = T)
        plt.plot(temp, m)
        plt.show()
        
    else:
        print("Wrong argument.")