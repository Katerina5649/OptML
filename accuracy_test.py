import numpy as np
from numpy import array  
from numpy.linalg import norm
import matplotlib.pyplot as plt


def plotError(T, temp):
    
    m = np.linspace(0, T, num = T)
    plt.semilogy(m, temp)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()
    
    
def accuracy_check(function_type, x, oracle, x_star, f_star, mu, W):
#for function type, 0 stands for non-convex, 1 stands for convex and 2 stands for strongly convex
#if function is not stronly convex, input -1 for mu
    (maxiter, params, nodes) = x.shape
    T = maxiter - 1
    x_bar = np.zeros((maxiter, params, nodes))
    for t in range(maxiter):
        for i in range (params):
            x_bar[t][i] = x[t][i].mean(axis = 0)
    
    if function_type == "non convex":
        temp = [0] * T;
        for t in range(0, T):
            temp2 = 0
            expf = 0
            _, grad_f = oracle(x_bar[t])
            expf = expf + np.power(norm(grad_f, 2), 2)
            temp[t] = (t > 0)*temp[t-1]+ expf
            temp[t] = temp[t] / (t + 1)
        
        plotError(T, temp)
            
    elif function_type == "convex":
        temp = [0] * T;
        for t in range(0, T):
            temp2 = 0
            expf = 0
            f, _ = oracle(x_bar[t])
            expf = expf + (f - f_star)
            temp[t] = (t > 0)*temp[t-1]+ expf
            temp[t] = temp[t] / (t + 1)
        
        plotError(T, temp)
            
    elif function_type == "strongly convex":
        temp = [0] * T;
        for t in range(0, T):
            temp2 = 0
            expf = 0
            f, _ = oracle(x_bar[t])
                #(W[t] > 0) * W[t] / W[T] *
            expf = expf + (f - f_star)
            temp[t] = (t > 0)*temp[t-1]+ expf + mu * np.power(norm(x_bar[T] - x_star, 2), 2)
            temp[t] = temp[t] / (t + 1)
        
        plotError(T, temp)
        
    else:
        print("Wrong argument.")