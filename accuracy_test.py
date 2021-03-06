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
    
    
def accuracy_check(function_type, x, oracle, x_star, f_star, mu, W, plot=True):
#for function type, 0 stands for non-convex, 1 stands for convex and 2 stands for strongly convex
#if function is not stronly convex, input -1 for mu
    (maxiter, params, nodes) = x.shape
    T = maxiter - 1
    x_bar = np.zeros((maxiter, params, 1))
    for t in range(maxiter):
        for i in range (params):
            x_bar[t][i] = x[t][i].mean(axis = 0)
            
    if function_type == "non convex":
        temp = [0] * T;
        for t in range(0, T):
            
            _, grad_f = oracle(x_bar[t])
            
            expf = np.power(norm(grad_f, 2), 2)
            temp[t] = (t > 0)*temp[t-1]+ expf
            temp[t] = temp[t] / (t + 1)
        
        if plot:
            plotError(T, temp)
        
        return temp
            
    elif function_type == "convex":
        temp = [0] * T;
        for t in range(0, T):
            temp2 = 0
            expf = 0
            f, _ = oracle(x_bar[t])
            expf = expf + (f[0] - f_star)
            temp[t] = (t > 0)*temp[t-1]+ expf
            temp[t] = temp[t] / (t + 1)

        if plot:
            plotError(T, temp)
        
        return temp
            
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
        
        if plot:
            plotError(T, temp)
        
        return temp
        
    else:
        print("Wrong argument.")
        
def accuracy_check_iter(x, current_sum, t, oracle, x_T=None):
    
    x_star, f_star = oracle.getMin()
    mu = oracle.getMu()
    function_type = oracle.getType()
    
    x_bar = np.expand_dims(np.mean(x, axis=1), axis=1)
    
    if function_type == "non convex":
        
        _, grad_f = oracle(x_bar)
        current_sum += np.power(norm(grad_f, 2), 2)
        error = current_sum / (t + 1)
        
        return error, current_sum
        
            
    elif function_type == "convex":
        
        f, _ = oracle(x_bar)
        current_sum += (f[0] - f_star)
        error = current_sum / (t + 1)
        
        return error, current_sum
            
    elif function_type == "strongly convex":
        
        f, _ = oracle(x_bar)
        
        current_sum += (f[0] - f_star)
        x_bar_T = np.expand_dims(np.mean(x_T, axis=1), axis=1)
        additional_term = mu * np.power(norm(x_bar_T - x_star, 2), 2)
        
        error = current_sum / (t + 1) + additional_term
        
        return error, current_sum
        
    else:
        print("Wrong argument.")
        
        
def accuracy_check_err(x, x_star):
    
    difference = x - x_star
    norm_squared = np.power(np.linalg.norm(difference, ord=2, axis=0), 2)
    
    return 1 / x.shape[1] * np.sum(norm_squared)