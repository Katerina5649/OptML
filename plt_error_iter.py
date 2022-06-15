import random
import numpy as np
import matplotlib.pyplot as plt
from oracle import Oracle
from experiment import decentralizedSGD, MetropolisHastings, toMatrix, buildTopology

def error_vs_iter_func(n_nodes, n_params, topology, lr, max_iter, threshold):
    
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    x0 = np.random.randn(n_params, n_nodes)
    W = buildTopology(n_nodes, topology)
    oracle = Oracle('strongly convex', n_params, n_nodes)
    itr1, errors1 = decentralizedSGD(x0, lr, max_iter, threshold, W, oracle)
    
    x1 = np.linspace(0, itr1, num = len(errors1))
    ax.plot(x1, errors1, label = 'strongly convex')
    
    x0 = np.random.randn(n_params, n_nodes)
    W = buildTopology(n_nodes, topology)
    oracle = Oracle("convex", n_params, n_nodes)
    itr2, errors2 = decentralizedSGD(x0, lr, max_iter, threshold, W, oracle)
    
    x2 = np.linspace(0, itr2, num = len(errors2))
    ax.plot(x2, errors2, label = 'convex')
    
    ax.axhline(y = 1e-5, color = 'r', linestyle = 'dashed')
    
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('Error')
    ax.legend()
    
    plt.show()

def error_vs_iter_topology(n_nodes, n_params, func_type, lr, max_iter, threshold):
    
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    x0 = np.random.randn(n_params, n_nodes)
    W = buildTopology(n_nodes, "dense")
    oracle = Oracle(func_type, n_params, n_nodes)
    itr1, errors1 = decentralizedSGD(x0, lr, max_iter, threshold, W, oracle)
    
    x1 = np.linspace(0, itr1, num = len(errors1))
    ax.plot(x1, errors1, label = 'dense')
    
    x0 = np.random.randn(n_params, n_nodes)
    W = buildTopology(n_nodes, "centralized")
    oracle = Oracle(func_type, n_params, n_nodes)
    itr2, errors2 = decentralizedSGD(x0, lr, max_iter, threshold, W, oracle)
    x2 = np.linspace(0, itr2, num = len(errors2))
    
    if(itr2 <= 2):
        ax.vlines(1, ymin = 0, ymax = 0.5, label = 'centralized', color = 'blue')
    else:
        ax.plot(x2, errors2, label = 'centralized', color = 'blue')
    
    x0 = np.random.randn(n_params, n_nodes)
    W = buildTopology(n_nodes, "ring")
    oracle = Oracle(func_type, n_params, n_nodes)
    itr3, errors3 = decentralizedSGD(x0, lr, max_iter, threshold, W, oracle)
    x3 = np.linspace(0, itr3, num = len(errors3))
    
    ax.plot(x3, errors3, label = 'ring')
    
    ax.axhline(y = 1e-5, color = 'r', linestyle = 'dashed')
    
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('Error')
    ax.legend()
    
    plt.show()
'''
n_params = 10
func_type = "strongly convex"
max_iter = 10000
lr = 1e-4
threshold = 0.5e-5

num_nodes = random.randint(10,50)
error_vs_iter_topology(num_nodes, n_params, func_type, lr, max_iter, threshold)
'''
n_params = 10
topology = "ring"
max_iter = 10000
lr = 1e-4
threshold = 0.5e-5

num_nodes = random.randint(10,50)
error_vs_iter_func(num_nodes, n_params, topology, lr, max_iter, threshold)