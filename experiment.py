import numpy as np
from oracle import Oracle
from accuracy_test import accuracy_check

def decentralizedSGD(x0, lr, max_iter, W, oracle):
    
    x_t = x0
    (params, nodes) = x0.shape
    x_store = np.zeros((max_iter, params, nodes))

    for t in range(max_iter):
         
        noise = np.random.normal(0, 1, size=x_t.shape)
        
        f, df = oracle(x_t + noise)
        
        y_t = x_t - lr * df
        x_t = np.matmul(y_t, W)
        x_store[t] = x_t
        
    return x_store

def MetropolisHastings(W):
    
    degrees = np.sum(W, axis=1)
    
    for i in range(W.shape[0]):
        for j in range(i, W.shape[0]):
            
            if W[i, j] != 0:
                
                weight = min(1 / (degrees[i] + 1), 1 / (degrees[j] + 1))
                
                W[i, j] = weight
                W[j, i] = weight
                
    return W


def toMatrix(size, adj_list):
    
    matrix = np.zeros((size, size))
    
    for node in adj_list:
        
        neighbors = adj_list[node]
        
        for neighbor in neighbors:
            
            matrix[node, neighbor] = 1
            matrix[neighbor, node] = 1
            
    return matrix.astype(np.float64)
nodes = 4
params = 6
shape = (params, nodes)
max_iter = 100
lr = 1e-4

x0 = np.random.randn(params, nodes)

connections = {0: [1, 2], 1: [3], 2: [1]}

adjacency_matrix = toMatrix(nodes, connections)

W = MetropolisHastings(adjacency_matrix)

func_type = "strongly convex"
oracle = Oracle(func_type)

x_store = decentralizedSGD(x0, lr, max_iter, W, oracle)


def f1(x):
    return np.linalg.norm(np.power(x, 2))
def g1(x):
    return 2*x

function_list = [f1]
gradient_list = [g1]
x_star = np.zeros((params, nodes))
f_star = f1(x_star)
accuracy_check(0, x_store, function_list, gradient_list, x_star, f_star, -1, W)