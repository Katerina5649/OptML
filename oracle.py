import numpy as np

class Oracle:

    def __init__(self, func_type, n_params, n_nodes):
        
        self.type = func_type
        self.n_params = n_params
        self.n_nodes = n_nodes
        
    
    def getOracle(self, x):

        if self.type == "strongly convex":
            return np.power(np.linalg.norm(x), 2), 2 * x
        
    def getMin(self):
        
        if self.type == "strongly convex":
            return np.zeros((self.n_params, self.n_nodes))
        
    def __call__(self, x):
        return self.getOracle(x)
        
    