import numpy as np

class Oracle:

    def __init__(self, func_type, n_params, n_nodes):
        
        self.type = func_type
        self.n_params = n_params
        self.n_nodes = n_nodes
        
        if self.type == "strongly convex":
            self.func = lambda x: (np.power(np.linalg.norm(x), 2), 2 * x)
            self.min = np.zeros((self.n_params, self.n_nodes))
        
        elif self.type == "convex":
            a = 0.75
            y = np.random.uniform(0, 5, (x.shape[0], 1))
            
            self.func = lambda x: (0.5 * np.power(np.linalg.norm(x * a - y), 2), a * (x * a - y)) ## really not sure
            self.min = y / a
        
    def getMin(self):
        return self.min
        
    def __call__(self, x):
        return self.func(x)
        
    