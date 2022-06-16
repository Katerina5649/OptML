import numpy as np

class Oracle:

    def __init__(self, func_type, n_params, n_nodes):
        
        self.type = func_type
        self.n_params = n_params
        self.n_nodes = n_nodes
        
        if self.type == "strongly convex":
            
            f = lambda col: np.power(np.linalg.norm(col, 2), 2)
            df = lambda col: 2 * col
            
            self.func = lambda X: (np.apply_along_axis(f, 0, X), np.apply_along_axis(df, 0, X))
            
            self.x_star = np.zeros((self.n_params, 1))
            self.f_star = 0.
        
        elif self.type == "convex":
            
            a = 0.75
            y = np.random.uniform(0, 5, (self.n_params, 1))
            f = lambda col: 0.5 * np.power(np.linalg.norm(col * a - y, 2), 2)
            df = lambda col: np.squeeze(a * (np.expand_dims(col, axis=1) * a - y), axis=1)
            
            self.func = lambda X: (np.apply_along_axis(f, 0, X), np.apply_along_axis(df, 0, X))
            
            self.x_star = y / a
            self.f_star = 0.
            
        elif self.type == "non convex":
            self.func = lambda x: (np.log(x**2 + 2), 2*x/(x**2 + 2))
            self.min = 0
        
    def getMin(self):
        return self.x_star, self.f_star
    
    def getMu(self):
        return -1 if (self.type != "strongly convex") else 0.5
    
    def getType(self):
        return self.type
         
    def __call__(self, x):
        return self.func(x)
        
    