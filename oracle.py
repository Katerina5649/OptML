import numpy as np

class Oracle:

    def __init__(self, func_type, n_params):
        
        self.type = func_type
        self.n_params = n_params
        
        if self.type == "strongly convex":
            
            f = lambda col: np.power(np.linalg.norm(col, 2), 2)
            df = lambda col: 2 * col
            
            self.func = lambda X: (np.apply_along_axis(f, 0, X), np.apply_along_axis(df, 0, X))
            
            self.x_star = np.zeros((self.n_params, 1))
            self.f_star = 0.
        
        elif self.type == "convex":
            
            f = lambda col: np.sum(np.power(col, 4))
            df = lambda col: 4 * np.power(col, 3)
            
            self.func = lambda X: (np.apply_along_axis(f, 0, X), np.apply_along_axis(df, 0, X))
            
            self.x_star = np.zeros((self.n_params, 1))
            self.f_star = 0.
            
        elif self.type == "non convex":
            
            self.func = lambda x: (np.sum(np.log(x**2 + 2)), np.divide(2*x, (x**2 + 2)))
            
            self.x_star = np.zeros((self.n_params, 1))
            self.f_star = self.n_params * np.log(2)
            
    def getMin(self):
        return self.x_star, self.f_star
    
    def getMu(self):
        return -1 if (self.type != "strongly convex") else 0.5
    
    def getType(self):
        return self.type
         
    def __call__(self, x):
        return self.func(x)
        
    