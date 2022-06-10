import numpy as np

class Oracle:

    def __init__(self, func_type):
        
        self.type = func_type
        
    
    def getOracle(self, x):

        if self.type == "strongly convex":
            return np.linalg.norm(np.power(x, 2)), 2 * x
        
    def getMin(self):
        
        if self.type == "strongly convex":
            return 0
        
    def __call__(self, x):
        return self.getOracle(x)
        
    
