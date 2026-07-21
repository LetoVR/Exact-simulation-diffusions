import numpy as np

class BSmodel:
    def __init__ (self, S0, r, sigma):
        self.X0 = S0
        self.r = r
        self.vol = sigma
    def b(self,x): 
        return self.r*x
    def sigma(self,x):
        return self.vol*x
    def g(self,x) : 
        return 1/self.vol*np.log(x)
    def g_inv(self,x) : 
        return np.exp(self.vol*x)
    def a(self,x) : 
        return self.r/self.vol-self.vol/2
    def h(self,x) : 
        return (self.r/self.vol-self.vol/2)*x
    def c(self,x) : 
        return (-1/2)*(self.r/self.vol-self.vol/2)**2
    def c_bound(self) : 
        return (1/2)*(self.r/self.vol-self.vol/2)**2
    

    
