from probability_model import ProbabilityModel
from data import Data
import numpy as np

class RegimeModel(ProbabilityModel):
    def __init__(self):
        ProbabilityModel.__init__(self)

    def likelihood_j(self,j=0,start_end=[(0,None)],y=None):
        if y is not None:
            return self.log_density(y)
        return(0)

    def log_density(self,y):
        if y[0]!=0:
            return(-float("inf"))#first regime must be zero
        n=len(y)
        k=max(y)#number of non-zero regimes
#        log_p_k=-np.log(n-1) #assumes k~U({0,1,..,n-1})
        first_occurrences=np.zeros(k+1,dtype=int)
        for r in range(1,k+1):
            first_occurrences[r]=next(i for i in range(first_occurrences[r-1]+1,n) if y[i]==r)
        d_first_occurrences=np.diff(first_occurrences)
        log_p_regimes=np.sum([-d_first_occurrences[i]*np.log((i+2)) for i in range(len(d_first_occurrences))])#assumes regime at each step is Uniform on available regimes
        print(d_first_occurrences)
        return(log_p_regimes)
