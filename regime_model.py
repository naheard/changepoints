## regime_model.py

from probability_model import ProbabilityModel
from data import Data
import numpy as np

LOG_TWO=np.log(2)

class RegimeModel(ProbabilityModel):
    def __init__(self,disallow_successive_regimes=True,spike_regimes=False):
        ProbabilityModel.__init__(self)
        self.disallow_successive_regimes=disallow_successive_regimes
        self.spike_regimes=spike_regimes
        self.data_type=int
        self.max_num_regimes=float("inf")

    def likelihood_component(self,j=0,start_end=[(0,None)],y=None):
        if y is not None:
            return self.log_density(y)
        return(0)

    def sample_parameter(self,n=1):
        regimes=np.zeros(n,dtype=int)
        num_regimes=1
        blocked_regimes=[]
        for i in range(1,n):
            if self.disallow_successive_regimes:
                blocked_regimes=[regimes[i-1]]
            possible_regimes=[0,num_regimes] if self.spike_regimes else list(range(num_regimes+1))
            allowed_regimes=[r for r in possible_regimes if r not in blocked_regimes]
            num_allowed_regimes=len(allowed_regimes)
            if num_allowed_regimes==1:
                regimes[i]=allowed_regimes[0]
            else:
                regimes[i]=np.random.choice(allowed_regimes)
            if regimes[i]==num_regimes:
                num_regimes+=1
        return(regimes)

    def simulate_data(self,n,thetas=None,x=None):
        return(sample_parameter())

    def log_density(self,y):
        if y[0]!=0 or (self.disallow_successive_regimes and 0 in np.diff(y)) or max(y)>=self.max_num_regimes:
            return(-float("inf"))#first regime must be zero
        n=len(y)
        if self.spike_regimes:
            if len(y)==1:
                return(0)
            non_zero_y=[y_i for y_i in y if y_i!=0]
            n_non_zero=len(non_zero_y)
            if n_non_zero != len(set(non_zero_y)):
                return(-float("inf"))#non zero-regimes must occur only once in this model
            if y[-1]!=0:
                n_non_zero-=1
            return(-(n_non_zero if self.disallow_successive_regimes else n-1) *LOG_TWO)

        k=max(y)#number of non-zero regimes
#        log_p_k=-np.log(n-1) #assumes k~U({0,1,..,n-1})
        first_occurrences=np.zeros(k+1,dtype=int)
        for r in range(1,k+1):
            first_occurrences[r]=next(i for i in range(first_occurrences[r-1]+1,n) if y[i]==r)
        d_first_occurrences=np.diff(first_occurrences)
        log_p_regimes=np.sum([-d_first_occurrences[i]*np.log((i+(1 if self.disallow_successive_regimes else 2))) for i in range(len(d_first_occurrences))])#assumes regime at each step is Uniform on available regimes
        return(log_p_regimes)

    def propose_regime(self,num_regimes,left_regime=None,right_regime=None, current_regime=None,solo=False):
        blocked_regimes=[current_regime] if current_regime is not None else []
        if self.disallow_successive_regimes:
            for r in (left_regime,right_regime):
                if r is not None:
                    blocked_regimes+=[r]

        possible_regimes=[0] if self.spike_regimes else list(range(num_regimes))
        if not solo and num_regimes<self.max_num_regimes:
            possible_regimes+=[num_regimes]
        allowed_regimes=[r for r in possible_regimes if r not in blocked_regimes]
        num_allowed_regimes=len(allowed_regimes)
        if num_allowed_regimes==0:
            return(None,-float("inf"))
        elif num_allowed_regimes==1:
            return(allowed_regimes[0],0)
        else:
            return(np.random.choice(allowed_regimes),-np.log(num_allowed_regimes))
