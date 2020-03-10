from probability_model import ProbabilityModel
from data import Data
import numpy as np
from scipy.special import gammaln

class NormalNIG(ProbabilityModel):
    def __init__(self,data=None,alpha_beta=(.1,.1),v=1,p=1):
        ProbabilityModel.__init__(self,data,alpha_beta) #y_i~N(mu,sigma**2)
        self.a0,self.b0=alpha_beta #sigma**(-2)~Gamma(a0,b0)
        self.v=v #a multiplier for the standard deviation of mu, mu~N(0,(sigma*v)**2)
        self.p=p if self.data is None else self.data.p
        self.density_constant=-np.log(self.v)+self.a0*np.log(self.b0)-gammaln(self.a0)
        if self.data is not None:
            self.data.calculate_y_cumulative_sum()
            self.data.calculate_y_cumulative_sum_sq()

    def likelihood_j(self,j=0,start_end=[(0,None)],y=None):
        lhd=self.density_constant
        sy=sum(y) if y is not None else self.data.get_combined_y_sums(j,start_end)
        sy2=sum(y*y) if y is not None else self.data.get_combined_y_sum_squares(j,start_end)
        n=1 if y is not None else sum([e-s for s,e in start_end])
        return(self.log_density(sy,sy2,n))

    def sample_parameter(self):
        sigma=1.0/np.sqrt(np.random.gamma(self.a0,1.0/self.b0,size=self.p))
        mu=np.random.normal(scale=sigma*self.v,size=self.p)
        return(mu,sigma)

    def simulate_data(self,n,mu=None,sigma=None):
        if mu is None or sigma is None:
            mu,sigma=self.sample_parameter()
        return([np.random.normal(mu[i],sigma[i],size=n) for i in range(self.p)])

    def log_density(self,sy,sy2,n=1):
        ld=self.density_constant+sy
        return(ld)
