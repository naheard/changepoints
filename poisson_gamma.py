## poisson_gamma.py

from probability_model import ProbabilityModel
from data import Data
import numpy as np
from scipy.special import gammaln

class PoissonGamma(ProbabilityModel):
    def __init__(self,data=None,alpha_beta=(.1,.1),T=1,p=1):
        ProbabilityModel.__init__(self,data,alpha_beta)
        self.a0,self.b0=alpha_beta
        self.T=1
        self.p=p if self.data is None else self.data.p
        self.density_constant=-gammaln(self.a0)+self.a0*np.log(self.b0)
        self.data_type=int
        self.calculate_data_summaries()

    def calculate_data_summaries(self):
        if self.data is not None:
            self.data.calculate_y_cumulative_sum()
            self.data.calculate_y_cumulative_sum_log_factorials()

    def likelihood_component(self,j=0,start_end=[(0,None)],y=None):
        r=y if y is not None else self.data.get_combined_y_sums(j,start_end)
        n=1 if y is not None else sum([e-s for s,e in start_end])
        slyf = gammaln(r+1) if y is not None else self.data.get_combined_y_factorials(j,start_end)
        return(self.log_density(r,n,slyf))

    def sample_parameter(self):
        return(np.random.gamma(self.a0,self.b0,size=self.p))

    def simulate_data(self,n,thetas=None,x=None):
        if thetas is None:
            thetas=self.sample_parameter()
        return([np.random.poisson(theta,size=n) for theta in thetas])

    def log_density(self,r,n=1,s=None):
        if n==0:
            return(0)
        if s is None:
            s = lngamma(r+1)
        ld=self.density_constant+gammaln(self.a0+r)-(self.a0+r)*np.log(self.b0+n*self.T)-s
        return(ld)

    def component_mean(self,j=0,start_end=[(0,None)],y=None):
        r=y if y is not None else self.data.get_combined_y_sums(j,start_end)
        n=1 if y is not None else sum([e-s for s,e in start_end])
        return(self.mean_par(r,n))

    def mean_par(self,r,n=1):
        return((self.a0+r)/(self.b0+n*self.T))
