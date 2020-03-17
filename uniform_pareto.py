from probability_model import ProbabilityModel
from data import Data
import numpy as np
from scipy.special import gammaln

class UniformPareto(ProbabilityModel):
    def __init__(self,data=None,alpha_beta=(.1,.1),p=1):
        ProbabilityModel.__init__(self,data,alpha_beta)
        self.a0,self.b0=alpha_beta
        self.p=p if self.data is None else self.data.p
        self.density_constant=np.log(self.a0)+self.a0*np.log(self.b0)

    def likelihood_component(self,j=0,start_end=[(0,None)],y=None):
        max_x=y if y is not None else self.data.get_combined_y_maxs(j,start_end)
        n=1 if y is not None else sum([e-s for s,e in start_end])
        return(self.log_density(max_x,n))

    def sample_parameter(self):
        return((np.random.pareto(self.a0,size=self.p)+1)*self.b0)

    def simulate_data(self,n,thetas=None,x=None):
        if thetas is None:
            thetas=self.sample_parameter()
        return([np.random.uniform(theta,size=n) for theta in thetas])

    def log_density(self,r,n=1):
        if n==0:
            return(0)
        ld=self.density_constant-np.log(self.a0+n)+(self.a0+n)*np.log(max(self.b0,r))
        return(ld)
