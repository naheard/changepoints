from probability_model import ProbabilityModel
from data import Data
import numpy as np
from scipy.special import gammaln

class PoissonGamma(ProbabilityModel):
    def __init__(self,data=None,alpha_beta=(.1,.1),T=1):
        ProbabilityModel.__init__(self,data,alpha_beta)
        self.a0,self.b0=alpha_beta
        self.T=1
        self.density_constant=-gammaln(self.a0)
        if self.data is not None:
            self.data.calculate_y_cumulative_sum()

    def likelihood_j(self,j=0,start_end=[(0,None)],y=None):
        r=y if y is not None else self.data.get_combined_y_sums(j,start_end)
        n=1 if y is not None else sum([e-s for s,e in start_end])
        return(self.log_density(r,n))

    def sample_parameter(self):
        return(np.random.gamma(self.a0,self.b0))

    def log_density(self,r,n=1):
        ld_terms_zero=self.a0*np.log(self.b0/(self.b0+float(n)))
        ld=ld_terms_zero+self.density_constant+(0 if r==0 else (gammaln(self.a0+r)-r*np.log(self.b0+n)))
        return(ld)
