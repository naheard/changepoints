## gaussian_process.py

from probability_model import ProbabilityModel
from data import Data
import numpy as np
from scipy.special import gammaln

LOG_PI=np.log(np.pi)

class GP(ProbabilityModel):
    def __init__(self,data=None,alpha_beta=(1,1),l=1,p=1):
        ProbabilityModel.__init__(self,data,alpha_beta) #y_i~N(mu,sigma**2)
        self.a0,self.b0=alpha_beta #sigma**(-2)~Gamma(a0,b0)
        self.l=l #a multiplier for the standard deviation of mu, mu~N(0,(sigma*v)**2)
        self.p=p if self.data is None else self.data.p
        self.density_constant=self.a0*np.log(self.b0)-gammaln(self.a0)
        if self.data:
            self.data.create_kernel_covariance_matrix()

    def likelihood_component(self,j=0,start_end=[(0,None)],y=None):
        ys=self.data.get_combined_ys(j,start_end)
        K=self.data.get_combined_kernel_covariance_matrix(start_end)
        n=len(ys)
        return(self.log_density(ys,K,n))

    def sample_parameter(self):
        sigma=1.0/np.sqrt(np.random.gamma(self.a0,1.0/self.b0,size=self.p))
        mu=np.array([np.random.multivariate_normal(mean=np.zeros(self.data.n),cov=s*self.data.Kx) for s in sigma])
        return(mu,sigma)

    def simulate_data(self,n,thetas=None,x=None):
        if thetas is None:
            mu,sigma=self.sample_parameter()
        else:
            mu,sigma=thetas[0],thetas[1]
        return([mu[i]+np.random.normal(scale=sigma[i],size=n) for i in range(self.p)])

    def log_density(self,ys,K,n=1):
        yKy=sum([sum([ys[i]*K[i,j]*ys[j] for j in range(n)]) for i in range(n)])
        ldetK=np.linalg.slogdet(K)[1]
        ld=self.density_constant-ldetK-(self.a0+.5*n)*np.log(self.b0+.5*yKy)-gammaln(self.a0+.5*n)
        return(ld)
