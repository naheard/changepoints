## gaussian_process.py

from probability_model import ProbabilityModel
from data import *
import numpy as np
from scipy.special import gammaln

LOG_PI=np.log(np.pi)

class GP(ProbabilityModel):
    def __init__(self,data=None,alpha_beta=(1,1),v=1,l=1,p=1):
        ProbabilityModel.__init__(self,data,alpha_beta) #y_i~N(mu,sigma**2)
        self.a0,self.b0=alpha_beta #sigma**(-2)~Gamma(a0,b0)
        self.v=v
        self.l=l #a multiplier for the standard deviation of mu, mu~N(0,(sigma*v)**2)
        self.p=p if self.data is None else self.data.p
        self.density_constant=-np.log(self.v)+self.a0*np.log(self.b0)-gammaln(self.a0)
        if self.data:
            self.data.create_kernel_covariance_matrix(gaussian_kernel,par=self.l)

    def likelihood_component(self,j=0,start_end=[(0,None)],y=None):
        ys=self.data.get_combined_ys(j,start_end)
        n=len(ys)
        K=self.v*self.data.get_combined_kernel_covariance_matrix(start_end)+np.identity(n)
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
        ldetK=np.linalg.slogdet(K)[1]
        K_inv=np.linalg.inv(K)
        yK_invy=sum([sum([ys[i]*K_inv[i,j]*ys[j] for j in range(n)]) for i in range(n)])
        ld=self.density_constant-ldetK-(self.a0+.5*n)*np.log(self.b0+.5*yK_invy)-gammaln(self.a0+.5*n)
        return(ld)

    def component_mean(self,j=0,start_end=[(0,None)],y=None):
        ys=self.data.get_combined_ys(j,start_end)
        n=len(ys)
        K=self.v*self.data.get_combined_kernel_covariance_matrix(start_end)+np.identity(n)
        return(self.mean_par(ys,K,n))

    def mean_par(self,ys,K,n=1):
        ldetK=np.linalg.slogdet(K)[1]
        K_inv=np.linalg.inv(K)
        K2 = K - np.identity(n)
        return([sum([sum([K2[i,_]*K_inv[i,j]*ys[j] for j in range(n)]) for i in range(n)]) for _ in range(n)])

