## multinomial_dirichlet.py

from probability_model import ProbabilityModel
from data import Data
import numpy as np
from scipy.special import gammaln

class MultinomialDirichlet(ProbabilityModel):
    def __init__(self,data=None,k=None,alpha=.1):
        ProbabilityModel.__init__(self,data,alpha)
        self.k=None
        if k is not None:
            self.k=k if hasattr(k, "__len__") else np.repeat(k,data.p)
        else:
            if self.data.num_categories is None:
                self.data.calculate_unique_categories()
#            self.k=self.data.num_categories
        self.calculate_data_summaries()
        if self.k is None:
            self.k=self.data.num_categories
        self.alphas=alpha if hasattr(alpha, "__len__") else np.array([np.repeat(alpha,self.k[j]) for j in range(len(self.k))])
        self.alpha_dots=np.array([sum(a) for a in self.alphas])
        self.data_type=int

    def calculate_data_summaries(self):
        if self.data is not None:
            self.data.calculate_y_cumulative_counts(self.k)
            if self.data.num_categories is None:
                self.data.calculate_unique_categories()

    def get_dimension(self):
        return(len(self.alphas))

    def likelihood_component(self,j=0,start_end=[(0,None)],y=None):
        counts=y if y is not None else self.data.get_combined_y_cumulative_counts(j,start_end)
        return(self.log_density(counts,self.alphas[j],self.alpha_dots[j]))

    def sample_parameter(self):
        return([np.random.dirichlet(a) for a in self.alphas])

    def simulate_data(self,n,thetas=None,x=None):
        if thetas is None:
            thetas=self.sample_parameter()
        return([np.random.choice(int(len(theta)),int(n),p=theta) for theta in thetas])

    def log_density(self,counts,alphas,alpha_dot=None):
        n_dot=sum(counts)
        if alpha_dot is None:
            alpha_dot=sum(alphas)
        lhd_terms=np.array([0.0 if c_k==0 else gammaln(a_k+c_k)-gammaln(a_k) for (a_k,c_k) in zip(alphas,counts)])
        const=-n_dot*np.log(n_dot) if (alpha_dot==0) else gammaln(alpha_dot)-gammaln(alpha_dot+n_dot)
        return(sum(lhd_terms)+const)
