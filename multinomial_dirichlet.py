from probability_model import ProbabilityModel
from data import Data
import numpy as np
from scipy.special import gammaln

class MultinomialDirichlet(ProbabilityModel):
    def __init__(self,data,k=None,alpha=.1):
        ProbabilityModel.__init__(self,data,alpha)
        if k is not None:
            self.k=k if hasattr(k, "__len__") else np.repeat(k,data.p)
        else:
            self.data.calculate_unique_categories()
            self.k=self.data.num_categories

        self.alphas=alpha if hasattr(alpha, "__len__") else np.array([np.repeat(alpha,self.k[j]) for j in range(self.data.p)])
        self.alpha_dots=np.array([sum(a) for a in self.alphas])
        self.data.calculate_y_cumulative_counts(self.k)

    def likelihood_j(self,j=0,start=0,end=None):
        counts=self.data.get_y_cumulative_counts(j,start,end)
        alpha_dot,n_dot=self.alpha_dots[j],sum(counts)
        lhd_terms=np.array([0.0 if c_k==0 else gammaln(a_k+c_k)-gammaln(a_k) for (a_k,c_k) in zip(self.alphas[j],counts)])
        const=-n_dot*np.log(n_dot) if (self.alpha_dots[j]==0) else gammaln(alpha_dot)-gammaln(alpha_dot+n_dot)
        return(sum(lhd_terms)+const)
