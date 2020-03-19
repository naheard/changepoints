## bernoulli_beta.py

from multinomial_dirichlet import MultinomialDirichlet
from data import Data
import numpy as np
from scipy.special import gammaln

class BernoulliBeta(MultinomialDirichlet):
    def __init__(self,data=None,p=1,alpha_beta=[1,1]):
        MultinomialDirichlet.__init__(self,data=data,k=np.repeat(2,p), alpha=np.tile(np.array(alpha_beta),[1,p]))
