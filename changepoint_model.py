from probability_model import ProbabilityModel
from data import Data
import numpy as np
from collections import defaultdict

class Changepoint(object):
    def __init__(self,t,probability_models=None):
        self.tau=t
        self.probability_models=probability_models
        if probability_models is not None:
            self.data_locations=self.find_data_positions()
#        self.lhds=np.array([],dtype=float)#empty vector of likelihoods, one for each probability model
#        self.indicators=None
#        self.regimes=None


    def find_data_positions(self,t=None):
        t=t if t is not None else self.tau
        positions=np.array([pm.find_data_position(t) for pm in self.probability_models],dtype=int)
        return(positions)

    def __lt__(self,other):
        return(self.tau<other.tau)

class ChangepointModel(object):
    def __init__(self,probability_models=np.array([],dtype=ProbabilityModel)):
        self.probability_models=probability_models
        self.num_probability_models=len(self.probability_models)
        self.T=max([pm.data.get_x_max() for pm in self.probability_models if pm.data is not None])
        self.baseline_changepoint=Changepoint(-float("inf"),self.probability_models)
        self.set_changepoints([])
        self.lhds=np.array([pm.likelihood() for pm in self.probability_models])#vector of likelihoods, one for each probability model

    def get_lhd(self):
        return(sum(self.lhds))

    def set_changepoints(self,tau):
        self.cps=np.sort(np.array([self.baseline_changepoint]+[Changepoint(t) for t in tau],dtype=Changepoint))
        self.num_cps=len(self.cps)
        self.regimes=[[i] for i in range(self.num_cps)]
        self.num_regimes=len(self.regimes)
#        self.regimes=np.arange(len(self.cps))
#        self.num_regimes=max(self.regimes)+1

    def find_position_in_changepoints(self,t):
        position=np.searchsorted(self.cps,Changepoint(t))
        return(position)

    def likelihood(self,cps=None,regimes=None):
        if cps is None:
            cps=self.cps
        if regimes is None:
            regimes=self.regimes
        for i in range(self.num_probability_models):
            self.lhds[i]=self.probability_models[i].likelihood()
        return(sum(self.lhds))
