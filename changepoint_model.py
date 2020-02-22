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
        self.regime_lhds=np.array([[] for pm in self.probability_models])
        self.lhds=np.array([pm.likelihood() for pm in self.probability_models])#vector of likelihoods, one for each probability model
        self.likelihood()

    def get_changepoint_index_segment_start_end(self,pm_index,index):
        start=self.cps[index].data_locations[pm_index]
        end=self.cps[index+1].data_locations[pm_index] if index<self.num_cps-1 else self.probability_models[pm_index].data.n
        return((start,end))

    def get_lhd(self):
        return(sum(self.lhds))

    def set_changepoints(self,tau,regimes=None):
        self.cps=np.sort(np.array([self.baseline_changepoint]+[Changepoint(t,self.probability_models) for t in tau],dtype=Changepoint))
        self.num_cps=len(self.cps)
        if regimes is None:
            self.regimes=[[_] for _ in range(self.num_cps)]
        else:
            self.regimes=regimes
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
        num_regimes=len(regimes)
        regime_lhds=np.array([np.zeros(num_regimes) for pm in self.probability_models])
        for pm_i in range(self.num_probability_models):
            for r_i in range(num_regimes):
                start_end=[self.get_changepoint_index_segment_start_end(pm_i,j) for j in regimes[r_i]]
                regime_lhds[pm_i][r_i]=self.probability_models[pm_i].likelihood(start_end)
        lhds=np.array([sum(regime_lhds[pm_i]) for pm_i in range(self.num_probability_models)],dtype=float)
        return(sum(lhds))
