from probability_model import ProbabilityModel
from data import Data
import numpy as np
import sys

class Changepoint(object):
    def __init__(self,t,probability_models=None):
        self.tau=t
        self.probability_models=probability_models
        if probability_models is not None:
            self.data_locations=self.find_data_positions()
#        self.lhds=np.array([],dtype=float)#empty vector of likelihoods, one for each probability model
#        self.indicators=None
        self.regime=None


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
        self.lhds=np.array([pm.likelihood() for pm in self.probability_models])#likelihoods for each prob. model
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
        for r_i in range(self.num_regimes):
            for i in self.regimes[r_i]:
                self.cps[i].regime=r_i
#        self.regimes=np.arange(len(self.cps))
#        self.num_regimes=max(self.regimes)+1

    def find_position_in_changepoints(self,t):
        position=np.searchsorted(self.cps,Changepoint(t))
        return(position)

    def write_changepoints_and_regimes(self,stream=sys.stdout,delim="\t"):
        stream.write(delim.join([":".join(map(str,(cp.tau,cp.regime))) for cp in self.cps])+"\n")

    def delete_regime(self,regime_index):
        for pm_i in range(self.num_probability_models):
            np.delete(self.regime_lhds[pm_i],regime_index)

        for r_i in range(regime_index+1,self.num_regimes):
            for cp_i in self.regimes[r_i]:
                self.cps[cp_i].regime-=1
        del self.regimes[regime_index]

    def delete_changepoint(self,index):
        if index==0:
            sys.exit('Error! Cannot delete basline changepoint')
        r_i=self.cps[index].regime
        if len(self.regimes[r_i])==1:
            self.delete_regime(r_i)
        else:
            del self.regimes[r_i][index]
        self.cps=np.delete(self.cps,index)

    def add_changepoint(self,t,regime=None):
        position=self.find_position_in_changepoints(t)

    def calculate_likelihood(self):
        self.regime_lhds=np.array([np.zeros(self.num_regimes) for pm in self.probability_models])
        for pm_i in range(self.num_probability_models):
            for r_i in range(self.num_regimes):
                start_end=[self.get_changepoint_index_segment_start_end(pm_i,j) for j in self.regimes[r_i]]
                self.regime_lhds[pm_i][r_i]=self.probability_models[pm_i].likelihood(start_end)
        self.lhds=np.array([sum(self.regime_lhds[pm_i]) for pm_i in range(self.num_probability_models)],dtype=float)
        return(sum(self.lhds))

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
