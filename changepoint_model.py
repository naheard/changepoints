from probability_model import ProbabilityModel
from data import Data
import numpy as np
import sys
from collections import defaultdict

class Changepoint(object):
    def __init__(self,t,probability_models=None,regime_number=None):
        self.tau=t
        self.probability_models=probability_models
        if probability_models is not None:
            self.find_data_positions()
#        self.lhds=np.array([],dtype=float)#empty vector of likelihoods, one for each probability model
#        self.indicators=None
        self.regime_number=regime_number

    def find_data_positions(self,t=None):
        t=t if t is not None else self.tau
        self.data_locations=np.array([pm.find_data_position(t) for pm in self.probability_models],dtype=int)

    def __lt__(self,other):
        return(self.tau<other.tau)

class Regime(object):
    def __init__(self,cp_indices):
        self.cp_indices=cp_indices

    def __lt__(self,other):#find which regime occurs first
        return(self.cp_indices[0]<other.cp_indices[0])

    def insert_cp_index(self,cp_i):
        position=np.searchsorted(self.cp_indices,cp_i)
        self.cp_indices.insert(position,cp_i)

    def remove_cp_index(self,cp_i):
        self.cp_indices.remove(cp_i)

    def length(self):
        return len(self.cp_indices)

class ChangepointModel(object):
    def __init__(self,probability_models=np.array([],dtype=ProbabilityModel)):
        self.probability_models=probability_models
        self.num_probability_models=len(self.probability_models)
        self.T=max([pm.data.get_x_max() for pm in self.probability_models if pm.data is not None])
        self.baseline_changepoint=Changepoint(-float("inf"),self.probability_models)
        self.set_changepoints([])
        self.regime_lhds=defaultdict(list)
        self.calculate_likelihood()#likelihoods for each prob. model

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
            self.regimes=[Regime([_]) for _ in range(self.num_cps)]
        else:
            self.regimes=[Regime(r) for r in regimes]
        self.num_regimes=len(self.regimes)
        for r_i in range(self.num_regimes):
            for i in self.regimes[r_i].cp_indices:
                self.cps[i].regime_number=r_i

    def find_position_in_changepoints(self,t):
        position=np.searchsorted(self.cps,Changepoint(t))
        return(position)

    def find_position_in_regimes(self,cp_indices):
        position=np.searchsorted(self.regimes,Regime(cp_indices))
        return(position)

    def write_changepoints_and_regimes(self,stream=sys.stdout,delim="\t"):
        stream.write(delim.join([":".join(map(str,(cp.tau,cp.regime_number))) for cp in self.cps])+"\n")

    def delete_regime(self,regime_index):
        for pm_i in range(self.num_probability_models):
            del self.regime_lhds[pm_i][regime_index]

        for r_i in range(regime_index+1,self.num_regimes):
            for cp_i in self.regimes[r_i].cp_indices:
                self.cps[cp_i].regime_number-=1
        del self.regimes[regime_index]
        self.num_regimes-=1

    def add_regime(self,cp_indices):
        regime_index=self.find_position_in_regimes(cp_indices)
        self.regimes.insert(regime_index,Regime(cp_indices))
        self.num_regimes+=1
        for pm_i in range(self.num_probability_models):
            self.regime_lhds[pm_i].insert(regime_index,0)

        for r_i in range(regime_index+1,self.num_regimes):
            for cp_i in self.regimes[r_i].cp_indices:
                self.cps[cp_i].regime_number+=1

    def add_changepoint_index_to_regime(self,index,regime_number):
        if regime_number==self.num_regimes:
            self.add_regime([index])
        else:
            self.regimes[regime_number].insert_cp_index(index)

    def delete_changepoint(self,index):
        if index==0:
            sys.exit('Error! Cannot delete basline changepoint')
        regime_number=self.cps[index].regime_number
        if self.regimes[regime_number].length()==1:
            self.delete_regime(regime_number)
        else:
            self.regimes[regime_number].remove_cp_index(index)
        self.cps=np.delete(self.cps,index)
        self.num_cps-=1
        for r in self.regimes:
            for i in range(r.length()):
                if r.cp_indices[i]>index:
                    r.cp_indices[i]-=1

    def add_changepoint(self,t,regime_number=None):
        index=self.find_position_in_changepoints(t)
        if regime_number is None:
            regime_number=self.num_regimes
        self.cps=np.insert(self.cps,index,Changepoint(t,self.probability_models,regime_number))
        self.num_cps+=1
        for r in self.regimes:
            for i in range(r.length()):
                if r.cp_indices[i]>=index:
                    r.cp_indices[i]+=1
        self.add_changepoint_index_to_regime(index,regime_number)

    def shift_changepoint(self,index,t):
        self.cps[index].tau=t
        self.cps[index].find_data_positions()

    def change_regime_of_changepoint(self,index,new_regime_number):
        regime_number=self.cps[index].regime_number
        if self.regimes[regime_number].length()==1:
            self.delete_regime(regime_number)
            if regime_number<new_regime_number:
                new_regime_number-=1
        else:
            self.regimes[regime_number].remove_cp_index(index)
        self.add_changepoint_index_to_regime(index,new_regime_number)

    def calculate_likelihood(self):
        for pm_i in range(self.num_probability_models):
            self.regime_lhds[pm_i]=[0]*self.num_regimes
            for r_i in range(self.num_regimes):
                start_end=[self.get_changepoint_index_segment_start_end(pm_i,j) for j in self.regimes[r_i].cp_indices]
#                print(r_i,start_end)
                self.regime_lhds[pm_i][r_i]=self.probability_models[pm_i].likelihood(start_end)
        self.lhds=np.array([sum(self.regime_lhds[pm_i]) for pm_i in range(self.num_probability_models)],dtype=float)
        return(sum(self.lhds))
