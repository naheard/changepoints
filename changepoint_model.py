from probability_model import ProbabilityModel
from data import Data
import numpy as np
import sys
from collections import defaultdict,Counter
from poisson_gamma import PoissonGamma
from regime_model import RegimeModel

class Changepoint(object):
    def __init__(self,t,probability_models=None,regime_number=None):
        self.tau=t
        self.probability_models=probability_models
        if probability_models is not None:
            self.find_data_positions()
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
        self.N=max([pm.data.n for pm in self.probability_models if pm.data is not None])
        self.max_num_changepoints=float("inf")
        self.min_cp_spacing=float(self.T)/self.N
        self.changepoint_prior=PoissonGamma(alpha_beta=[.01,10])
        self.regimes_prior=RegimeModel()
        self.baseline_changepoint=Changepoint(-float("inf"),self.probability_models)
        self.set_changepoints([])
        self.regime_lhds=[np.zeros(self.num_probability_models) for _ in range(self.num_regimes)]
        self.calculate_posterior()#calculate likelihoods for each prob. model
        self.create_mh_dictionary()
        self.proposal_move_counts=Counter()
        self.proposal_acceptance_counts=Counter()
        self.num_cps_counter=Counter()

    def create_mh_dictionary(self):
        self.proposal_functions={}
        self.proposal_functions["delete_changepoint"]=self.propose_delete_changepoint
        self.proposal_functions["add_changepoint"]=self.propose_add_changepoint
        self.undo_proposal_functions={}
        self.undo_proposal_functions["delete_changepoint"]=self.undo_propose_delete_changepoint
        self.undo_proposal_functions["add_changepoint"]=self.undo_propose_add_changepoint

    def get_changepoint_index_segment_start_end(self,pm_index,index):
        start=self.cps[index].data_locations[pm_index]
        end=self.cps[index+1].data_locations[pm_index] if index<self.num_cps else self.probability_models[pm_index].data.n
        return((start,end))

    def distance_to_rh_cp(self,index):
        rh_cp_location=self.T if index==self.num_cps else self.cps[index+1].tau
        return(rh_cp_location-self.cps[index].tau)

    def set_changepoints(self,tau,regimes=None):
        self.cps=np.sort(np.array([self.baseline_changepoint]+[Changepoint(t,self.probability_models) for t in tau],dtype=Changepoint))
        self.num_cps=len(self.cps)-1
        if regimes is None:
            self.regimes=[Regime([_]) for _ in range(self.num_cps+1)]
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
        self.deleted_regime_lhd=self.regime_lhds[regime_index]
        del self.regime_lhds[regime_index]
        for r_i in range(regime_index+1,self.num_regimes):
            for cp_i in self.regimes[r_i].cp_indices:
                self.cps[cp_i].regime_number-=1
        del self.regimes[regime_index]
        self.num_regimes-=1

    def add_regime(self,cp_indices):
        regime_index=self.find_position_in_regimes(cp_indices)
        self.regimes.insert(regime_index,Regime(cp_indices))
        self.num_regimes+=1
        for i in cp_indices:
            self.cps[i].regime_number=regime_index

        self.regime_lhds.insert(regime_index,np.zeros(self.num_probability_models) if self.deleted_regime_lhd is None else self.deleted_regime_lhd)
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
        self.cps=np.insert(self.cps,index,Changepoint(t,self.probability_models,regime_number))
        self.num_cps+=1
        for r in self.regimes:
            for i in range(r.length()):
                if r.cp_indices[i]>=index:
                    r.cp_indices[i]+=1
        if regime_number is None:
            regime_number=self.num_regimes
        self.add_changepoint_index_to_regime(index,regime_number)
        return(index)

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
        for r_i in range(self.num_regimes):
            for pm_i in range(self.num_probability_models):
                start_end=[self.get_changepoint_index_segment_start_end(pm_i,j) for j in self.regimes[r_i].cp_indices]
#                print(r_i,start_end)
                self.regime_lhds[r_i][pm_i]=self.probability_models[pm_i].likelihood(start_end)
        self.likelihood=sum([sum(lr) for lr in self.regime_lhds])
        return(self.likelihood)

    def get_effective_changepoint_locations(self):
        return([self.cps[i+1].tau for i in range(self.num_cps) if self.cps[i+1].regime_number!=self.cps[i].regime_number])

    def calculate_prior(self):
        self.prior=self.changepoint_prior.likelihood(y=self.num_cps)
        if self.num_cps>0 and min([self.distance_to_rh_cp(i) for i in range(self.num_cps+1)])<self.min_cp_spacing:
            self.prior=-float("inf")
        self.regime_sequence=[self.cps[i].regime_number for i in range(self.num_cps+1)]
        regime_prior=self.regimes_prior.likelihood(y=self.regime_sequence)
        self.prior+=regime_prior
        return(self.prior)

    def calculate_posterior(self):
        self.calculate_likelihood()
        self.calculate_prior()
        self.posterior=self.likelihood+self.prior
        return(self.posterior)

    def mcmc(self,iterations=20,burnin=0,seed=None):
        if seed is not None:
            np.random.seed(seed)
        for self.iteration in range(-burnin,iterations):
            self.mh_accept=True
            self.deleted_regime_lhd=None
            self.propose_move()
            self.accept_reject()
            if not self.mh_accept:
                self.undo_move()
            self.num_cps_counter[self.num_cps]+=1

        self.write_changepoints_and_regimes()
        print(self.posterior)
        self.print_acceptance_rates()
        self.calculate_posterior_means()
        sys.stdout.write("E[#Changepoints] = "+str(self.mean_num_cps)+"\n")

    def choose_move(self):
        self.available_move_types=[]
        if self.num_cps>0:
            self.available_move_types.append("delete_changepoint")
        if self.num_cps<self.max_num_changepoints:
            self.available_move_types.append("add_changepoint")
        self.move_type=np.random.choice(self.available_move_types)

    def propose_move(self):
        self.choose_move()
        self.proposal_move_counts[self.move_type]+=1
        self.proposal_functions[self.move_type]()

    def undo_move(self):
        self.undo_proposal_functions[self.move_type]()

    def accept_reject(self):
        if self.mh_accept and self.posterior==-float("inf"):
            self.mh_accept=False
        if not self.mh_accept:
            return()
        self.accpeptance_ratio=self.posterior-self.stored_posterior
        if self.accpeptance_ratio<0 and np.random.exponential()<-self.accpeptance_ratio:
            self.mh_accept=False
        if self.mh_accept:
            self.proposal_acceptance_counts[self.move_type]+=1
#        else:
#            self.undo_move()

    def propose_delete_changepoint(self,index=None):
        self.proposed_index=index if index is not None else (1 if self.num_cps==1 else np.random.randint(1,self.num_cps))
        self.stored_cp=self.cps[self.proposed_index]
        self.stored_num_regimes=self.num_regimes
        self.stored_posterior=self.posterior
        self.affected_regimes=set([self.cps[self.proposed_index-1].regime_number,self.cps[self.proposed_index].regime_number])
        self.delete_changepoint(self.proposed_index)
        self.calculate_posterior()

    def undo_propose_delete_changepoint(self):
        regime=self.num_regimes if self.stored_num_regimes>self.num_regimes else self.stored_cp.regime_number
        self.add_changepoint(self.stored_cp.tau,regime)
        self.calculate_posterior()

    def propose_add_changepoint(self,t=None,regime=None):
        self.stored_num_regimes=self.num_regimes
        self.stored_posterior=self.posterior
        if t is None:
            t=np.random.uniform(0,self.T)
        self.proposed_index=self.add_changepoint(t,regime)
        self.affected_regimes=set([self.cps[self.proposed_index-1].regime_number,self.cps[self.proposed_index].regime_number])
        self.calculate_posterior()

    def undo_propose_add_changepoint(self):
        self.delete_changepoint(self.proposed_index)
        self.calculate_posterior()

    def calculate_posterior_means(self):
        self.mean_num_cps=sum([k*self.num_cps_counter[k] for k in self.num_cps_counter])/sum(self.num_cps_counter.values())

    def print_acceptance_rates(self,stream=sys.stdout):
        for m,c in self.proposal_move_counts.most_common():
            stream.write(m+":\t"+str(self.proposal_acceptance_counts[m]/float(c)*100)+"%\n")
