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

    def find_position(self,cp_i):
        return(np.searchsorted(self.cp_indices,cp_i))

    def insert_cp_index(self,cp_i):
        position=self.find_position(cp_i)
        self.cp_indices.insert(position,cp_i)

    def remove_cp_index(self,cp_i):
        self.cp_indices.remove(cp_i)

    def length(self):
        return len(self.cp_indices)

    def write(self,stream=sys.stdout,delim=" "):
        stream.write(delim.join(map(str,self.cp_indices))+"\n")

class ChangepointModel(object):
    def __init__(self,probability_models=np.array([],dtype=ProbabilityModel),infer_regimes=False):
        self.probability_models=probability_models
        self.num_probability_models=len(self.probability_models)
        self.T=max([pm.data.get_x_max() for pm in self.probability_models if pm.data is not None])
        self.N=max([pm.data.n for pm in self.probability_models if pm.data is not None])
        self.max_num_changepoints=float("inf")
        self.min_cp_spacing=float(self.T)/self.N
        self.infer_regimes=infer_regimes
        self.changepoint_prior=PoissonGamma(alpha_beta=[.01,10])
        self.regimes_prior=RegimeModel()
        self.baseline_changepoint=Changepoint(-float("inf"),self.probability_models)
        self.set_changepoints([])
        self.regime_lhds=[np.zeros(self.num_probability_models) for _ in range(self.num_regimes)]
        self.affected_regimes=self.revised_affected_regimes=None
        self.deleted_regime_lhd=None
        self.calculate_posterior()#calculate likelihoods for each prob. model
        self.create_mh_dictionary()
        self.proposal_move_counts=Counter()
        self.proposal_acceptance_counts=Counter()
        self.num_cps_counter=Counter()
        self.move_type=None

    def create_mh_dictionary(self):
        self.proposal_functions={}
        self.proposal_functions["delete_changepoint"]=self.propose_delete_changepoint
        self.proposal_functions["add_changepoint"]=self.propose_add_changepoint
        self.proposal_functions["change_regime"]=self.propose_change_regime_of_changepoint
        self.undo_proposal_functions={}
        self.undo_proposal_functions["delete_changepoint"]=self.undo_propose_delete_changepoint
        self.undo_proposal_functions["add_changepoint"]=self.undo_propose_add_changepoint
        self.undo_proposal_functions["change_regime"]=self.undo_propose_change_regime_of_changepoint

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

        self.regime_lhds=[np.zeros(self.num_probability_models) for _ in range(self.num_regimes)]

    def find_position_in_changepoints(self,t):
        position=np.searchsorted(self.cps,Changepoint(t))
        return(position)

    def find_position_in_regimes(self,cp_indices):
        position=np.searchsorted(self.regimes,Regime(cp_indices))
        return(position)

    def write_changepoints_and_regimes(self,stream=sys.stderr,delim="\t"):
        stream.write(delim.join([":".join(map(str,(cp.tau,cp.regime_number))) for cp in self.cps])+"\n")

    def write_regimes(self,stream=sys.stdout,delim=" "):
        for r_i in self.regimes:
            r_i.write(stream,delim)

    def delete_regime(self,regime_index):
        self.deleted_regime_index=regime_index
        self.deleted_regime_lhd=self.regime_lhds[regime_index]
        del self.regime_lhds[regime_index]
        for r_i in range(regime_index+1,self.num_regimes):
            for cp_i in self.regimes[r_i].cp_indices:
                self.cps[cp_i].regime_number-=1
        del self.regimes[regime_index]
        self.num_regimes-=1
        if self.affected_regimes is not None:
            self.revised_affected_regimes=[]
            for r in self.affected_regimes:
                if r!=regime_index:
                     self.revised_affected_regimes.append(r if r<regime_index else r-1)

    def add_regime(self,cp_indices,regime_index=None):
        if regime_index is None:
            regime_index=self.find_position_in_regimes(cp_indices)
        self.regimes.insert(regime_index,Regime(cp_indices))
        self.num_regimes+=1
        for i in cp_indices:
            self.cps[i].regime_number=regime_index

        self.regime_lhds.insert(regime_index,np.zeros(self.num_probability_models) if self.deleted_regime_lhd is None else self.deleted_regime_lhd)
        for r_i in range(regime_index+1,self.num_regimes):
            for cp_i in self.regimes[r_i].cp_indices:
                self.cps[cp_i].regime_number+=1

    def move_regime_position(self,rn1,rn2): #move regime from postion rn1 to position rn2
        if rn1<=rn2:
            for r_i in range(rn1+1,rn2+1):
                for cp_i in self.regimes[r_i].cp_indices:
                    self.cps[cp_i].regime_number-=1

            self.switch_regimes(rn1,rn2)
        else:
            for r_i in range(rn2,rn1):
                for cp_i in self.regimes[r_i].cp_indices:
                    self.cps[cp_i].regime_number+=1

            self.switch_regimes(rn1,rn2)

    def switch_regimes(self,rn1,rn2): # exectute the moving of regime positions
        self.regimes.insert(rn2,self.regimes.pop(rn1))
        self.regime_lhds.insert(rn2,self.regime_lhds.pop(rn1))
        if self.revised_affected_regimes is not None and self.revised_affected_regimes[-1]==rn1:
            self.revised_affected_regimes[-1]=rn2
        for cp_i in self.regimes[rn2].cp_indices:
            self.cps[cp_i].regime_number=rn2

    def update_and_reposition_regimes(self,cp_index=None,from_regime_number=None,to_regime_number=None):
        new_from_position=new_to_position=None
        if from_regime_number is not None: #cp is being deleted or moved from
            if cp_index==self.regimes[from_regime_number].cp_indices[0]: #currently first cp of regime
                new_from_position=self.find_position_in_regimes(self.regimes[from_regime_number].cp_indices[1:])
                if new_from_position > from_regime_number:
                    new_from_position-=1
                    self.move_regime_position(from_regime_number,new_from_position)
                    if to_regime_number is not None and from_regime_number < to_regime_number and to_regime_number <= new_from_position:
                        to_regime_number-=1
                    if self.revised_affected_regimes is not None:
                        for r_i in range(len(self.revised_affected_regimes)):
                            if from_regime_number < self.revised_affected_regimes[r_i] and self.revised_affected_regimes[r_i] < new_from_position:
                                self.revised_affected_regimes[r_i]-=1
            else:
                new_from_position=from_regime_number
            self.regimes[new_from_position].remove_cp_index(cp_index)

        if to_regime_number is not None: #cp is being added or moved to
            if to_regime_number==self.num_regimes or self.regimes[to_regime_number].find_position(cp_index)==0: #will be first cp of regime
                new_to_position=self.find_position_in_regimes([cp_index])
                if to_regime_number==self.num_regimes:
                    self.add_regime([],self.num_regimes)
                if new_to_position < to_regime_number:
                    self.move_regime_position(to_regime_number,new_to_position)
                    if new_from_position is not None and new_to_position <= new_from_position and new_from_position < to_regime_number:
                        new_from_position+=1
                    if self.revised_affected_regimes is not None:
                        for r_i in range(len(self.revised_affected_regimes)):
                            if new_to_position < self.revised_affected_regimes[r_i] and self.revised_affected_regimes[r_i] < to_regime_number:
                                self.revised_affected_regimes[r_i]+=1
            else:
                new_to_position=to_regime_number
            self.regimes[new_to_position].insert_cp_index(cp_index)
            self.cps[cp_index].regime_number=new_to_position
        return([new_from_position,new_to_position])

    def add_changepoint_index_to_regime(self,index,regime_number):
        if regime_number==self.num_regimes:
            self.add_regime([index])
        else:
            regime_number=self.update_and_reposition_regimes(index,None,regime_number)[1]
        return(regime_number)

    def delete_changepoint(self,index):
        if index==0:
            sys.exit('Error! Cannot delete basline changepoint')
        regime_number=self.cps[index].regime_number
        if self.regimes[regime_number].length()==1:
            self.delete_regime(regime_number)
        else:
            self.update_and_reposition_regimes(index,regime_number,None)
        self.cps=np.delete(self.cps,index)
        self.num_cps-=1
        for r in self.regimes:
            for i in range(r.length()):
                if r.cp_indices[i]>index:
                    r.cp_indices[i]-=1

    def add_changepoint(self,t,regime_number=None,index=None):
        self.proposed_index=index if index is not None else self.find_position_in_changepoints(t)
        self.cps=np.insert(self.cps,self.proposed_index,Changepoint(t,self.probability_models,regime_number))
        self.num_cps+=1
        for r in self.regimes:
            for i in range(r.length()):
                if r.cp_indices[i]>=self.proposed_index:
                    r.cp_indices[i]+=1
        if regime_number is None:
            regime_number=self.num_regimes
        self.add_changepoint_index_to_regime(self.proposed_index,regime_number)

    def shift_changepoint(self,index,t):
        self.cps[index].tau=t
        self.cps[index].find_data_positions()

    def change_regime_of_changepoint(self,index,new_regime_number):
        regime_number=self.cps[index].regime_number
        if self.regimes[regime_number].length()==1:
            self.delete_regime(regime_number)
            if regime_number<new_regime_number:
                new_regime_number-=1
            return([self.add_changepoint_index_to_regime(index,new_regime_number)])
        else:
            return(self.update_and_reposition_regimes(index,regime_number,new_regime_number))

    def calculate_likelihood(self,regimes=None):
        for r_i in (range(self.num_regimes) if regimes is None else np.unique(regimes)):# if self.affected_regimes is None else affected_regimes:
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

    def calculate_posterior(self,regimes=None):
        self.calculate_likelihood(regimes=regimes)
        self.calculate_prior()
        self.posterior=self.likelihood+self.prior
        return(self.posterior)

    def mcmc(self,iterations=20,burnin=0,seed=None):
        if seed is not None:
            np.random.seed(seed)
        for self.iteration in range(-burnin,iterations):
            self.mcmc_refresh()
            self.propose_move()
            self.accept_reject()
            if not self.mh_accept:
                self.undo_move()
            self.num_cps_counter[self.num_cps]+=1
#            self.check_posterior()

        self.write_changepoints_and_regimes()
        sys.stderr.write("Final posterior="+str(self.posterior)+"\n")
        self.print_acceptance_rates()
        self.calculate_posterior_means()
        sys.stderr.write("E[#Changepoints] = "+str(self.mean_num_cps)+"\n")

    def mcmc_refresh(self):
        self.mh_accept=True
        self.deleted_regime_lhd=None
        self.affected_regimes=self.revised_affected_regimes=None

    def choose_move(self):
        self.available_move_types=[]
        if self.num_cps>0:
            self.available_move_types.append("delete_changepoint")
        if self.num_cps<self.max_num_changepoints:
            self.available_move_types.append("add_changepoint")
        if self.infer_regimes and self.num_cps>1:
            self.available_move_types.append("change_regime")
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
        self.stored_regime_lhds={}
        self.affected_regimes=[self.cps[self.proposed_index-1].regime_number,self.cps[self.proposed_index].regime_number]
        self.stored_regime_lhds[self.affected_regimes[0]]=np.copy(self.regime_lhds[self.affected_regimes[0]])
        if self.regimes[self.affected_regimes[1]].length()==1:#regime will be deleted
            self.stored_regime_lhds[self.affected_regimes[1]]=self.regime_lhds[self.affected_regimes[1]]
            if self.affected_regimes[1]<self.affected_regimes[0]:
                self.revised_affected_regimes=[self.affected_regimes[0]-1]
            else:
                self.revised_affected_regimes=[self.affected_regimes[0]]
        else:
            self.revised_affected_regimes=self.affected_regimes[:]
            self.stored_regime_lhds[self.affected_regimes[1]]=np.copy(self.regime_lhds[self.affected_regimes[1]])
        self.delete_changepoint(self.proposed_index)
        self.calculate_posterior(self.revised_affected_regimes)

    def undo_propose_delete_changepoint(self):
        regime=self.num_regimes if self.stored_num_regimes>self.num_regimes else self.stored_cp.regime_number
        self.add_changepoint(self.stored_cp.tau,regime)
        self.stored_num_regimes=self.num_regimes
        for r in self.affected_regimes:
            self.regime_lhds[r]=self.stored_regime_lhds[r]
        self.calculate_posterior(regimes=[])

    def propose_add_changepoint(self,t=None,regime_number=None):
        self.stored_num_regimes=self.num_regimes
        self.stored_posterior=self.posterior
        if t is None:
            t=np.random.uniform(0,self.T)
        self.proposed_index=self.find_position_in_changepoints(t)
        if regime_number is None:
            if not self.infer_regimes:
                regime_number=self.num_regimes
            else:
#                regime_number=np.random.randint(self.num_regimes+1)
                regime_number=np.random.randint(self.num_regimes)
                if regime_number==self.cps[self.proposed_index-1].regime_number:
                    regime_number=self.num_regimes
        self.affected_regimes=[self.cps[self.proposed_index-1].regime_number]
        if regime_number==self.num_regimes:
            revised_regime_number=self.find_position_in_regimes([self.proposed_index])
        else:
            revised_regime_number=regime_number
            self.affected_regimes+=[regime_number]
        self.store_affected_regime_lhds()
        self.revised_affected_regimes=[self.affected_regimes[0]+(1 if (regime_number== self.num_regimes and revised_regime_number<=self.affected_regimes[0]) else 0),revised_regime_number]
        self.add_changepoint(t,regime_number)
        self.calculate_posterior(self.revised_affected_regimes)

    def undo_propose_add_changepoint(self):
        self.delete_changepoint(self.proposed_index)
        for r in self.affected_regimes:
            self.regime_lhds[r]=self.stored_regime_lhds[r]
        self.calculate_posterior(regimes=[])

    def propose_change_regime_of_changepoint(self,index=None,new_regime_number=None):
        self.proposed_index=index if index is not None else (2 if self.num_cps==2 else np.random.randint(2,self.num_cps))#first two cps have regimes 0 and 1 resp.
        regime_number=self.cps[self.proposed_index].regime_number
        if new_regime_number is None:
            blocked_regimes=[regime_number]#,self.cps[self.proposed_index-1].regime_number]
#            if self.proposed_index<self.num_cps:
#                blocked_regimes+=[self.cps[self.proposed_index+1].regime_number]
            new_regime_number=np.random.randint(self.num_regimes-len(blocked_regimes)+1)
            for br in np.sort(np.unique(blocked_regimes)):
                if new_regime_number>=br:
                    new_regime_number+=1

        self.affected_regimes=[regime_number]
        if new_regime_number<self.num_regimes:
            self.affected_regimes+=[new_regime_number]
        self.store_affected_regime_lhds()
        self.revised_affected_regimes=self.change_regime_of_changepoint(self.proposed_index,new_regime_number)
#        print("affected:",self.revised_affected_regimes)
        self.calculate_posterior(self.revised_affected_regimes)

    def undo_propose_change_regime_of_changepoint(self):
        self.change_regime_of_changepoint(self.proposed_index,self.num_regimes if len(self.revised_affected_regimes)==1 else self.revised_affected_regimes[0])
        for r in self.affected_regimes:
            self.regime_lhds[r]=self.stored_regime_lhds[r]
        self.calculate_posterior(regimes=[])

    def store_affected_regime_lhds(self):
        self.stored_regime_lhds={}
        for ar in self.affected_regimes:
            self.stored_regime_lhds[ar]=np.copy(self.regime_lhds[ar])

    def calculate_posterior_means(self):
        self.mean_num_cps=sum([k*self.num_cps_counter[k] for k in self.num_cps_counter])/sum(self.num_cps_counter.values())

    def print_acceptance_rates(self,stream=sys.stderr):
        for m,c in self.proposal_move_counts.most_common():
            stream.write(m+":\t"+str(self.proposal_acceptance_counts[m]/float(c)*100)+"%\n")

    def check_posterior(self):
        if significantly_different(self.posterior,self.calculate_posterior()):
            print(self.iteration,self.move_type,self.mh_accept)
            self.write_changepoints_and_regimes()
            exit()


def significantly_different(x,y,epsilon=1e-5):
    return(np.abs(x-y)>epsilon)
