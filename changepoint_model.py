from probability_model import ProbabilityModel
from data import Data
import numpy as np
import sys
from collections import defaultdict,Counter
from poisson_gamma import PoissonGamma
from regime_model import RegimeModel
from bernoulli_beta import BernoulliBeta
import itertools

class Changepoint(object):
    def __init__(self,t):
        self.tau=t
        if Changepoint.probability_models is not None:
            self.find_data_positions()

    def find_data_positions(self,t=None):
        t=t if t is not None else self.tau
        self.data_locations=np.array([pm.find_data_position(t) for pm in Regime.probability_models],dtype=int)

    def set_location(self,t):
        self.tau=t
        if Changepoint.probability_models is not None:
            self.find_data_positions()

    def __lt__(self,other):
        return(self.tau<other.tau)

class Regime(object):
    probability_models=None
    num_probability_models=0
    def __init__(self,cps,inclusion_vector=None,lhds=None):
        self.cps=cps
        self.inclusion_vector=inclusion_vector if inclusion_vector is not None else np.ones(Regime.num_probability_models,dtype="bool")
        self.lhds=lhds if lhds is not None else np.zeros(Regime.num_probability_models,dtype="float")
        self.stored_lhds=np.copy(self.lhds)

    def __lt__(self,other):#find which regime occurs first
        return(self.cps[0]<other.cps[0])

    def find_position(self,cp):
        return(np.searchsorted(self.cps,cp))

    def insert_cp(self,cp):
        position=self.find_position(cp)
        self.cps.insert(position,cp)

    def remove_cp(self,cp):
        self.cps.remove(cp)

    def length(self):
        return(len(self.cps))

    def model_is_active(self,pm_i):
        return(self.inclusion_vector[pm_i] if self.inclusion_vector is not None else True)

    def is_active(self):
        return(any(self.inclusion_vector) if self.inclusion_vector is not None else True)

    def inclusion_vector_flip_position(self,pm_i):
        self.inclusion_vector[pm_i]=not self.inclusion_vector[pm_i]

    def get_inclusion_counts(self):
        if self.inclusion_vector is None:
            return(None)
        n_active=np.count_nonzero(self.inclusion_vector)
        return([Regime.num_probability_models-n_active,n_active])

    def set_model_lhd(self,pm_i,val):
        self.stored_lhds[pm_i]=self.lhds[pm_i]
        self.lhds[pm_i]=val

    def get_total_lhd(self):
        return(sum([self.lhds[pm_i] for pm_i in range(Regime.num_probability_models) if self.model_is_active(pm_i)]))

    def revert_model_lhds(self):
        for pm_i in range(Regime.num_probability_models):
            if self.model_is_active(pm_i):
                self.lhds[pm_i]=self.stored_lhds[pm_i]

    def write(self,stream=sys.stdout,delim=" "):
        stream.write(delim.join(map(str,[cp.tau for cp in self.cps]))+"\n")

class ChangepointModel(object):
    def __init__(self,probability_models=np.array([],dtype=ProbabilityModel),infer_regimes=False,disallow_successive_regimes=True,spike_regimes=False):
        self.probability_models=probability_models
        self.num_probability_models=len(self.probability_models)
        Regime.probability_models=self.probability_models
        Regime.num_probability_models=self.num_probability_models
        self.T=max([pm.data.get_x_max() for pm in self.probability_models if pm.data is not None])
        self.LOG_T=np.log(self.T)
        self.N=max([pm.data.n for pm in self.probability_models if pm.data is not None])
        self.max_num_changepoints=float("inf")
        self.min_cp_spacing=float(self.T)/self.N
        self.infer_regimes=infer_regimes
        self.changepoint_prior=PoissonGamma(alpha_beta=[.01,10])
        self.inclusion_prior=None#BernoulliBeta(alpha_beta=[.01,10])
        if self.infer_regimes:
            self.regimes_model=RegimeModel(disallow_successive_regimes=disallow_successive_regimes,spike_regimes=spike_regimes)
        self.regime_of_changepoint={}
        self.zeroth_regime=Regime([0],np.ones(Regime.num_probability_models,dtype="bool"))
        self.regimes=[self.zeroth_regime]
        self.baseline_changepoint=Changepoint(-float("inf"))
        self.regime_of_changepoint[self.baseline_changepoint]=self.zeroth_regime
        self.set_changepoints([])
        self.affected_regimes=self.revised_affected_regimes=None
        self.move_type=None
        self.calculate_posterior()#calculate likelihoods for each prob. model
        self.create_mh_dictionary()
        self.proposal_move_counts=Counter()
        self.proposal_acceptance_counts=Counter()
        self.num_cps_counter=Counter()

    def create_mh_dictionary(self):
        self.proposal_functions={}
        self.proposal_functions["shift_changepoint"]=self.propose_shift_changepoint
        self.proposal_functions["delete_changepoint"]=self.propose_delete_changepoint
        self.proposal_functions["add_changepoint"]=self.propose_add_changepoint
        self.proposal_functions["change_regime"]=self.propose_change_regime_of_changepoint
        self.proposal_functions["change_regime_inclusion"]=self.propose_change_regime_inclusion_vector
        self.undo_proposal_functions={}
        self.undo_proposal_functions["shift_changepoint"]=self.undo_propose_shift_changepoint
        self.undo_proposal_functions["delete_changepoint"]=self.undo_propose_delete_changepoint
        self.undo_proposal_functions["add_changepoint"]=self.undo_propose_add_changepoint
        self.undo_proposal_functions["change_regime"]=self.undo_propose_change_regime_of_changepoint
        self.undo_proposal_functions["change_regime_inclusion"]=self.undo_propose_change_regime_inclusion_vector

    def get_changepoint_segment_start_end(self,pm_index,cp):
        cp_index=self.cps.index(cp)
        start=cp.data_locations[pm_index]
        next_cp_index=self.get_active_cp_to_right(cp_index,pm_index)
        end=self.cps[next_cp_index].data_locations[pm_index] if next_cp_index<=self.num_cps else self.probability_models[pm_index].data.n
        return((start,end))

    def get_active_cp_to_right(self,cp_index,pm_index=None):
        if pm_index is None:
            return(cp_index+1)
        return(next((i for i in range(cp_index+1,self.num_cps+1) if self.cps[i].regime.model_is_active(pm_index)),self.num_cps+1))

    def get_active_cp_to_left(self,cp_index,pm_index=None):
        if pm_index is None:
            return(cp_index-1)
        return(next((i for i in range(cp_index-1,-1,-1) if self.cps[i].regime.model_is_active(pm_index))))

    def distance_to_rh_cp(self,index):
        rh_cp_location=self.T if index==self.num_cps else self.cps[index+1].tau
        return(rh_cp_location-self.cps[index].tau)

    def set_changepoints(self,tau,regimes_numbers=None):
        self.num_cps=len(tau)
        self.cps=np.sort(np.array([self.baseline_changepoint]+[Changepoint(t) for t in tau],dtype=Changepoint))
        if regime_numbers is None:
            regime_numbers=range(self.num_cps+1)
        else:
            regime_numbers=[0]+list(regime_numbers)
        zeroth.regime.cp_indices=numpy.where(regime_numbers==0)
        self.regimes=[self.zeroth_regime]
        self.num_regimes=max(regime_numbers)+1
        for rn in range(1,self.num_regimes):
            rn_indices=numpy.where(regime_numbers==rn)
            regime_rn=Regime([self.cps[cp_i] for cp_i in rn_indices])
            self.regimes+=[regime_rn]
            for cp in regime_rn:
                regime_of_changepoint[cp]=regime_rn

    def find_position_in_changepoints(self,t=None,cp=None):
        if cp is None:
            cp=Changepoint(t)
        position=np.searchsorted(self.cps,cp))
        return(position)

    def find_position_in_regimes(self,cps=None,regime=None):
        if regime is None:
            regime=Regime(cps)
        position=np.searchsorted(self.regimes,regime)
        return(position)

    def order_regimes(self):
        self.regimes=np.sort(self.regimes)

    def write_changepoints_and_regimes(self,stream=sys.stderr,delim="\t"):
        self.order_regimes()
        stream.write(delim.join([":".join(map(str,(cp.tau,np.searchsorted(self.regimes,cp.regime)))) for cp in self.cps])+"\n")

    def write_regime_inclusion_vectors(self,stream=sys.stderr,delim="\t"):
        self.order_regimes()
        try:
            stream.write(delim.join([str(r_i)+":"+",".join(map(str,self.regimes[r_i].inclusion_vector)) for r_i in range(self.num_regimes)])+"\n")
        except:
            None

    def write_regimes(self,stream=sys.stdout,delim=" "):
        for r in self.regimes:
            r.write(stream,delim)

    def delete_regime(self,regime):
        self.deleted_regime=regime
        self.regimes.remove(regime)
        self.num_regimes-=1

    def add_regime(self,regime):
        self.added_regime=regime
        self.regimes.insert(self.find_position_in_regimes(regime),regime)
        self.num_regimes+=1

    def add_changepoint_to_regime(self,cp,regime_number):
        if regime_number==self.num_regimes:
            if self.inclusion_prior is not None:
                inclusion_vector=self.sample_inclusion_vector()
            self.new_regime=Regime([cp])
            self.add_regime(self.new_regime)
            regime_of_changepoint[cp]=self.new_regime
        else:
            regime_number=self.update_and_reposition_regimes(index,None,regime_number)[1]
        return(regime_number)

    def delete_changepoint(self,index):
        if index==0:
            sys.exit('Error! Cannot delete baseline changepoint')
        cp=self.cps[index]
        regime=regime_of_changepoint[cp]
        if regime.length()==1:
            self.delete_regime(regime)
        else:
            regime.cps.remove(cp)
        self.stored_changepoint=cp
        self.cps=np.delete(self.cps,index)
        self.num_cps-=1

    def add_changepoint(self,t=None,regime=None,cp=None,index=None):
        if cp is None:
            cp=Changepoint(t)
        self.proposed_index=index if index is not None else self.find_position_in_changepoints(cp)
        self.cps=np.insert(self.cps,self.proposed_index,cp)
        self.num_cps+=1
        self.stored_changepoint=cp
        self.add_cp_to_regime(cp,regime)

    def add_cp_to_regime(self,cp,regime=None):
        if regime is None:
            regime=Regime([cp])
            self.add_regime(regime)
        else:
            regime.insert_cp(cp)
        self.regime_of_changepoint[cp]=regime

    def shift_changepoint(self,index,t):
        self.cps[index].set_location(t)

    def change_regime_of_changepoint(self,cp,new_regime=None):
        self.stored_regime=self.regime_of_changepoint[cp]
        if self.stored_regime.length()==1:
            self.delete_regime(self.stored_regime)
        self.add_cp_to_regime(cp,new_regime)

    def calculate_likelihood(self,regime_model_pairs=None):
        if self.inclusion_prior is not None:
            regime_model_pairs=None
        if regime_model_pairs is None:
            regime_model_pairs=list(itertools.product(self.regimes,range(self.num_probability_models)))
        for r,pm_i in regime_model_pairs:
            if r.model_is_active(pm_i):
                start_end=[self.get_changepoint_segment_start_end(pm_i,cp) for cp in r.cps]
                r.set_model_lhd(self,pm_i,self.probability_models[pm_i].likelihood(start_end))
        self.likelihood=sum([r.get_total_lhd() for r in self.regimes])
        return(self.likelihood)

    def get_effective_changepoint_locations(self):
        regime=self.zeroth_regime
        effective_cps=[]
        for i in range(1,self.num_cps+1):
            if self.cps[i].regime!=regime and self.cps[i].regime.is_active():
                regime=self.cps[i].regime
                effective_cps+=[self.cps[i].tau]

        return(effective_cps)

    def calculate_prior(self):
        self.prior=self.changepoint_prior.likelihood(y=self.num_cps)
        if self.num_cps>0 and min([self.distance_to_rh_cp(i) for i in range(self.num_cps+1)])<self.min_cp_spacing:
            self.prior=-float("inf")
        if self.infer_regimes:
            self.regime_sequence=create_numbering([self.cps[i].regime for i in range(self.num_cps+1)])
            regime_prior=self.regimes_model.likelihood(y=self.regime_sequence)
            self.prior+=regime_prior
            if self.inclusion_prior is not None:
                for r in self.regimes:
                    inclusion_counts=r.get_inclusion_counts()
                    if inclusion_counts is not None:
                        self.prior+=self.inclusion_prior.likelihood(y=inclusion_counts)
        return(self.prior)

    def calculate_posterior(self,regime_model_pairs=None):
        self.calculate_prior()
        self.calculate_likelihood(regime_model_pairs=regime_model_pairs)
        self.posterior=self.prior+self.likelihood
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
        self.write_regime_inclusion_vectors()
        sys.stderr.write("Final posterior="+str(self.posterior)+"\n")
        self.print_acceptance_rates()
        self.calculate_posterior_means()
        sys.stderr.write("E[#Changepoints] = "+str(self.mean_num_cps)+"\n")

    def mcmc_refresh(self):
        self.mh_accept=True
        self.deleted_regime_lhd=None
        self.affected_regimes=self.revised_affected_regimes=None
        self.stored_tau=None
        self.proposal_ratio=0

    def choose_move(self):
        self.get_available_move_types()
        self.move_type=np.random.choice(self.available_move_types)
        self.proposal_ratio=np.log(len(self.available_move_types))

    def get_available_move_types(self):
        self.available_move_types=[]
        if self.num_cps>0:
            self.available_move_types.append("shift_changepoint")
            self.available_move_types.append("delete_changepoint")
        if self.num_cps<self.max_num_changepoints:
            self.available_move_types.append("add_changepoint")
        if self.infer_regimes and self.num_cps>1:
            self.available_move_types.append("change_regime")
        if self.infer_regimes and self.inclusion_prior is not None and self.num_regimes>1:
            self.available_move_types.append("change_regime_inclusion")

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
        self.get_available_move_types()
        self.proposal_ratio-=np.log(len(self.available_move_types))
        self.accpeptance_ratio=self.proposal_ratio+self.posterior-self.stored_posterior
        if self.accpeptance_ratio<0 and np.random.exponential()<-self.accpeptance_ratio:
            self.mh_accept=False
        if self.mh_accept:
            self.proposal_acceptance_counts[self.move_type]+=1
#        else:
#            self.undo_move()

    def propose_shift_changepoint(self,index=None):
        self.proposed_index=index if index is not None else (1 if self.num_cps==1 else np.random.randint(1,self.num_cps+1))
        self.stored_cp=self.cps[self.proposed_index]
        left_boundary=self.cps[self.proposed_index-1].tau if self.proposed_index>1 else 0
        right_boundary=self.cps[self.proposed_index+1].tau if self.proposed_index<self.num_cps else self.T
        self.revised_affected_regimes=self.affected_regimes=[self.cps[self.proposed_index-1].regime_number,self.cps[self.proposed_index].regime_number]
        self.store_affected_regime_lhds()
        t=np.random.uniform(left_boundary,right_boundary)
        self.cps[self.proposed_index]=Changepoint(t,regime_number=self.cps[self.proposed_index].regime_number)
        self.calculate_posterior(self.revised_affected_regimes)

    def undo_propose_shift_changepoint(self):
        self.cps[self.proposed_index]=self.stored_cp
        for r in self.affected_regimes:
            self.regime_lhds[r]=self.stored_regime_lhds[r]
        self.calculate_posterior(regimes=[])

    def propose_delete_changepoint(self,index=None):
        self.proposed_index=index if index is not None else (1 if self.num_cps==1 else 1+np.random.randint(self.num_cps))
        self.stored_cp=self.cps[self.proposed_index]
        self.stored_num_regimes=self.num_regimes
        self.stored_posterior=self.posterior
        self.stored_regime_lhds={}
        regime_will_vanish=False
        self.affected_regimes=[self.cps[self.proposed_index-1].regime_number,self.cps[self.proposed_index].regime_number]
        self.stored_regime_lhds[self.affected_regimes[0]]=np.copy(self.regime_lhds[self.affected_regimes[0]])
        if self.regimes[self.affected_regimes[1]].length()==1:#regime will be deleted
            regime_will_vanish=True
            self.stored_regime_lhds[self.affected_regimes[1]]=self.regime_lhds[self.affected_regimes[1]]
            if self.affected_regimes[1]<self.affected_regimes[0]:
                self.revised_affected_regimes=[self.affected_regimes[0]-1]
            else:
                self.revised_affected_regimes=[self.affected_regimes[0]]
        else:
            self.revised_affected_regimes=self.affected_regimes[:]
            self.stored_regime_lhds[self.affected_regimes[1]]=np.copy(self.regime_lhds[self.affected_regimes[1]])
#        self.proposal_ratio-=self.LOG_T
        if self.infer_regimes:
            dummy_regime_number,regime_log_proposal=self.regimes_model.propose_regime(self.num_regimes-(1 if regime_will_vanish else 0),self.cps[self.proposed_index-1].regime_number,None if self.proposed_index==self.num_cps else self.cps[self.proposed_index+1].regime_number)
            self.proposal_ratio+=regime_log_proposal
        self.delete_changepoint(self.proposed_index)
        self.calculate_posterior(self.revised_affected_regimes)

    def undo_propose_delete_changepoint(self):
        regime=self.num_regimes if self.stored_num_regimes>self.num_regimes else self.stored_cp.regime_number
        self.add_changepoint(cp=self.stored_cp,index=self.proposed_index)#self.stored_cp.tau,regime)
        self.stored_num_regimes=self.num_regimes
        for r in self.affected_regimes:
            self.regime_lhds[r]=self.stored_regime_lhds[r]
        self.calculate_posterior(regimes=[])

    def propose_add_changepoint(self,t=None,regime_number=None):
        self.stored_num_regimes=self.num_regimes
        self.stored_posterior=self.posterior
        if t is None:
            t=np.random.uniform(0,self.T)
#            self.proposal_ratio+=self.LOG_T
        self.proposed_index=self.find_position_in_changepoints(t)
        if regime_number is None:
            if not self.infer_regimes:
                regime_number=self.num_regimes
            else:
#                regime_number=np.random.randint(self.num_regimes+1)
                regime_number,regime_log_proposal=self.regimes_model.propose_regime(self.num_regimes,self.cps[self.proposed_index-1].regime_number,None if self.proposed_index==self.num_cps+1 else self.cps[self.proposed_index].regime_number)
                self.proposal_ratio-=regime_log_proposal
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
        self.proposed_index=index if index is not None else (2 if self.num_cps==2 else np.random.randint(2,self.num_cps+1))#first two cps have regimes 0 and 1 resp.
        regime_number=self.cps[self.proposed_index].regime_number
        if new_regime_number is None:
            new_regime_number,regime_log_proposal=self.regimes_model.propose_regime(self.num_regimes,self.cps[self.proposed_index-1].regime_number,None if self.proposed_index==self.num_cps else self.cps[self.proposed_index+1].regime_number,regime_number)

        self.affected_regimes=[regime_number]
        if new_regime_number<self.num_regimes:#not creating a new regime
            self.affected_regimes+=[new_regime_number]
        self.store_affected_regime_lhds()
        self.revised_affected_regimes=self.change_regime_of_changepoint(self.proposed_index,new_regime_number)
        self.calculate_posterior(self.revised_affected_regimes)

    def undo_propose_change_regime_of_changepoint(self):
        copy_affected_regimes=self.affected_regimes[:]
        self.affected_regimes,self.revised_affected_regimes=self.revised_affected_regimes,self.affected_regimes
        self.change_regime_of_changepoint(self.proposed_index,self.num_regimes if len(self.affected_regimes)==1 else self.affected_regimes[0])
        self.affected_regimes=[r for r in copy_affected_regimes if r is not None]
        self.recover_affected_regime_lhds()
        self.calculate_posterior(regimes=[])

    def propose_change_regime_inclusion_vector(self,regime_number=None,pm_index=None):
        self.proposed_regime_number=np.random.randint(1,self.num_regimes) if regime_number is None else regime_number
        self.proposed_pm_index=np.random.randint(self.num_probability_models) if pm_index is None else pm_index
        self.regimes[self.proposed_regime_number].inclusion_vector_flip_position(self.proposed_pm_index)
        self.revised_affected_regimes=self.affected_regimes=range(self.num_regimes)
        self.store_affected_regime_lhds()
        self.calculate_posterior()

    def sample_inclusion_vector(self):
        return(np.array(self.inclusion_prior.simulate_data(self.num_probability_models),dtype="bool")[0])

    def undo_propose_change_regime_inclusion_vector(self):
        self.regimes[self.proposed_regime_number].inclusion_vector_flip_position(self.proposed_pm_index)
        self.recover_affected_regime_lhds()
        self.calculate_posterior(regimes=[])

    def store_affected_regime_lhds(self):
        self.stored_regime_lhds={}
        for ar in self.affected_regimes:
            for pm_i in range(self.num_probability_models):
                self.stored_regime_lhds[ar,pm_i]=self.regime_lhds[ar,pm_i]

    def recover_affected_regime_lhds(self):
        for ar in self.affected_regimes:
            self.regime_lhds[ar]=self.stored_regime_lhds[ar]

    def calculate_posterior_means(self):
        self.mean_num_cps=sum([k*self.num_cps_counter[k] for k in self.num_cps_counter])/sum(self.num_cps_counter.values())

    def print_acceptance_rates(self,stream=sys.stderr):
        for m,c in self.proposal_move_counts.most_common():
            a=self.proposal_acceptance_counts[m]
            stream.write(m+":\t"+str(a)+"/"+str(c)+"\t"+str(a/float(c)*100)+"%\n")

    def check_posterior(self):
        if significantly_different(self.posterior,self.calculate_posterior()):
            print(self.iteration,self.move_type,self.mh_accept)
            self.write_changepoints_and_regimes()
            self.print_acceptance_rates()
            exit()

def significantly_different(x,y,epsilon=1e-5):
    return(np.abs(x-y)>epsilon)

def create_numbering(x):
    if len(x)==0:
        return([])
    z=np.zeros(len(x),dtype=int)
    x_vals={}
    x_vals[x[0]]=0
    for i in range(1,len(x)):
        if x[i] not in x_vals:
            x_vals[x_i]=len(x_vals)
        z[i]=x_vals[x[i]]

    return(z)

