from probability_model import ProbabilityModel
from data import Data
import numpy as np
import sys
from collections import defaultdict,Counter
from poisson_gamma import PoissonGamma
from regime_model import RegimeModel
from bernoulli_beta import BernoulliBeta
from normal_normal_inverse_gamma import NormalNIG
import itertools

class Changepoint(object):
    probability_models=None
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
        if cp in self.cps:
            return()
        position=self.find_position(cp)
        self.cps.insert(position,cp)

    def remove_cp(self,cp):
        self.cps.remove(cp)

    def replace_cp(self,old_cp,new_cp):
        position=self.cps.index(old_cp)
        self.cps[position]=new_cp

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

    def revert_model_lhd(self,pm_i):
        self.lhds[pm_i]=self.stored_lhds[pm_i]

    def get_total_lhd(self):
        return(sum([self.lhds[pm_i] for pm_i in range(Regime.num_probability_models) if self.model_is_active(pm_i)]))

    def revert_model_lhds(self):
        for pm_i in range(Regime.num_probability_models):
            if self.model_is_active(pm_i):
                self.revert_model_lhd(pm_i)

    def write(self,stream=sys.stdout,delim=" "):
        stream.write(delim.join(map(str,[cp.tau for cp in self.cps]))+"\n")

class ChangepointModel(object):
    def __init__(self,probability_models=np.array([],dtype=ProbabilityModel),infer_regimes=False,disallow_successive_regimes=True,spike_regimes=False):
        self.probability_models=probability_models
        self.num_probability_models=len(self.probability_models)
        Changepoint.probability_models=Regime.probability_models=self.probability_models
        Regime.num_probability_models=self.num_probability_models
        self.T=max([pm.data.get_x_max() for pm in self.probability_models if pm.data is not None])
        self.LOG_T=np.log(self.T)
        self.N=max([pm.data.n for pm in self.probability_models if pm.data is not None])
        self.max_num_changepoints=float("inf")
        self.min_cp_spacing=float(self.T)/self.N
        self.infer_regimes=infer_regimes
        self.changepoint_prior=PoissonGamma(alpha_beta=[.01,10])
        self.inclusion_prior=None if self.num_probability_models==1 else BernoulliBeta(alpha_beta=[.01,10])
        if self.infer_regimes:
            self.regimes_model=RegimeModel(disallow_successive_regimes=disallow_successive_regimes,spike_regimes=spike_regimes)
        self.regime_of_changepoint={}
        self.baseline_changepoint=Changepoint(-float("inf"))
        self.zeroth_regime=Regime([self.baseline_changepoint],np.ones(Regime.num_probability_models,dtype="bool"))
        self.regimes=[self.zeroth_regime]
        self.regime_of_changepoint[self.baseline_changepoint]=self.zeroth_regime
        self.set_changepoints([])
        self.iteration=-1
        self.affected_regime_model_pairs=None
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
        self.undo_proposal_functions["shift_changepoint"]=self.undo_shift_changepoint_proposal
        self.undo_proposal_functions["delete_changepoint"]=self.undo_delete_changepoint_proposal
        self.undo_proposal_functions["add_changepoint"]=self.undo_add_changepoint_proposal
        self.undo_proposal_functions["change_regime"]=self.undo_change_regime_of_changepoint_proposal
        self.undo_proposal_functions["change_regime_inclusion"]=self.undo_change_regime_inclusion_vector_proposal

    def get_changepoint_segment_start_end(self,pm_index,cp):
        cp_index=list(self.cps).index(cp)
        start=cp.data_locations[pm_index]
        next_cp_index=self.get_active_cp_to_right(cp_index,pm_index)
        end=self.cps[next_cp_index].data_locations[pm_index] if next_cp_index<=self.num_cps else self.probability_models[pm_index].data.n
        return((start,end))

    def get_active_cp_to_right(self,cp_index,pm_index=None):
        if pm_index is None:
            return(cp_index+1)
        return(next((i for i in range(cp_index+1,self.num_cps+1) if self.regime_of_changepoint[self.cps[i]].model_is_active(pm_index)),self.num_cps+1))

    def get_active_cp_to_left(self,cp_index,pm_index=None):
        if pm_index is None:
            return(cp_index-1)
        return(next((i for i in range(cp_index-1,-1,-1) if self.regime_of_changepoint[self.cps[i]].model_is_active(pm_index))))

    def distance_to_rh_cp(self,index):
        rh_cp_location=self.T if index==self.num_cps else self.cps[index+1].tau
        return(rh_cp_location-self.cps[index].tau)

    def set_changepoints(self,tau,regime_numbers=None):
        self.num_cps=len(tau)
        self.cps=np.sort(np.array([self.baseline_changepoint]+[Changepoint(t) for t in tau],dtype=Changepoint))
        if regime_numbers is None:
            regime_numbers=list(range(self.num_cps+1))
        else:
            regime_numbers=[0]+list(regime_numbers)
        self.zeroth_regime.cp_indices=np.where(regime_numbers==0)
        self.regimes=[self.zeroth_regime]
        self.num_regimes=max(regime_numbers)+1
        for rn in range(1,self.num_regimes):
            rn_indices=[i for i in range(len(regime_numbers)) if regime_numbers[i]==rn]
            regime_rn=Regime([self.cps[cp_i] for cp_i in rn_indices])
            self.regimes+=[regime_rn]
            for cp in regime_rn.cps:
                self.regime_of_changepoint[cp]=regime_rn

    def find_position_in_changepoints(self,t=None,cp=None):
        if cp is None:
            cp=Changepoint(t)
        position=np.searchsorted(self.cps,cp)
        return(position)

    def find_position_in_regimes(self,regime=None,cps=None):
        if regime is None:
            regime=Regime(cps)
        position=np.searchsorted(self.regimes,regime)
        return(position)

    def order_regimes(self):
        self.regimes.sort()

    def write_changepoints_and_regimes(self,stream=sys.stderr,delim="\t"):
        self.get_regime_numbers()
        stream.write(delim.join([":".join(map(str,(cp.tau,self.regime_numbers[self.regime_of_changepoint[cp]]))) for cp in self.cps])+"\n")

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

    def delete_changepoint(self,index):
        if index==0:
            sys.exit('Error! Cannot delete baseline changepoint')
        cp=self.cps[index]
        regime=self.regime_of_changepoint[cp]
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
        self.proposed_index=index if index is not None else self.find_position_in_changepoints(cp=cp)
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
            if regime not in self.regimes:
                self.add_regime(regime)
        self.regime_of_changepoint[cp]=regime
        self.proposed_regime=regime

    def shift_changepoint(self,index,t):
        self.cps[index].set_location(t)

    def change_regime_of_changepoint(self,cp,from_regime,to_regime):
        if from_regime.length()==1:
            self.delete_regime(from_regime)
        else:
            from_regime.remove_cp(cp)
        self.add_cp_to_regime(cp,to_regime)

    def calculate_likelihood(self,regime_model_pairs=None):
        if self.inclusion_prior is not None:
            regime_model_pairs=None
        if regime_model_pairs is None:
            regime_model_pairs=list(itertools.product(self.regimes,range(self.num_probability_models)))
        for r,pm_i in regime_model_pairs:
            if r.model_is_active(pm_i):
                start_end=[self.get_changepoint_segment_start_end(pm_i,cp) for cp in r.cps]
                r.set_model_lhd(pm_i,self.probability_models[pm_i].likelihood(start_end))
        self.lhd=sum([r.get_total_lhd() for r in self.regimes])
        return(self.lhd)

    def get_effective_changepoint_locations(self):
        self.get_regime_numbers()
        regime=self.zeroth_regime
        effective_cps=[]
        effective_regime_numbers=[]
        for cp in self.cps:
            regime_cp=self.regime_of_changepoint[cp]
            if regime_cp!=regime and regime_cp.is_active():
                regime=regime_cp
                effective_cps+=[cp.tau]
                effective_regime_numbers+=[self.regime_numbers[regime]]

        return(np.array(effective_cps),np.array(effective_regime_numbers),self.get_inclusion_vectors_as_matrix())

    def get_regime_numbers(self):
        self.order_regimes()
        self.regime_numbers={}
        for r_i in range(self.num_regimes):
            self.regime_numbers[self.regimes[r_i]]=r_i

    def get_inclusion_vectors_as_matrix(self):
        inclusion_vectors_mx=np.zeros([self.num_probability_models,self.num_regimes],dtype=int)
        for r_i in range(self.num_regimes):
            inclusion_vectors_mx[:,self.regime_numbers[self.regimes[r_i]]]=self.regimes[r_i].inclusion_vector

        return(inclusion_vectors_mx)

    def calculate_prior(self):
        self.prior=self.changepoint_prior.likelihood(y=self.num_cps)
        if self.num_cps>0 and min([self.distance_to_rh_cp(i) for i in range(self.num_cps+1)])<self.min_cp_spacing:
            self.prior=-float("inf")
        if self.infer_regimes:
            self.regime_sequence,self.regime_sequence_inverse=create_numbering([self.regime_of_changepoint[self.cps[i]] for i in range(self.num_cps+1)])
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
        self.posterior=self.prior+self.lhd
        return(self.posterior)

    def mcmc(self,iterations=20,burnin=0,seed=None,hill_climbing=False):
        if seed is not None:
            np.random.seed(seed)
        self.hill_climbing=hill_climbing
        for self.iteration in range(-burnin,iterations):
            self.mcmc_refresh()
            self.propose_move()
            self.accept_reject()
            if self.mh_accept:
                self.finalise_move()
            else:
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
        self.affected_regime_model_pairs=set()
        self.stored_tau=None
        self.proposal_ratio=0
        self.stored_posterior=self.posterior

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

    def finalise_move(self):
        if self.move_type=="delete_changepoint":
            del self.regime_of_changepoint[self.stored_cp]

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
        if self.hill_climbing and self.iteration>=0:
            self.mh_accept=self.posterior>=self.stored_posterior
        if self.mh_accept:
            self.proposal_acceptance_counts[self.move_type]+=1
#        else:
#            self.undo_move()

    def randomly_select_cp_index(self):
        return(1 if self.num_cps==1 else np.random.randint(1,self.num_cps+1))

    def undo_lhd_changes(self):
        regime_model_pairs=self.affected_regime_model_pairs
        if regime_model_pairs is None:
            regime_model_pairs=set(itertools.product(self.regimes,range(self.num_probability_models)))
        for r,pm_i in regime_model_pairs:
            r.revert_model_lhd(pm_i)

    def propose_shift_changepoint(self,cp_index=None):
        self.proposed_index=cp_index if cp_index is not None else self.randomly_select_cp_index()
        self.stored_cp=self.cps[self.proposed_index]
        affected_regime=self.regime_of_changepoint[self.stored_cp]
        for pm_i in range(self.num_probability_models):
            if affected_regime.model_is_active(pm_i):
                self.affected_regime_model_pairs.add((affected_regime,pm_i))
                left_regime=self.regime_of_changepoint[self.cps[self.get_active_cp_to_left(self.proposed_index,pm_i)]]
                self.affected_regime_model_pairs.add((left_regime,pm_i))

        left_boundary=self.cps[self.proposed_index-1].tau if self.proposed_index>1 else 0
        right_boundary=self.cps[self.proposed_index+1].tau if self.proposed_index<self.num_cps else self.T
        t=np.random.uniform(left_boundary,right_boundary)
        regime=self.regime_of_changepoint[self.cps[self.proposed_index]]
        self.cps[self.proposed_index]=Changepoint(t)
        regime.replace_cp(self.stored_cp,self.cps[self.proposed_index])
        self.regime_of_changepoint[self.cps[self.proposed_index]]=regime
        self.calculate_posterior(self.affected_regime_model_pairs)

    def undo_shift_changepoint_proposal(self):
        del self.regime_of_changepoint[self.cps[self.proposed_index]]
        self.regime_of_changepoint[self.stored_cp].replace_cp(self.cps[self.proposed_index],self.stored_cp)
        self.cps[self.proposed_index]=self.stored_cp
        self.undo_lhd_changes()
        self.calculate_posterior(regime_model_pairs=[])

    def propose_delete_changepoint(self,index=None):
        self.proposed_index=index if index is not None else self.randomly_select_cp_index()
        self.stored_cp=self.cps[self.proposed_index]
        self.stored_regime=self.regime_of_changepoint[self.stored_cp]
        for pm_i in range(self.num_probability_models):
            if self.stored_regime.model_is_active(pm_i):
                if self.stored_regime.length()>1:
                    self.affected_regime_model_pairs.add((self.stored_regime,pm_i))
                cp_index=self.get_active_cp_to_left(self.proposed_index,pm_i)
                self.affected_regime_model_pairs.add((self.regime_of_changepoint[self.cps[cp_index]],pm_i))

#        self.proposal_ratio-=self.LOG_T
        self.delete_changepoint(self.proposed_index)
        if self.infer_regimes:
            self.regime_sequence,self.regime_sequence_inverse=create_numbering([self.regime_of_changepoint[self.cps[i]] for i in range(self.num_cps+1)])

            dummy_regime_number,regime_log_proposal=self.regimes_model.propose_regime(self.num_regimes,self.regime_sequence[self.proposed_index-1],None if self.proposed_index==self.num_cps+1 else self.regime_sequence[self.proposed_index])
            self.proposal_ratio+=regime_log_proposal
        self.calculate_posterior(self.affected_regime_model_pairs)

    def undo_delete_changepoint_proposal(self):
        self.add_changepoint(cp=self.stored_cp,index=self.proposed_index,regime=self.stored_regime)
        self.undo_lhd_changes()
        self.calculate_posterior(regime_model_pairs=[])

    def propose_add_changepoint(self,t=None,regime_number=None):
        if t is None:
            t=np.random.uniform(0,self.T)
#            self.proposal_ratio+=self.LOG_T
        self.proposed_index=self.find_position_in_changepoints(t)
        if regime_number is None:
            if not self.infer_regimes:
                regime_number=self.num_regimes
            else:
#                regime_number=np.random.randint(self.num_regimes+1)
                self.regime_sequence,self.regime_sequence_inverse=create_numbering([self.regime_of_changepoint[self.cps[i]] for i in range(self.num_cps+1)])
                regime_number,regime_log_proposal=self.regimes_model.propose_regime(self.num_regimes,self.regime_sequence[self.proposed_index-1],None if self.proposed_index==self.num_cps+1 else self.regime_sequence[self.proposed_index])
                self.proposal_ratio-=regime_log_proposal

        regime=None if regime_number==self.num_regimes else self.regime_sequence_inverse[regime_number]
        self.add_changepoint(t,regime,index=self.proposed_index)
        for pm_i in range(self.num_probability_models):
            if self.proposed_regime.model_is_active(pm_i):
                self.affected_regime_model_pairs.add((self.proposed_regime,pm_i))
                cp_index=self.get_active_cp_to_left(self.proposed_index,pm_i)
                left_regime=self.regime_of_changepoint[self.cps[cp_index]]
                self.affected_regime_model_pairs.add((left_regime,pm_i))
        self.calculate_posterior(self.affected_regime_model_pairs)

    def undo_add_changepoint_proposal(self):
        cp=self.cps[self.proposed_index]
        self.delete_changepoint(self.proposed_index)
        del self.regime_of_changepoint[cp]
        self.undo_lhd_changes()
        self.calculate_posterior(regime_model_pairs=[])

    def propose_change_regime_of_changepoint(self,index=None,new_regime_number=None):
        self.proposed_index=index if index is not None else (2 if self.num_cps==2 else np.random.randint(2,self.num_cps+1))#first two cps have regimes 0 and 1 resp.
        self.from_regime=self.regime_of_changepoint[self.cps[self.proposed_index]]
        from_will_close=self.from_regime.length()==1
        self.regime_sequence,self.regime_sequence_inverse=create_numbering([self.regime_of_changepoint[self.cps[i]] for i in range(self.num_cps+1)])
        regime_number=self.regime_sequence[self.proposed_index]
        if new_regime_number is None:
            new_regime_number,regime_log_proposal=self.regimes_model.propose_regime(self.num_regimes,self.regime_sequence[self.proposed_index-1],None if self.proposed_index==self.num_cps else self.regime_sequence[self.proposed_index+1],self.regime_sequence[self.proposed_index],from_will_close)
        if new_regime_number is None:
            mh_accept=False
            return()
        self.proposed_regime=None if new_regime_number==self.num_regimes else self.regime_sequence_inverse[new_regime_number]

        self.change_regime_of_changepoint(self.cps[self.proposed_index],self.from_regime,self.proposed_regime)
        for pm_i in range(self.num_probability_models):
            from_active=not from_will_close and self.from_regime.model_is_active(pm_i)
            to_active=self.proposed_regime.model_is_active(pm_i)
            if from_active:
                self.affected_regime_model_pairs.add((self.from_regime,pm_i))
            if to_active:
                self.affected_regime_model_pairs.add((self.proposed_regime,pm_i))
            if from_active!=to_active:
                cp_index=self.get_active_cp_to_left(self.proposed_index,pm_i)
                self.affected_regime_model_pairs.add((self.regime_of_changepoint[self.cps[cp_index]],pm_i))
        self.calculate_posterior(self.affected_regime_model_pairs)

    def undo_change_regime_of_changepoint_proposal(self):
        self.change_regime_of_changepoint(self.cps[self.proposed_index],self.proposed_regime,self.from_regime)
        self.undo_lhd_changes()
        self.calculate_posterior(regime_model_pairs=[])

    def propose_change_regime_inclusion_vector(self,regime_number=None,pm_index=None):
        self.proposed_regime_number=np.random.randint(1,self.num_regimes) if regime_number is None else regime_number
        self.proposed_pm_index=np.random.randint(self.num_probability_models) if pm_index is None else pm_index
        self.regimes[self.proposed_regime_number].inclusion_vector_flip_position(self.proposed_pm_index)
        self.calculate_posterior()

    def sample_inclusion_vector(self):
        return(np.array(self.inclusion_prior.simulate_data(self.num_probability_models),dtype="bool")[0])

    def undo_change_regime_inclusion_vector_proposal(self):
        self.regimes[self.proposed_regime_number].inclusion_vector_flip_position(self.proposed_pm_index)
        self.undo_lhd_changes()
        self.calculate_posterior(regime_model_pairs=[])

    def calculate_posterior_means(self):
        self.mean_num_cps=sum([k*self.num_cps_counter[k] for k in self.num_cps_counter])/sum(self.num_cps_counter.values())

    def print_acceptance_rates(self,stream=sys.stderr):
        for m,c in self.proposal_move_counts.most_common():
            a=self.proposal_acceptance_counts[m]
            stream.write(m+":\t"+str(a)+"/"+str(c)+"\t"+str(a/float(c)*100)+"%\n")

    def check_posterior(self):
        store=self.posterior
        if significantly_different(self.posterior,self.calculate_posterior()):
            print(self.iteration,self.move_type,self.mh_accept)
            print(store,"!=",self.posterior)
            self.write_changepoints_and_regimes()
            self.print_acceptance_rates()
            exit()

    def changepoint_vector_precision_recall(self,cp_locations):
        return(changepoint_vectors_precision_recall([cp.tau for cp in self.cps],cp_locations))

def significantly_different(x,y,epsilon=1e-5):
    return(np.abs(x-y)>epsilon)

def changepoint_vectors_precision_recall(x,y,window=.1):
    n_x,n_y=len(x),len(y)
    n_x_in_y=0
    j=0
    for i in range(n_x):
        while j<n_y and y[j]<x[i]-window:
            j+=1
        if j<n_y and y[j]<=x[i]+window:
            n_x_in_y+=1
            j+=1

    rate_x=n_x_in_y/float(n_x) if n_x>0 else 1
    n_y_in_x=0
    j=0
    for i in range(n_y):
        while j<n_x and x[j]<y[i]-window:
            j+=1
        if j<n_x and x[j]<=y[i]+window:
            n_y_in_x+=1
            j+=1

    rate_y=n_y_in_x/float(n_y) if n_y>0 else 1
    return(rate_x,rate_y)

def create_numbering(x):
    if len(x)==0:
        return([])
    z=np.zeros(len(x),dtype=int)
    x_vals={}
    x_vals[x[0]]=0
    z_vals=[x[0]]
    for i in range(1,len(x)):
        if x[i] not in x_vals:
            x_vals[x[i]]=len(z_vals)
            z_vals+=[x[i]]
        z[i]=x_vals[x[i]]

    return(z,z_vals)

