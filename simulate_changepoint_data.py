#! /usr/bin/env python3
import numpy as np
from data import Data
from probability_model import ProbabilityModel
from poisson_gamma import PoissonGamma
from regime_model import RegimeModel
from multinomial_dirichlet import MultinomialDirichlet
from normal_normal_inverse_gamma import NormalNIG
from collections import defaultdict
import sys

def simulate_changepoints_and_regimes(n=1000,lambda_cps=.01,n_cps=None,n_prob_models=1,inclusion_ps=.8,seed=None,tau_filename=None):
    if seed is not None:
        np.random.seed(seed)
    if n_cps is None:
        n_cps=np.random.poisson(n*lambda_cps) # 1/lambda_cps parameter is the prior expected segment size between changepoints
    tau=np.sort(np.random.uniform(size=n_cps))
    regimes=RegimeModel().sample_parameter(n_cps+1)
    n_regimes=max(regimes)+1
    inclusion_ps=inclusion_ps if hasattr(inclusion_ps, "__len__") else np.repeat(inclusion_ps,n_prob_models)
    inclusion_vectors=np.zeros([n_prob_models,n_regimes],dtype=int)
    inclusion_vectors[:,0]=1 #zeroth regime applies to all models
    for j in range(1,n_regimes):
        while sum(inclusion_vectors[:,j])==0:
            inclusion_vectors[:,j]=[np.random.binomial(1,inclusion_ps[i]) for i in range(n_prob_models)]

    if tau_filename!=None:
        np.savetxt(tau_filename,tau,delimiter=",",fmt='%1.3f',newline=" ")
    return(tau,regimes,inclusion_vectors)

def simulate_parameters(inclusion_vectors,prob_models):
    regime_parameters=defaultdict(dict)
    for i in range(len(prob_models)):
        for j in range(inclusion_vectors.shape[1]):
            if inclusion_vectors[i,j]:
                regime_parameters[prob_models[i]][j]=prob_models[i].sample_parameter()

    return(regime_parameters)

def simulate_changepoint_data(n,inclusion_vector=[1],tau=[],regimes=[0],num_categories=np.array([3,5],dtype=int),seed=None,model=None,theta_map=None,y_filename=None,x_filename=None):
    if seed is not None:
        np.random.seed(seed)
    n_cps=len(tau)
    tau_positions=np.ceil(tau*(n-1)).astype(int)
    p=model.get_dimension()
    y=np.empty((p,n),dtype=model.data_type)
    start=0
    for i in range(n_cps+1):
        end=tau_positions[i] if i<n_cps else n
        r_i=regimes[i]
        if r_i in theta_map:
            thetas=theta_map[r_i]
        if end>start: #else segment empty
#            print(start,end,model.simulate_data(end-start,thetas))
#            print('**',y[:,start:end])
            y[:,start:end]=model.simulate_data(end-start,thetas)
            start=end

    if y_filename!=None:
        np.savetxt(y_filename,y,fmt='%i' if model!="normal" else '%.18e',delimiter=",")
    if x_filename!=None:
        np.savetxt(x_filename,np.arange(n)/float(n-1),delimiter=",")

def main():
    n=1000 if len(sys.argv)<2 else int(sys.argv[1])
    s=0 if len(sys.argv)<3 else int(sys.argv[2])
    probability_models=[MultinomialDirichlet(k=np.array([3,5],dtype=int),alpha=1),NormalNIG(p=3,alpha_beta=[.1,.1],v=1)]#PoissonGamma(p=3,alpha_beta=[1,10])]
    cps,regimes,inclusion_vectors=simulate_changepoints_and_regimes(n=n,n_cps=5,n_prob_models=len(probability_models),inclusion_ps=.8,seed=None,tau_filename="tau.txt")
    theta_maps=simulate_parameters(inclusion_vectors,probability_models)
    for i in range(len(probability_models)):
        pm=probability_models[i]
        file_ending="_"+str(i)+".txt"
        simulate_changepoint_data(n=n,y_filename="y"+file_ending,x_filename="x"+file_ending,seed=s,tau=cps,regimes=regimes,model=pm,theta_map=theta_maps[pm])##"multinomial")#normal")

main()
