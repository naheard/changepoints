#! /usr/bin/env python3
import numpy as np
from data import Data
from poisson_gamma import PoissonGamma
from regime_model import RegimeModel
from multinomial_dirichlet import MultinomialDirichlet
from normal_normal_inverse_gamma import NormalNIG
import sys

def simulate_segment_data(n,k,alpha=1,theta=None,model="multinomial"):
    if model=="multinomial":
        if theta is None:
            theta=np.random.dirichlet(np.repeat(alpha,k))
        y=np.random.choice(int(k),int(n),p=theta)
    if model=="normal":
        m=NormalNIG(p=1)
        if theta is None:
            theta=m.sample_parameter()
        print(theta)
        y=m.simulate_data(n=n,mu=theta[0],sigma=theta[1])
    return(y)

def simulate_changepoints_and_regimes(n=1000,n_cps=5,inclusion_ps=np.repeat(0.3*0+1,2),seed=None,tau_filename=None):
    if seed is not None:
        np.random.seed(seed)
    tau=np.sort(np.random.uniform(size=n_cps))
    regimes=RegimeModel().sample_parameter(n_cps+1)
    if tau_filename!=None:
        np.savetxt(tau_filename,tau,delimiter=",",fmt='%1.3f',newline=" ")
    return(tau,regimes)

def simulate_changepoint_data(n=1000,p=2,tau=[],regimes=[0],num_categories=np.array([3,5],dtype=int),inclusion_ps=np.repeat(0.3*0+1,2),seed=None,model="multinomial",y_filename=None,x_filename=None):
    if seed is not None:
        np.random.seed(seed)
    n_cps=len(tau)
    tau_indicators=np.array([np.random.binomial(1,inclusion_ps[j],n_cps) for j in range(p)])
    tau_positions=np.ceil(tau*(n-1)).astype(int)
    y=np.empty((p,n),dtype=int)
    for j in range(p):
        start=0
        for i in range(n_cps+1):
            end=tau_positions[i] if i<n_cps else n
            cp_included=1 if i>=n_cps else tau_indicators[j,i]
            if cp_included:
                if end>start: #else segment empty
                    y[j,start:end]=simulate_segment_data(end-start,num_categories[j],model=model)
                start=end

    if y_filename!=None:
        np.savetxt(y_filename,y,fmt='%i' if model!="normal" else '%.18e',delimiter=",")
    if x_filename!=None:
        np.savetxt(x_filename,np.arange(n)/float(n-1),delimiter=",")

def main():
    n=1000 if len(sys.argv)<2 else int(sys.argv[1])
    p=2 if len(sys.argv)<3 else int(sys.argv[2])
    k=5 if len(sys.argv)<4 else int(sys.argv[3])
    s=0 if len(sys.argv)<5 else int(sys.argv[4])
    probability_models=[MultinomialDirichlet(k=np.array([3,5],dtype=int),alpha=1),NormalNIG(p=3,alpha_beta=[.1,.1],v=1)]#PoissonGamma(p=3,alpha_beta=[1,10])]
    cps,regimes=simulate_changepoints_and_regimes(n=n,n_cps=5,inclusion_ps=np.repeat(0.3*0+1,2),seed=None,tau_filename="tau.txt")
    for i in range(len(probability_models)):
        file_ending="_"+str(i)+".txt"
        simulate_changepoint_data(n=n,y_filename="y"+file_ending,x_filename="x"+file_ending,seed=s,tau=cps,regimes=regimes,model="normal")

main()
