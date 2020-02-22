#! /usr/bin/env python3
import numpy as np
from data import Data
import sys

def simulate_segment_data(n,k,alpha=1):
    theta=np.random.dirichlet(np.repeat(alpha,k))
    y=np.random.choice(int(k),int(n),p=theta)
    return(y)

def simulate_multinomial_changepoint_data(n=1000,p=2,len_tau=5,num_categories=np.array([3,5],dtype=int),inclusion_ps=np.repeat(0.3*0+1,2),seed=None,y_filename=None,x_filename=None,tau_filename=None):
    if seed is not None:
        np.random.seed(seed)
    tau=np.sort(np.random.uniform(size=len_tau))
    tau_indicators=np.array([np.random.binomial(1,inclusion_ps[j],len_tau) for j in range(p)])
    tau_positions=np.ceil(tau*(n-1)).astype(int)
    y=np.empty((p,n))
    for j in range(p):
        start=0
        for i in range(len_tau+1):
            end=tau_positions[i] if i<len_tau else n
            cp_included=1 if i>=len_tau else tau_indicators[j,i]
            if cp_included:
                if end>start: #else segment empty
                    y[j,start:end]=simulate_segment_data(end-start,num_categories[j])
                start=end

    if y_filename!=None:
        np.savetxt(y_filename,y,fmt='%i',delimiter=",")
    if x_filename!=None:
        np.savetxt(x_filename,np.arange(n)/float(n-1),delimiter=",")
    if tau_filename!=None:
        np.savetxt(tau_filename,tau,delimiter=",",fmt='%1.3f',newline=" ")

def main():
    n=1000 if len(sys.argv)<2 else int(sys.argv[1])
    p=2 if len(sys.argv)<3 else int(sys.argv[2])
    k=5 if len(sys.argv)<4 else int(sys.argv[3])
    s=0 if len(sys.argv)<5 else int(sys.argv[4])
    simulate_multinomial_changepoint_data(n=n,y_filename="y.txt",x_filename="x.txt",tau_filename="tau.txt",seed=s)

main()
