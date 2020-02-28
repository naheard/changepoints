#! /usr/bin/env python3
from data import Data
from multinomial_dirichlet import MultinomialDirichlet
from poisson_gamma import PoissonGamma
from changepoint_model import ChangepointModel
import numpy as np

dat=Data.from_arguments('y.txt','x.txt',dtype=int,xdtype=float)
md=MultinomialDirichlet(dat,alpha=.01)
pg=PoissonGamma(dat,alpha_beta=[.01,.01])
tau=np.loadtxt('tau.txt')
#print(md.changepoint_likelihood(tau=tau))
cpm=ChangepointModel([md])#,pg])
do_mcmc=True
if not do_mcmc:
    cpm.write_changepoints_and_regimes()
    print(cpm.calculate_posterior())
    cpm.set_changepoints(tau,[[0],[1],[2],[3],[4],[5]])
    cpm.write_changepoints_and_regimes()
    print(cpm.calculate_posterior())
    print(cpm.calculate_prior())
    cpm.set_changepoints(tau,[[0,5],[1],[2],[3],[4]])
    cpm.write_changepoints_and_regimes()
    print(cpm.calculate_posterior())
    print(cpm.calculate_prior())
    cpm.delete_changepoint(2)
    cpm.write_changepoints_and_regimes()
    print(cpm.calculate_posterior())
    cpm.add_changepoint(tau[2-1])
    cpm.write_changepoints_and_regimes()
    print(cpm.calculate_posterior())
    cpm.change_regime_of_changepoint(1,2)
    cpm.write_changepoints_and_regimes()
    print(cpm.calculate_posterior())
    print(cpm.get_effective_changepoint_locations())
    print(cpm.calculate_prior())
else:
    cpm.mcmc(10000,seed=0)
#print(cpm.find_position_in_changepoints(.3))
#print(cpm.get_lhd())
