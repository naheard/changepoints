#! /usr/bin/env python3
from data import Data
from multinomial_dirichlet import MultinomialDirichlet
from poisson_gamma import PoissonGamma
from normal_normal_inverse_gamma import NormalNIG
from normal_linear_model import NormalLinearModel
from uniform_pareto import UniformPareto
from changepoint_model import ChangepointModel
import numpy as np

dat=Data.from_arguments('y_0.txt','x_0.txt',dtype=int,xdtype=float)
dat_txt=Data.from_arguments('text.txt',y_textfile=True)
md=MultinomialDirichlet(dat_txt,alpha=.01)
pg=PoissonGamma(dat,alpha_beta=[.01,.01])
nig=NormalNIG(dat,alpha_beta=[.01,.01],v=1)
nlm=NormalLinearModel(dat,alpha_beta=[1,10.0],v=1)
up=UniformPareto(dat,alpha_beta=[.01,4])
tau=np.loadtxt('tau.txt')
#print(md.changepoint_likelihood(tau=tau))
cpm=ChangepointModel([md],infer_regimes=True,disallow_successive_regimes=True,spike_regimes=not False)#,pg])
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
#    cpm.delete_changepoint(2)
#    cpm.write_changepoints_and_regimes()
#    print(cpm.calculate_posterior())
#    cpm.add_changepoint(tau[2-1])
#    cpm.write_changepoints_and_regimes()
    print(cpm.calculate_posterior())
    cpm.set_changepoints(tau,[[0],[1,4],[2],[3],[5]])
    cpm.write_changepoints_and_regimes()
    print(cpm.calculate_posterior())
    cpm.change_regime_of_changepoint(1,5)
    cpm.write_changepoints_and_regimes()
    print(cpm.calculate_posterior())
    print(cpm.get_effective_changepoint_locations())
    print(cpm.calculate_prior())
    np.random.seed(1000)
    for _ in range(4):
        cpm.mcmc_refresh()
        cpm.propose_change_regime_of_changepoint()
        cpm.write_changepoints_and_regimes()
        print(cpm.posterior)
        print(cpm.calculate_posterior())
else:
    cpm.mcmc(10000,seed=10,hill_climbing=True)
    print(cpm.get_effective_changepoint_locations())

#print(cpm.find_position_in_changepoints(.3))
#print(cpm.get_lhd())
