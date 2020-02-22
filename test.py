#! /usr/bin/env python3
from data import Data
from multinomial_dirichlet import MultinomialDirichlet
from changepoint_model import ChangepointModel
import numpy as np

dat=Data.from_arguments('y.txt','x.txt',int,float)
md=MultinomialDirichlet(dat,alpha=.01)
tau=np.loadtxt('tau.txt')
#print(md.changepoint_likelihood(tau=tau))
cpm=ChangepointModel([md,md])
print(cpm.likelihood())
cpm.set_changepoints(tau,[[0],[1],[2],[3],[4],[5]])
print(cpm.likelihood())
cpm.set_changepoints(tau,[[0,5],[1],[2],[3],[4]])
print(cpm.likelihood())
#print(cpm.find_position_in_changepoints(.3))
#print(cpm.get_lhd())
