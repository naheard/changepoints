#! /usr/bin/env python3
from data import Data
from multinomial_dirichlet import MultinomialDirichlet
from changepoint_model import ChangepointModel

dat=Data.from_arguments('y.txt','x.txt',int,float)
md=MultinomialDirichlet(dat,alpha=.01)
tau=[0.004718856190972565,0.27836938509379616,0.4245175907491331,0.5434049417909654,0.8447761323199037]
print(md.changepoint_likelihood(tau=tau))
cpm=ChangepointModel([md,md])
print(cpm.likelihood())
cpm.set_changepoints(tau)
print(cpm.likelihood())
print(cpm.find_position_in_changepoints(.3))
print(cpm.get_lhd())
