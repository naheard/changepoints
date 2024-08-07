## probability_model.py

import numpy as np
from abc import ABC, abstractmethod
from data import Data

class ProbabilityModel(ABC):

    def __init__(self,data=None,parameters=None):
        self.initialise_data(data)
        self.parameters=parameters

    def initialise_data(self,data=None):
        self.data=data
        self.data_type=float if self.data is None else self.data.y.dtype
        self.p=1 if self.data is None else self.data.p

    def add_data(self,data):
        self.data=data
        self.calculate_data_summaries()

    def calculate_data_summaries(self):
        None

    def likelihood(self,start_end=[(0,None)],y=None):
        if len(start_end)==0:
            return(0)
        lhd=0
        if self.data is not None:
            for j in range(self.data.p):
                lhd+=self.likelihood_component(j,start_end)
        else:
            lhd=self.likelihood_component(0,y=y)
        return(lhd)

    def mean(self,start_end=[(0,None)],y=None):
        means = []
        if self.data is not None:
            means+=[self.component_mean(j,start_end) for j in range(self.data.p)]
        else:
            means+=[self.component_mean(0,y=y)]
        return(means)

    def get_dimension(self):
        return(self.p)

    @abstractmethod
    def likelihood_component(self,j=0,start_end=[(0,None)],y=None):
        pass #A likelihood function for the jth row of data.y, taking several slices starting and ending at the positions of start_end

    @abstractmethod
    def sample_parameter(self):
        pass

    @abstractmethod
    def simulate_data(self,n,thetas=None,x=None):
        pass

    def find_data_position(self,t):
        if self.data is None:
            position=0
        else:
            position=self.data.find_position(t)
        return(position)
