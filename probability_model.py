import numpy as np
from abc import ABC, abstractmethod
from data import Data

class ProbabilityModel(ABC):

    def __init__(self,data,parameters=None):
        self.data=data
        self.parameters=parameters

    def likelihood(self,start_end=[(0,None)]):
        lhd=0
        for j in range(self.data.p):
            lhd+=self.likelihood_j(j,start_end)

        return(lhd)

    def changepoint_likelihood(self,start=0,end=None,tau=np.array([],dtype=float)):
        lhd=0
        start=0
        len_tau=len(tau)
        for i in range(len_tau+1):
            end=self.data.find_position(tau[i]) if i<len_tau else self.data.n
            lhd+=self.likelihood([(start,end)])
            start=end

        return(lhd)

    @abstractmethod
    def likelihood_j(self,j=0,start_end=[(0,None)]):
        pass #A likelihood function for the jth row of data.y, taking several slices starting and ending at the positions of start_end

    def find_data_position(self,t):
        if self.data is None:
            position=0
        else:
            position=self.data.find_position(t)
        return(position)
