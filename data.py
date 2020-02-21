import numpy as np
#from bisect import bisect

class Data(object):
    #y is a (pxn) matrix of data 'observations'
    #x is an n-vector of time points (or some other univariate covariate)
    def __init__(self, y=[], x=None, transpose=False):
        self.y=np.atleast_2d(y)
        if transpose:
            self.y=self.y.transpose()
        self.x=None if x is None else np.array(x)
        self.p, self.n=self.y.shape
        self.Sy = self.Sy2 = self.Sx = self.categories = self.num_categories = self.cum_counts = None

    @classmethod
    def from_arguments(cls, yfile, xfile=None, dtype=float, xdtype=float, transpose=False,delim=","):
        y=np.loadtxt(yfile,dtype=dtype,delimiter=delim)
        x=None if xfile is None else np.loadtxt(xfile,dtype=xdtype,delimiter=delim)
        return(cls(y,x,transpose))

    def find_position(self,t):
        position=int(ceil(t)) if self.x is None else np.searchsorted(self.x,t)
        return(position)

    def calculate_y_cumulative_sum(self):
        self.Sy=np.cumsum(self.y,axis=1)

    def calculate_y_cumulative_sum_sq(self):
        self.Sy2=np.cumsum(self.y*self.y,axis=1)

    def calculate_x_cumulative_sum(self):
        self.Sx=np.cumsum(self.x)

    @staticmethod
    def get_sum_between(mx,start,end):
        return(mx[:,end-1]-(mx[:,start-1] if start>0 else 0))

    def get_ysum_between(self,start,end):
        #return sum of y[start:end]
        if self.Sy is not None:
            return(self.get_sum_between(self.Sy,start,end))
        else:
            return(np.sum(self.y[:,start:end],axis=1))

    def calculate_unique_categories(self):
        self.categories=[{} for j in range(self.p)]
        self.num_categories=np.empty(self.p,dtype=int)
        for j in range(self.p):
            self.categories[j]={key: v for key,v in enumerate(np.unique(self.y[j,:]))}
            self.num_categories[j]=len(self.categories[j])

    def calculate_y_cumulative_counts(self,num_categories=None):
        if num_categories is None:
            self.calculate_unique_categories()
            num_categories=self.num_categories

        self.cum_counts = np.array([np.zeros((num_categories[j],self.n),dtype=int) for j in range(self.p)])
        for j in range(self.p):
            for i in range(self.n):
                if i>0:
                    self.cum_counts[j][:,i]=self.cum_counts[j][:,i-1]
                cat_ji=self.categories[j][self.y[j,i]] if self.categories is not None else self.y[j,i]
                self.cum_counts[j][cat_ji,i]+=1

    def get_y_cumulative_counts(self,dim=0,start=0,end=None):
        start_cts=np.zeros(self.num_categories[dim]) if start==0 else self.cum_counts[dim][:,(start-1)]
        end_cts=self.cum_counts[dim][:,self.n-1] if (end is None or end>=self.n) else self.cum_counts[dim][:,end]
        return(end_cts-start_cts)

    def get_x_max(self):
        return(self.n-1 if self.x is None else max(self.x))
