## data.py

import numpy as np
from collections import defaultdict,Counter
from scipy.special import gammaln
#from bisect import bisect

class Data(object):
    #y is a (pxn) matrix of data 'observations'
    #x is an n-vector of time points (or some other univariate covariate)
    def __init__(self, y=[], x=None, cts=None, transpose=False):
        self.Sy = self.Sy2 = self.Sx = self.Sx2= self.Sxy = self.Slyf = self.categories = self.num_categories = self.cum_counts = self.Kx = None
        if y is not None:
            self.y=np.atleast_2d(y)
            self.p, self.n=self.y.shape
        else:
            self.y=np.atleast_2d(cts)
            self.cum_counts=np.array([np.cumsum(self.y,axis=1)])
            self.num_categories=[self.cum_counts.shape[1]]
            self.p=1
            self.n=self.y.shape[1]
        if transpose:
            self.y=self.y.transpose()
            self.p, self.n=self.n, self.p
        self.x=None if x is None else np.array(x)

    @classmethod
    def from_arguments(cls, yfile=None, xfile=None, xyfile=None, dtype=float, xdtype=float, transpose=False,delim=",",y_textfile=False):
        y=cts=None
        if xyfile is not None: #a single file containing both xs and ys
            xy = np.loadtxt(xyfile,delimiter=delim)
            if not transpose: # first row are xs, subsequent rows are ys
                x = np.array(xy[0,:],dtype=xdtype)
                y = np.array(xy[1:,:],dtype=dtype)
            else: # first column are xs, subsequent columns are ys
                x = np.array(xy.T[0,:],dtype=xdtype)
                y = np.array(xy.T[1:,:].T,dtype=dtype)
            return(cls(y,x,cts,transpose))
        if not y_textfile:
            y=np.loadtxt(yfile,dtype=dtype,delimiter=delim)
        else:
            cts=word_counts_from_file(yfile)[0]#,delim=delim)
        x=None if xfile is None else np.loadtxt(xfile,dtype=xdtype,delimiter=delim)
        return(cls(y,x,cts,transpose))

    def find_position(self,t):
        if t==-float("inf"):
            return(0)
        elif t==float("inf"):
            return(n)
        position=int(np.ceil(t)) if self.x is None else np.searchsorted(self.x,t)
        return(position)

    def calculate_y_cumulative_sum(self):
        self.Sy=np.cumsum(self.y,axis=1)

    def calculate_y_cumulative_sum_sq(self):
        self.Sy2=np.cumsum(self.y*self.y,axis=1)

    def calculate_y_cumulative_sum_log_factorials(self):
        self.Slyf=np.atleast_2d([np.cumsum([gammaln(yi+1) for yi in y]) for y in self.y])

    def calculate_x_cumulative_sum(self):
        if self.x is not None:
            self.Sx=np.cumsum(self.x)
        else:
            self.Sx=np.array([i*(i+1)/2 for i in range(self.n)])

    def calculate_x_cumulative_sum_sq(self):
        if self.x is not None:
            self.Sx2=np.cumsum(self.x*self.x)
        else:
            self.Sx=np.array([i*(i+1)*(2*i+1) for i in range(self.n)])

    def calculate_xy_cumulative_sum(self):
        if self.x is not None:
            self.Sxy=np.cumsum(np.multiply(self.x,self.y),axis=1)
        else:
            self.Sxy=np.cumsum(np.multiply(range(self.n),self.y),axis=1)

    @staticmethod
    def get_mx_diff_between(mx,start,end,j=None):
        end_index=mx.shape[1] if end is None else end
        if end_index ==0:
            return 0
        if j is None:
            return(mx[:,end_index-1]-(mx[:,start-1] if start>0 else 0))
        else:
            return(mx[j,end_index-1]-(mx[j,start-1] if start>0 else 0))

    @staticmethod
    def get_diff_between(ar,start,end):
        end_index=len(ar) if end is None else end
        if end_index ==0:
            return 0
        return(ar[end_index-1]-(ar[start-1] if start>0 else 0))

    def get_y_sum_between(self,start,end,j=None):
        #return sum of y[start:end]
        if self.Sy is not None:
            return(self.get_mx_diff_between(self.Sy,start,end,j))
        else:
            if j is None:
                return(np.sum(self.y[:,start:end],axis=1))
            else:
                return(np.sum(self.y[j,start:end]))

    def get_y_max_between(self,start,end,j=None):
        #return maximum of y[start:end]
        if j is None:
            return(np.max(self.y[:,start:end],axis=1))
        else:
            return(np.max(self.y[j,start:end]))

    def get_y_sum_squares_between(self,start,end,j=None):
        #return sum of squares of y[start:end]
        if self.Sy2 is not None:
            return(self.get_mx_diff_between(self.Sy2,start,end,j))
        else:
            if j is None:
                return(np.sum(self.y[:,start:end]**2,axis=1))
            else:
                return(np.sum(self.y[j,start:end]**2))

    def get_y_factorial_prod_between(self,start,end,j=None):
        #return product of factorial terms of y[start:end]
        if self.Slyf is not None:
            return(self.get_mx_diff_between(self.Slyf,start,end,j))
        else:
            if j is None:
                return([np.sum([gammaln(yi+1) for yi in y]) for y in self.y[:,start:end]**2])
            else:
                return(np.sum([gammaln(yi+1) for yi in self.y[j,start:end]**2]))

    def get_x_sum_between(self,start,end):
        #return sum of x[start:end]
        if self.Sx is not None:
            return(self.get_diff_between(self.Sx,start,end))
        else:
            if self.x is None:
                if end is None:
                    end=self.n
                return(end-start)
            else:
                return(np.sum(self.x[start:end]))

    def get_x_sum_squares_between(self,start,end):
        #return sum of squares of x[start:end]
        if self.Sx2 is not None:
            return(self.get_diff_between(self.Sx2,start,end))
        else:
            if self.x is None:
                if end is None:
                    end=self.n
                if start<=1:
                    return(end*(end-1)*(2*end-1)/6.0)
                return((end*(end-1)*(2*end-1)-start*(start-1)*(2*start-1))/6.0)
            else:
                return(np.sum(self.x[start:end]**2))

    def get_xy_sum_between(self,start,end,j=None):
        #return sum of x[start:end]*y[start:end]
        if self.Sxy is not None:
            return(self.get_mx_diff_between(self.Sxy,start,end,j))
        else:
            if self.x is None:
                if j is None:
                    return(np.sum(np.multiply(range(start,end),self.y[:,start:end])))
                else:
                    return(np.sum(np.multiply(range(start,end),self.y[j,start:end])))
            else:
                if j is None:
                    return(np.sum(np.multiply(self.x[start:end],self.y[:,start:end])))
                else:
                    return(np.sum(np.multiply(self.x[start:end],self.y[j,start:end])))

    def get_combined_indices(self,start_end=[(0,None)]):
        return(np.concatenate([np.arange(start,end if end is not None else self.n) for start,end in start_end]))

    def get_combined_ys(self,dim=0,start_end=[(0,None)]):
        return(np.concatenate([self.y[dim,start:end] for start,end in start_end],axis=0))

    def get_combined_xs(self,start_end=[(0,None)]):
        if self.x is None:
            return(self.get_combined_indices(start_end))
        return(np.concatenate([self.x[start:end] for start,end in start_end]))

    def get_combined_y_sums(self,dim=0,start_end=[(0,None)]):
        return(np.sum([self.get_y_sum_between(start,end,dim) for start,end in start_end],axis=0))

    def get_combined_y_maxs(self,dim=0,start_end=[(0,None)]):
        return(np.max([self.get_y_max_between(start,end,dim) for start,end in start_end],axis=0))

    def get_combined_y_sum_squares(self,dim=0,start_end=[(0,None)]):
        return(np.sum([self.get_y_sum_squares_between(start,end,dim) for start,end in start_end],axis=0))

    def get_combined_y_factorials(self,dim=0,start_end=[(0,None)]):
        return(np.prod([self.get_y_factorial_prod_between(start,end,dim) for start,end in start_end],axis=0))

    def get_combined_x_sums(self,start_end=[(0,None)]):
        return(np.sum([self.get_x_sum_between(start,end) for start,end in start_end],axis=0))

    def get_combined_x_sum_squares(self,start_end=[(0,None)]):
        return(np.sum([self.get_x_sum_squares_between(start,end) for start,end in start_end],axis=0))

    def get_combined_xy_sums(self,dim=0,start_end=[(0,None)]):
        return(np.sum([self.get_xy_sum_between(start,end,dim) for start,end in start_end],axis=0))

    def calculate_unique_categories(self):
        self.categories=[{} for j in range(self.p)]
        self.num_categories=np.empty(self.p,dtype=int)
        for j in range(self.p):
            self.categories[j]={key: v for key,v in enumerate(np.unique(self.y[j,:]))}
            self.num_categories[j]=len(self.categories[j])

    def calculate_y_cumulative_counts(self,num_categories=None):
        if self.cum_counts is not None:
            return()
        if num_categories is None:
            self.calculate_unique_categories()
            num_categories=self.num_categories
        else:
            self.num_categories=num_categories

        self.cum_counts = np.array([np.zeros((num_categories[j],self.n),dtype=int) for j in range(self.p)], dtype=object)
        for j in range(self.p):
            for i in range(self.n):
                if i>0:
                    self.cum_counts[j][:,i]=self.cum_counts[j][:,i-1]
                cat_ji=self.categories[j][self.y[j,i]] if self.categories is not None else self.y[j,i]
                self.cum_counts[j][cat_ji,i]+=1

    def get_y_cumulative_counts(self,dim=0,start=0,end=None):
        start_cts=np.zeros(self.num_categories[dim],dtype=int) if start==0 else self.cum_counts[dim][:,(start-1)]
        end_cts=self.cum_counts[dim][:,self.n-1] if (end is None or end>=self.n) else self.cum_counts[dim][:,end-1]
        return(end_cts-start_cts)

    def get_combined_y_cumulative_counts(self,dim=0,start_end=[(0,None)]):
        return(np.sum([self.get_y_cumulative_counts(dim,start,end) for start,end in start_end],axis=0))

    def get_x_max(self):
        return(self.n-1 if self.x is None else max(self.x))

    def create_kernel_covariance_matrix(self,kernel=None,par=None):
        if kernel is None:
            kernel=gaussian_kernel
        self.Kx=np.ones([self.n,self.n])
        for i in range(self.n):
            for j in range(i):
                kij=kernel(i,j) if self.x is None else kernel(self.x[i],self.x[j])
                self.Kx[i,j]=self.Kx[j,i]=kernel(self.x[i],self.x[j],par=par)

    def get_combined_kernel_covariance_matrix(self,start_end):
        indices=self.get_combined_indices(start_end)
        n_indices=len(indices)
        K=np.ones([n_indices,n_indices])
        for i_ in range(n_indices):
            i=indices[i_]
            for j_ in range(i_):
                j=indices[j_]
                K[i_,j_]=K[j_,i_]=self.Kx[i,j]

        return(K)

    def plot(self,cps, regimes,inclusion_vectors,means=None,plot_dim=0,log_x=False,log_y=False):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4*self.p))
        for index in range(self.p):
            plt.subplot(self.p,1,index+1)
            plt.autoscale(enable=True, axis='x', tight=True)
            r=0
            for i in range(len(cps)):
                new_r=regimes[i] if regimes is not None else i+1
                if (inclusion_vectors[index,new_r] if inclusion_vectors is not None else True) and new_r!=r:
                    plt.axvline(x=cps[i],linewidth=.8,color='red')
                    r=new_r
                else:
                    plt.axvline(x=cps[i],linewidth=.8,color='pink')
            if means is not None:
                xs = np.insert(np.insert(cps,len(cps),np.max(self.x)),0,np.min(self.x))
                for i in range(len(means)):
                    plt.plot((xs[i],xs[i+1]),(means[i],means[i]),c='#8ace00')
        plt.scatter(self.x,self.y[plot_dim,:],marker="x",s=1.5)
        if log_x:
            plt.xscale('log')
        if log_y:
            plt.yscale('log')
        plt.show()

def gaussian_kernel(x1,x2,par=None):
    l=1 if par is None else par
    return(np.exp(-.5*((x1-x2)/float(l))**2))

def word_counts_from_file(file,delim=" "):
    import string
    word_counts=defaultdict(Counter)
    total_word_counts=Counter()
    with open(file,"r") as f:
        line_count=0
        for line in f:
            word_counts[line_count]=Counter(line.strip().translate(str.maketrans('', '', string.punctuation)).lower().split(delim))
            total_word_counts+=word_counts[line_count]
            line_count+=1

    n=line_count
    word_map={}
    ctr=0
    V=len(total_word_counts)
    for w,c in total_word_counts.most_common():
        word_map[w]=ctr
        ctr+=1

    y_counts=np.zeros((V,n),dtype=int)
    for line_count in range(n):
        for w in word_counts[line_count]:
            y_counts[word_map[w],line_count]=word_counts[line_count][w]

    return(y_counts,word_map)
