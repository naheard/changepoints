from probability_model import ProbabilityModel
from data import Data
import numpy as np
from scipy.special import gammaln

LOG_PI=np.log(np.pi)

class NormalLinearModel(ProbabilityModel):
    def __init__(self,data=None,alpha_beta=(1,0.01),v=.1,p=1):
        ProbabilityModel.__init__(self,data,alpha_beta) #y_i~N(mu,sigma**2)
        self.a0,self.b0=alpha_beta #sigma**(-2)~Gamma(a0,b0)
        self.v1=self.v2=v #a multiplier for the standard deviation of mu, mu~N(0,(sigma*v)**2)
        self.v_inv1,self.v_inv2=1.0/self.v1,1.0/self.v2
        self.log_det_V=np.log(self.v1*self.v2)
        self.p=p if self.data is None else self.data.p
        self.density_constant=-.5*self.log_det_V+self.a0*np.log(self.b0)-gammaln(self.a0)
        self.calculate_data_summaries()

    def calculate_data_summaries(self):
        if self.data is not None:
            self.data.calculate_y_cumulative_sum()
            self.data.calculate_y_cumulative_sum_sq()
            self.data.calculate_x_cumulative_sum()
            self.data.calculate_x_cumulative_sum_sq()
            self.data.calculate_xy_cumulative_sum()

    def likelihood_component(self,j=0,start_end=[(0,None)],y=None,x=None):
        lhd=self.density_constant
        sy=sum(y) if y is not None else self.data.get_combined_y_sums(j,start_end)
        sy2=sum(y*y) if y is not None else self.data.get_combined_y_sum_squares(j,start_end)
        sx=sum(x) if x is not None else self.data.get_combined_x_sums(start_end)
        sx2=sum(x*x) if x is not None else self.data.get_combined_x_sum_squares(start_end)
        sxy=sum(x*y) if y is not None else self.data.get_combined_xy_sums(j,start_end)
        n=len(y) if y is not None else sum([(e if e is not None else self.data.n)-s for s,e in start_end])
        return(self.log_density(sy,sy2,sx,sx2,sxy,n))

    def sample_parameter(self):
        sigma=1.0/np.sqrt(np.random.gamma(self.a0,1.0/self.b0,size=self.p))
        mu=np.multiply(np.atleast_2d(sigma).transpose(),np.random.multivariate_normal(mean=np.zeros(2),cov=np.diag([self.v1,self.v2]),size=self.p))
        return(mu,sigma)

    def simulate_data(self,n,thetas=None,x=None):
        if thetas is None:
            mu,sigma=self.sample_parameter()
        else:
            mu,sigma=thetas[0],thetas[1]
        if x is None:
            x=np.array(range(n))
        mean=[mu[i][0]+mu[i][1]*x for i in range(self.p)]
        return([mean[i]+np.random.normal(0,sigma[i],size=n) for i in range(self.p)])

    def log_density(self,y,y2,x,x2,xy,n=1):
        #V_n_inv=XtX+V^{-1}=[[n+self.v_inv1,x][x,x2+self.v_inv2]]
        #V_n=[[x2+self.v_inv2,-x][-x,n+self.v_inv1]]/det_V_n_inv
        #XtY=[x,xy]
        det_V_n_inv=(n+self.v_inv1)*(x2+self.v_inv2)-x**2
        log_det_V_n=-np.log(det_V_n_inv)
        m_n=np.array([(x2+self.v_inv2)*y-x*xy,-x*y+(n+self.v_inv1)*xy])/det_V_n_inv #Posterior mean of mu, V_n * XtY
        m_nT__V_n_inv__m_n=m_n[0]*(m_n[0]*(n+self.v_inv1)+m_n[1]*x) + m_n[1]*(m_n[0]*x+m_n[1]*(x2+self.v_inv2))
        a_n=self.a0+n #Posterior gamma shape parameter of sigma
        b_n=self.b0+.5*(y2-m_nT__V_n_inv__m_n) #Posterior gamma scale parameter of sigma
        lhd=self.density_constant+.5*log_det_V_n-a_n*np.log(b_n)+gammaln(a_n)
        return(lhd)
