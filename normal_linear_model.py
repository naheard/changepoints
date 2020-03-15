from probability_model import ProbabilityModel
from data import Data
import numpy as np
from scipy.special import gammaln

LOG_PI=np.log(np.pi)

class NormalLinearModel(ProbabilityModel):
    def __init__(self,data=None,alpha_beta=(.1,.1),v=1,p=1):
        ProbabilityModel.__init__(self,data,alpha_beta) #y_i~N(mu,sigma**2)
        self.a0,self.b0=alpha_beta #sigma**(-2)~Gamma(a0,b0)
        self.v1=self.v2=v #a multiplier for the standard deviation of mu, mu~N(0,(sigma*v)**2)
        self.v_inv1,self.v_inv2=1.0/self.v1,1.0/self.v2
        self.log_det_V=np.log(self.v1*self.v2)
        self.p=p if self.data is None else self.data.p
        self.density_constant=-.5*self.log_det_V+self.a0*np.log(self.b0)-gammaln(self.a0)
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
        mu=np.random.normal(scale=sigma*self.v,size=self.p)
        return(mu,sigma)

    def simulate_data(self,n,mu=None,sigma=None):
        if mu is None or sigma is None:
            mu,sigma=self.sample_parameter()
        return([np.random.normal(mu[i],sigma[i],size=n) for i in range(self.p)])

    def log_density(self,y,y2,x,x2,xy,n=1):
        #V_n^{-1}=XtX+V^{-1}=[[n+self.v_inv1,x][x,x2+self.v_inv2]]
        #V_n=[[x2+self.v_inv2,-x][-x,n+self.v_inv1]]/det_V_n_inv
        #XtY=[x,xy]
        det_V_n_inv=(n+self.v_inv1)*(x2+self.v_inv2)-x**2
        log_det_V_n=-np.log(det_V_n_inv)
        m_n=np.array([(x2+self.v_inv2)*y-x*xy,-x*y+(n+self.v_inv1)*xy])/det_V_n_inv #Posterior mean of mu, V_n * XtY
        a_n=self.a0+n
        b_n=self.b0+.5*(y2-(x*m_n[0]+xy*m_n[1])) #Posterior gamma shape parameter of sigma
        lhd=self.density_constant+.5*log_det_V_n-a_n*np.log(b_n)+gammaln(a_n)
        return(lhd)

####
def get_linear_model_lhd(y,ydot,yty,a,b,lam,yn1=None):
    global m_n
#    if len(y)==1: #or np.isscalar(y)
#        k=1
    k=len(y)
    lam1,lam2=.0001*lam,lam
    Dty=np.array([ydot,np.dot(y,range(k))])
    V_n=np.array([[k+lam1,k*(k-1)/2],[k*(k-1)/2,k*(k-1)*(2*k-1)/6+lam2]])#np.dot(D.T,D)+lam*np.eye(D.shape[1])
    det_V_n=(k+lam1)*(k*(k-1)*(2*k-1)/6+lam2)-(k*(k-1))**2/4.0
    ldet_V_n=np.log(det_V_n)# np.linalg.slogdet(V_n)[1]
    V_inv_n=np.array([[k*(k-1)*(2*k-1)/6+lam2,-k*(k-1)/2],[-k*(k-1)/2,k+lam1]])/det_V_n #np.linalg.inv(V_n)
    m_n=np.dot(V_inv_n,Dty.T)
    a_n=a+k
    b_n=b+.5*(yty-np.dot(m_n.T,Dty.T))
    llam=.5*(np.log(lam1*lam2))
    if yn1 is None:
        a0lb0=a*np.log(b)
        lgama0=gammaln(a)
#        lhd=a0lb0-a_n*np.log(b_n)+llam-.5*ldet_V_n+lgama0-gammaln(a+k)
        lhd=a0lb0-a_n*np.log(b_n)+llam-.5*ldet_V_n-lgama0+gammaln(a+k)
    else:
        V_n1=np.array([[k+1+lam1,k*(k+1)/2],[k*(k+1)/2,k*(k+1)*(2*k+1)/6+lam2]]) #V_n+np.array([[1,k],[k,k**2]])
        det_V_n1=(k+1+lam1)*(k*(k+1)*(2*k+1)/6+lam2)-(k*(k+1))**2/4.0
        ldet_V_n1=np.log(det_V_n1) #np.linalg.slogdet(V_n1)[1]
        V_inv_n1=np.array([[k*(k+1)*(2*k+1)/6+lam2,-k*(k+1)/2],[-k*(k+1)/2,k+1+lam1]])/det_V_n1 #np.linalg.inv(V_n1)
        m_n1=np.dot(V_inv_n1,(Dty+[yn1,yn1*k]).T)
        b_n1=b+.5*(yty+yn1**2-np.dot(m_n1.T,(Dty+[yn1,yn1*k]).T))
        lhd=-np.log(a+k)+a_n*np.log(b_n)+.5*ldet_V_n  -(a_n+1)*np.log(b_n1)-.5*ldet_V_n1#+llam
    return lhd
