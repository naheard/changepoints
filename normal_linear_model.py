from probability_model import ProbabilityModel
from data import Data
import numpy as np
from scipy.special import gammaln

LOG_PI=np.log(np.pi)

class NormalLinearModel(ProbabilityModel):
    def __init__(self,data=None,alpha_beta=(.1,.1),v=1,p=1):
        ProbabilityModel.__init__(self,data,alpha_beta) #y_i~N(mu,sigma**2)
        self.a0,self.b0=alpha_beta #sigma**(-2)~Gamma(a0,b0)
        self.v=v #a multiplier for the standard deviation of mu, mu~N(0,(sigma*v)**2)
        self.p=p if self.data is None else self.data.p
        self.density_constant=-np.log(self.v)+self.a0*np.log(self.b0)-gammaln(self.a0)
        if self.data is not None:
            self.data.calculate_y_cumulative_sum()
            self.data.calculate_y_cumulative_sum_sq()
            self.data.calculate_xy_cumulative_sum()

    def likelihood_component(self,j=0,start_end=[(0,None)],y=None,x=None):
        lhd=self.density_constant
        sy=sum(y) if y is not None else self.data.get_combined_y_sums(j,start_end)
        sy2=sum(y*y) if y is not None else self.data.get_combined_y_sum_squares(j,start_end)
        sxy=sum(x*y) if y is not None else self.data.get_combined_xy_sums(j,start_end)
        n=1 if y is not None else sum([e-s for s,e in start_end])
        return(self.log_density(sy,sy2,sxy,n))

    def sample_parameter(self):
        sigma=1.0/np.sqrt(np.random.gamma(self.a0,1.0/self.b0,size=self.p))
        mu=np.random.normal(scale=sigma*self.v,size=self.p)
        return(mu,sigma)

    def simulate_data(self,n,mu=None,sigma=None):
        if mu is None or sigma is None:
            mu,sigma=self.sample_parameter()
        return([np.random.normal(mu[i],sigma[i],size=n) for i in range(self.p)])

    def log_density(self,y,y2,xy,n=1):
        ld=self.density_constant-.5*n*LOG_PI-.5*np.log(1.0/self.v+n)+gammaln(self.a0+.5*n)-(self.a0+.5*n)*np.log(self.b0+0.5*(y2-y*y*(self.v/(n*self.v+1.0))))
        return(ld)

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
