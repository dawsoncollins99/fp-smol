import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

# units: um, s, cm^-3
T, p = 288.0, 1000e2
def esat_Pa(T):
    return 100.*10.0**(23.832241-5.02808*np.log10(T)-1.3816e-7*10.0**(11.344-0.0303998*T)
                       +8.1328e-3*10.0**(3.49149-1302.8844/T)-2949.076/T)
def growth_coefficient(T,p):
    T0,Lv,Rv,rhow=273.15,2.5e6,461.5,1e3
    K=7.7e-5*(T-T0)+0.02399
    Dv=(1.57e-7*(T-T0)+2.211e-5)*1e5/p
    return 1e12/(rhow*((Lv/(Rv*T))*Lv/(K*T)+Rv*T/(esat_Pa(T)*Dv)))   # xi in um^2/s
xi=growth_coefficient(T,p); a=2*xi
sigma_w,M_w,R,rho_w=0.075,18e-3,8.314,1000.
rho_s,M_s,nu_s=2160.,58.5e-3,2.0
r_dry=0.065
m_s=(4/3)*np.pi*(r_dry*1e-6)**3*rho_s
A_k=2*sigma_w*M_w/(rho_w*R*T)*1e6
B_k=3*nu_s*m_s*M_w/(4*np.pi*rho_w*M_s)*1e18
rho_air=p/(287*T)
C_stk=2*(rho_w-rho_air)*9.81/(9*1.8e-5)     # 1/(m s)
h=1.0
k_settle=C_stk*1e-12/h                       # 1/(um^2 s)
s=0.0009
J_inj=100.0                                  # cm^-3 s^-1

def bern(z):
    z=np.asarray(z,float); out=np.empty_like(z)
    sm=np.abs(z)<0.1; zs=z[sm]
    out[sm]=1-zs/2+zs**2/12-zs**4/720+zs**6/30240
    pos=(~sm)&(z>0); out[pos]=z[pos]*np.exp(-z[pos])/(-np.expm1(-z[pos]))
    neg=(~sm)&(z<0); out[neg]=z[neg]/np.expm1(z[neg])
    return out
def bern_prime(z):
    z=np.asarray(z,float); out=np.empty_like(z)
    sm=np.abs(z)<0.1; zs=z[sm]
    out[sm]=-0.5+zs/6-zs**3/180+zs**5/5040
    pos=(~sm)&(z>0); em=np.exp(-z[pos]); d=-np.expm1(-z[pos]); out[pos]=em/d-z[pos]*em/d**2
    neg=(~sm)&(z<0); E=np.expm1(z[neg]); out[neg]=1/E-z[neg]*(E+1)/E**2
    return out

class Model:
    def __init__(self,n_cells=8000,r_max=30.0):
        self.n=n_cells
        self.xf=np.linspace((r_dry/2)**2,r_max**2,n_cells+1)
        self.x=0.5*(self.xf[:-1]+self.xf[1:]); self.dx=self.xf[1]-self.xf[0]
        r=np.sqrt(self.x)
        drift=a*(s-(A_k/r-B_k/r**3))
        self.drift_face=0.5*(drift[:-1]+drift[1:])
        self.lam=k_settle*self.x
        self.b=np.zeros(n_cells); self.b[0]=J_inj
    def _assemble(self,alpha,beta,diag_extra):
        main=diag_extra.copy(); main[:-1]+=alpha; main[1:]-=beta
        return diags([-alpha,main,beta],[-1,0,1],format='csc')
    def A(self,D):
        Pe=self.drift_face*self.dx/D
        return self._assemble((D/self.dx)*bern(-Pe),-(D/self.dx)*bern(Pe),self.lam*self.dx)
    def dA_dD(self,D):
        Pe=self.drift_face*self.dx/D
        return self._assemble((bern(-Pe)+Pe*bern_prime(-Pe))/self.dx,-(bern(Pe)-Pe*bern_prime(Pe))/self.dx,np.zeros(self.n))
    def solve(self,D):
        lu=splu(self.A(D)); return lu.solve(self.b),lu

def loss_and_grad(model,ell,N_obs):
    D=np.exp(ell)
    N,lu=model.solve(D)
    scale=np.sum(N_obs**2)
    res=N-N_obs
    J=0.5*np.sum(res**2)/scale
    dJdN=res/scale
    psi=lu.solve(-dJdN,trans='T')
    dJdD=psi@(model.dA_dD(D)@N)
    return J,D*dJdD

def gradient_descent(f,ell0,step0=1.0,max_move=1.0,tol=1e-10,max_iter=300,c=1e-4):
    ell=ell0; J,g=f(ell); step=step0; hist=[(ell,J,g)]; nevals=1
    for k in range(max_iter):
        if abs(g)<tol: break
        while True:
            move=-step*g
            if abs(move)>max_move: move=np.sign(move)*max_move
            trial=ell+move
            Jt,gt=f(trial); nevals+=1
            if Jt<=J+c*g*move: break
            step*=0.5
            if step<1e-16: return ell,hist,nevals,'line search failed'
        # secant (Barzilai-Borwein) estimate of the best step for next iteration
        dg=gt-g
        step = move/dg if move*dg>0 else 2*step
        ell,J,g=trial,Jt,gt; hist.append((ell,J,g))
    return ell,hist,nevals,('converged' if abs(g)<tol else 'max iterations')
