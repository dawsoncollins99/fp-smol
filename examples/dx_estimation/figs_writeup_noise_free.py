"""
Figures for docs/dx_estimation/writeup_noise_free (Dx_steady_estimation.tex).

Usage (from the repo root): python examples/dx_estimation/figs_writeup_noise_free.py
"""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import *

OUT = Path(__file__).resolve().parents[2] / "docs" / "dx_estimation" / "writeup_noise_free" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size":9,"font.family":"serif","mathtext.fontset":"cm","axes.grid":True,"grid.alpha":0.3})
m=Model(8000); r=np.sqrt(m.x)

# Fig: random walks of droplets in x (drift + diffusion), no settling
rng=np.random.default_rng(1)
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6),sharey=True)
tt=np.linspace(0,150,301); dt=tt[1]-tt[0]
for k,(D,title) in enumerate([(1.0,r"weak turbulence, $D_x=1$"),(10.0,r"strong turbulence, $D_x=10$")]):
    for j in range(25):
        w=np.cumsum(np.r_[0,rng.normal(0,np.sqrt(2*D*dt),len(tt)-1)])
        x=200+a*s*tt+w
        ax[k].plot(tt,x,lw=0.6,alpha=0.8)
    ax[k].plot(tt,200+a*s*tt,"k--",lw=1.2,label=r"growth with steady $s$ only")
    ax[k].set(xlabel="time (s)",title=title)
ax[0].set_ylabel(r"$x=r^2$ ($\mu$m$^2$)"); ax[0].legend(loc="upper left",fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "f_walks.pdf")

# Fig: drift and settling time
drift=a*(s-(A_k/r-B_k/r**3))
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
ax[0].loglog(r,drift); ax[0].axhline(a*s,color="k",ls=":",label=r"$a\,s$ (large droplets)")
ax[0].axvline(np.sqrt(3*B_k/A_k),color="C3",ls="--",lw=0.8,label=r"$r\approx1\ \mu$m bottleneck")
ax[0].set(xlabel=r"radius $r$ ($\mu$m)",ylabel=r"$A$ ($\mu$m$^2$/s)",title=r"(a) Growth rate $A=dx/dt$"); ax[0].legend(fontsize=7)
ax[1].semilogy(r,1/(k_settle*m.x)); ax[1].set(xlabel=r"radius $r$ ($\mu$m)",ylabel="seconds",title=r"(b) Typical time before falling out, $1/\lambda$")
fig.tight_layout(); fig.savefig(OUT / "f_drift.pdf")

# Fig: steady DSDs
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
for D in [0.01,0.1,1,10,100]:
    N,_=m.solve(D); ax[0].plot(m.x,N,label=f"{D:g}")
    nr=2*r*N; ax[1].plot(r,nr/np.trapezoid(nr,r),label=f"{D:g}")
ax[0].set(xscale="log",xlim=(0.05,400),xlabel=r"$x=r^2$ ($\mu$m$^2$, log scale)",ylabel=r"$N$ (cm$^{-3}\,\mu$m$^{-2}$)",title=r"(a) $N(x)$, the quantity we solve for")
ax[1].set(xlim=(0,25),xlabel=r"radius $r$ ($\mu$m)",ylabel=r"fraction per $\mu$m",title="(b) Same, as a distribution of radius")
ax[1].legend(title=r"$D_x$ ($\mu$m$^4$/s)",fontsize=7,title_fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "f_dsd.pdf")

# loss landscape
Nobs,_=m.solve(10.0)
Ds=np.logspace(-3,3,61); sc=np.array([loss_and_grad(m,np.log(D),Nobs) for D in Ds])
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
ax[0].loglog(Ds,sc[:,0],".-",ms=3); ax[0].axvline(10,color="k",ls="--",lw=0.8)
ax[0].set(xlabel=r"$D_x$ ($\mu$m$^4$/s)",ylabel=r"$\mathcal{J}$",title=r"(a) Loss, true $D_x=10$")
ax[1].semilogx(Ds,sc[:,1],".-",ms=3); ax[1].axhline(0,color="k",lw=0.5); ax[1].axvline(10,color="k",ls="--",lw=0.8)
ax[1].set(xlabel=r"$D_x$ ($\mu$m$^4$/s)",ylabel=r"$d\mathcal{J}/d\ell$",title=r"(b) Gradient with respect to $\ell=\ln D_x$")
fig.tight_layout(); fig.savefig(OUT / "f_loss.pdf")

# adjoint at 20
D=20.; N,lu=m.solve(D); scale=(Nobs**2).sum(); psi=lu.solve(-(N-Nobs)/scale,trans="T")
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
ax[0].plot(r,Nobs,label=r"data ($D_x=10$)"); ax[0].plot(r,N,label=r"model ($D_x=20$)")
ax[0].set(xlim=(0,20),xlabel=r"radius $r$ ($\mu$m)",ylabel=r"$N$ (cm$^{-3}\,\mu$m$^{-2}$)",title="(a) Data and a wrong guess"); ax[0].legend(fontsize=7)
ax[1].plot(r,psi); ax[1].axhline(0,color="k",lw=0.5)
ax[1].set(xlim=(0,20),xlabel=r"radius $r$ ($\mu$m)",ylabel=r"$\psi$",title=r"(b) Adjoint $\psi$ at $D_x=20$")
fig.tight_layout(); fig.savefig(OUT / "f_adjoint.pdf")

# optimization path from 0.1
f=lambda l: loss_and_grad(m,l,Nobs)
ell,h,ne,st=gradient_descent(f,np.log(0.1)); h=np.array(h)
fig,ax=plt.subplots(1,2,figsize=(7.0,2.5))
ax[0].semilogy(np.exp(h[:,0]),"o-",ms=3); ax[0].axhline(10,color="k",ls="--",lw=0.8)
ax[0].set(xlabel="iteration",ylabel=r"estimate of $D_x$",title=r"(a) Estimate, starting from $D_x=0.1$")
ax[1].semilogy(h[:,1],"o-",ms=3); ax[1].set(xlabel="iteration",ylabel=r"$\mathcal{J}$",title="(b) Loss")
fig.tight_layout(); fig.savefig(OUT / "f_path.pdf")
print("iters from 0.1:",len(h)-1)

# sweep
trues=[0.01,0.1,1,10,100]; starts=[1e-3,1e-2,1e-1,1,10,100,1e3]
fig,axes=plt.subplots(1,5,figsize=(7.2,2.3),sharey=True)
for axx,Dt in zip(axes,trues):
    No,_=m.solve(Dt); ff=lambda l: loss_and_grad(m,l,No)
    for D0 in starts:
        e,hh,_,_=gradient_descent(ff,np.log(D0)); hh=np.array(hh)
        axx.plot(np.exp(hh[:,0]),".-",lw=0.8,ms=2,label=f"{D0:g}")
    axx.axhline(Dt,color="k",ls="--",lw=0.7); axx.set(yscale="log",title=f"true {Dt:g}",xlabel="iteration")
axes[0].set_ylabel(r"estimate ($\mu$m$^4$/s)")
axes[-1].legend(title="start",fontsize=5.5,title_fontsize=6,loc="lower right")
fig.tight_layout(w_pad=0.3); fig.savefig(OUT / "f_sweep.pdf")
