"""
Figures for docs/dx_estimation/writeup_noisy (Dx_noisy_estimation.tex).

Needs the Monte Carlo results: run run_monte_carlo.py first.
Usage (from the repo root): python examples/dx_estimation/figs_writeup_noisy.py
"""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import lgamma
from noisy import *
from run_monte_carlo import load_result

OUT = Path(__file__).resolve().parents[2] / "docs" / "dx_estimation" / "writeup_noisy" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size":9,"font.family":"serif","mathtext.fontset":"cm","axes.grid":True,"grid.alpha":0.3})
m=Model(8000)
R=load_result

# --- noisy histograms at three V (display bins of 80 cells = 9 um^2)
rng=np.random.default_rng(7); N10,_=m.solve(10.0)
fig,ax=plt.subplots(1,3,figsize=(7.2,2.4),sharey=True)
for a,V in zip(ax,[0.01,0.1,1.0]):
    ob=Obs(m,80,V); mu=ob.mu(N10); c=rng.poisson(mu); w=80*m.dx
    xe=m.xf[::80]
    a.stairs(c/(V*w),xe,color="C0",lw=1,label="sampled")
    a.plot(m.x,N10,"k",lw=1.2,label="true $N$")
    a.set(xlim=(0,250),title=f"$V={V:g}$ cm$^3$",xlabel=r"$x=r^2$ ($\mu$m$^2$)")
    a.text(0.95,0.6,f"~{mu.sum():.0f} droplets",transform=a.transAxes,ha="right",fontsize=8)
ax[0].set_ylabel(r"cm$^{-3}\,\mu$m$^{-2}$"); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "f_samples.pdf")

# --- Poisson pmfs
fig,ax=plt.subplots(1,2,figsize=(7.0,2.3))
for a,mu in zip(ax,[4,100]):
    k=np.arange(0,int(mu+5*np.sqrt(mu))+3)
    p=np.exp(k*np.log(mu)-mu-np.array([lgamma(kk+1) for kk in k]))
    a.bar(k,p,width=0.8 if mu<10 else 1.0,color="C0",alpha=0.8)
    a.axvline(mu,color="k",lw=1); a.axvspan(mu-np.sqrt(mu),mu+np.sqrt(mu),color="C1",alpha=0.2,label=r"mean $\pm\sqrt{\mu}$")
    a.set(xlabel="count $c$",ylabel="probability",title=f"$\\mu={mu}$: spread $\\pm${np.sqrt(mu):.0f} ({100/np.sqrt(mu):.0f}%)")
    a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "f_poisson.pdf")

# --- single sample: loss near minimum with curvature parabola
obs=Obs(m,8,0.1); mu_t=obs.mu(N10); c=np.random.default_rng(3).poisson(mu_t).astype(float)
f=lambda l: poisson_loss_grad(obs,l,c)
ell,J,g,it,st=descend(f,0.0); sd=curvature_sd(f,ell); Jpp=2/sd**2
ls=np.linspace(ell-4*sd,ell+4*sd,41); Js=np.array([f(l)[0] for l in ls])
fig,ax=plt.subplots(figsize=(4.6,2.8))
ax.plot(np.exp(ls),Js-J,"o",ms=3,label="deviance (minus its minimum)")
ax.plot(np.exp(ls),0.5*Jpp*(ls-ell)**2,"-",lw=1,label="parabola from curvature")
ax.axvline(10,color="k",ls="--",lw=0.8,label="true $D_x$")
ax.axvspan(np.exp(ell-sd),np.exp(ell+sd),color="C2",alpha=0.15,label=r"estimate $\pm$ 1 sd")
ax.axhline(1,color="gray",lw=0.6,ls=":")
ax.set(xlabel=r"$D_x$ ($\mu$m$^4$/s)",ylabel=r"$\mathcal{J}-\mathcal{J}_{\min}$",ylim=(-0.5,17))
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "f_single.pdf")
print("single:",np.exp(ell),sd,J,it)

# --- histograms of estimates for D=10 at three V
fig,ax=plt.subplots(1,3,figsize=(7.2,2.4))
for a,V in zip(ax,[0.01,0.1,1.0]):
    r=R(10,V,0); e=r["ell"]-np.log(10); s=1/np.sqrt(r["I"])
    a.hist(e,bins=20,density=True,color="C0",alpha=0.7,label="200 samples")
    z=np.linspace(-4*s,4*s,200); a.plot(z,np.exp(-z**2/(2*s*s))/np.sqrt(2*np.pi)/s,"k",lw=1.2,label="prediction")
    a.set(title=f"$V={V:g}$ cm$^3$",xlabel=r"$\hat\ell-\ell_{\rm true}$")
ax[0].legend(fontsize=7); ax[0].set_ylabel("density")
fig.tight_layout(); fig.savefig(OUT / "f_hist.pdf")

# --- scaling with V
fig,ax=plt.subplots(figsize=(4.4,2.9))
Vs=np.array([0.01,0.1,1.0])
for D,col in [(10,"C0"),(0.1,"C3")]:
    mc=[R(D,V,0)["ell"].std(ddof=1) for V in Vs]; pr=[1/np.sqrt(R(D,V,0)["I"]) for V in Vs]
    ax.loglog(Vs,mc,"o",color=col,label=f"observed, $D_x={D:g}$")
    ax.loglog(Vs,pr,"-",color=col,lw=1,label=f"predicted, $D_x={D:g}$")
ax.set(xlabel=r"sample volume $V$ (cm$^3$)",ylabel=r"spread of $\hat\ell$ (≈ relative error)")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "f_scaling.pdf")

# --- per-run error bars
r=R(10,0.1,0); n=40
fig,ax=plt.subplots(figsize=(7.0,2.4))
D_hat=np.exp(r["ell"][:n]); lo=np.exp(r["ell"][:n]-r["sd"][:n]); hi=np.exp(r["ell"][:n]+r["sd"][:n])
inside=(lo<=10)&(hi>=10)
ax.errorbar(np.arange(n)[inside],D_hat[inside],yerr=[D_hat[inside]-lo[inside],hi[inside]-D_hat[inside]],fmt="o",ms=3,color="C0",capsize=2,label="interval contains truth")
ax.errorbar(np.arange(n)[~inside],D_hat[~inside],yerr=[D_hat[~inside]-lo[~inside],hi[~inside]-D_hat[~inside]],fmt="o",ms=3,color="C3",capsize=2,label="interval misses truth")
ax.axhline(10,color="k",ls="--",lw=0.8)
ax.set(xlabel="sample number",ylabel=r"$\hat D_x\ \pm$ 1 sd",title=r"First 40 samples, true $D_x=10$, $V=0.1$ cm$^3$")
ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(OUT / "f_errorbars.pdf")
print("inside first 40:",inside.mean())

# --- detection limit, D=0.1, V=1
fig,ax=plt.subplots(1,4,figsize=(7.2,2.3),sharey=True)
for a,rd in zip(ax,[0,1,2,3]):
    r=R(0.1,1,rd); e=r["ell"]-np.log(0.1); s=1/np.sqrt(r["I"])
    a.hist(np.clip(e,-2,2),bins=np.linspace(-2,2,33),density=True,color="C0",alpha=0.7)
    z=np.linspace(-2,2,300); a.plot(z,np.exp(-z**2/(2*s*s))/np.sqrt(2*np.pi)/s,"k",lw=1.1)
    a.set(title=("no cutoff" if rd==0 else f"cutoff {rd} $\\mu$m"),xlabel=r"$\hat\ell-\ell_{\rm true}$")
ax[0].set_ylabel("density")
fig.tight_layout(); fig.savefig(OUT / "f_cutoff.pdf")
