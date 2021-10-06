import numpy as np
from matplotlib import pyplot as plt
def gauss(pars,x):
    offset=pars[0]
    amp=pars[1]
    x0=pars[2]
    sig=pars[3]
    return offset+amp*np.exp(-0.5*(x-x0)**2/sig**2)

def gauss_chisq(pars,x,y,noise=None):
    pred=gauss(pars,x)
    if noise is None:
        return np.sum((y-pred)**2)
    else:
        return np.sum (((y-pred)/noise)**2)

def mcmc(pars,step_size,x,y,fun,nstep=1000,noise=None):
    chi_cur=fun(pars,x,y,noise)
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    for i in range(nstep):
        trial_pars=pars+step_size*np.random.randn(npar)
        trial_chisq=fun(trial_pars,x,y,noise)
        delta_chisq=trial_chisq-chi_cur
        accept_prob=np.exp(-0.5*delta_chisq)
        accept=np.random.rand(1)<accept_prob
        if accept:
            pars=trial_pars
            chi_cur=trial_chisq
        chain[i,:]=pars
        chivec[i]=chi_cur
    return chain,chivec

x=np.linspace(-2,2,1001)
pars_true=np.asarray([1,3,0.5,0.5])
npar=len(pars_true)
y_true=gauss(pars_true,x)
y=y_true+np.random.randn(len(x))
plt.ion()
plt.clf()
plt.plot(x,y,'.')
plt.show()
step_size=np.asarray([0.1,0.1,0.1,0.1])
chain,chisq=mcmc(pars_true,step_size,x,y,gauss_chisq,nstep=20000)
step_size_new=np.std(chain,axis=0)
starting_pars=np.mean(chain,axis=0)+3*np.random.randn(npar)*step_size_new
chain2,chisq2=mcmc(starting_pars,step_size_new,x,y,gauss_chisq,nstep=20000)
