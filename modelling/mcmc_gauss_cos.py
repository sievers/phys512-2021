import numpy as np

def gauss_fun(pars,x):
    pred=pars[0]+pars[1]*np.exp(-0.5*pars[2]*(x-pars[3])**2)
    return pred
def gauss_chisq(pars,data):
    x=data[0]
    y=data[1]
    errs=data[2]
    pred=gauss_fun(pars,x)
    chisq=np.sum((pred-y)**2/errs**2)
    return chisq

def get_derivs(pars,x,fun,dpar=None):
    if dpar is None:
        dpar=0.001*pars
    npar=len(pars)
    derivs=np.zeros([len(x),npar])
    for i in range(npar):
        pp=pars.copy()
        pp[i]=pp[i]+dpar[i]
        vec1=fun(pp,x)
        pp=pars.copy()
        pp[i]=pp[i]-dpar[i]
        vec2=fun(pp,x)
        derivs[:,i]=(vec1-vec2)/(2*dpar[i])
    return derivs
def run_chain(fun,pars,data,sigs,nstep=5000,T=1.0,sampfun=None):
    npar=len(pars)
    chain=np.zeros([nstep,npar])

    chisq=fun(pars,data)
    chivec=np.zeros(nstep)
    for i in range(nstep):
        if sampfun is None:
            pars_trial=pars+sigs*np.random.randn(npar)
        else:
            pars_trial=pars+sampfun(sigs)
        chisq_trial=fun(pars_trial,data)
        dchi=(chisq_trial-chisq)/T
        if np.random.rand(1)<np.exp(-0.5*dchi):
            pars=pars_trial
            chisq=chisq_trial
        chain[i,:]=pars
        chivec[i]=chisq
    return chain,chivec



x=np.linspace(-5,5,2001)
pars_true=np.asarray([1.5,2.5,1.0,0])
y_true=gauss_fun(pars_true,x)
sig=0.1
y=y_true+sig*np.random.randn(len(x))
pars=pars_true.copy()
pars=pars+0.2*np.random.randn(len(pars)) #add some noise
dpar=0.01*pars_true
dpar[-1]=0.001
Ninv=np.eye(len(x))/sig**2

for i in range(10):
    pred=gauss_fun(pars,x)
    derivs=get_derivs(pars,x,gauss_fun,dpar)
    resid=y-pred
    lhs=derivs.T@Ninv@derivs
    rhs=derivs.T@Ninv@resid
    lhs_inv=np.linalg.inv(lhs)
    step=lhs_inv@rhs
    pars=pars+step
    print(pars)

sigs=np.sqrt(np.diag(lhs_inv))

data=[x,y,sig]
chain,chivec=run_chain(gauss_chisq,pars,data,0.5*sigs,nstep=15000)
