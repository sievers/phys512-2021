import numpy as np
from multiprocessing import Pool
from functools import partial

def integrate(lims,fun,dx):
    x1=lims[0]
    x2=lims[1]
    print('integrating ',x1,x2)
    n=int(np.ceil((x2-x1)/dx))
    if (n&1)==0:
        n=n+1
    x=np.linspace(x1,x2,n)
    dx=x[1]-x[0]
    tot=fun(x[0])+fun(x[-1])
    tot=tot+4*np.sum(fun(x[1::2]))
    tot=tot+2*np.sum(fun(x[2:-1:2]))
    return dx*tot/3


if __name__=='__main__':
    nthread=8
    x=np.linspace(0,1,nthread+1)
    lims=[None]*nthread
    for i in range(nthread):
        lims[i]=[x[i],x[i+1]]
    dx=0.001
    with Pool(nthread) as p:
        out=p.map(partial(integrate,fun=np.exp,dx=dx),lims)
    print('total is    ',np.sum(out))
    print('analytis is ',np.exp(x[-1])-np.exp(x[0]))

