import time
import numpy as np
from multiprocessing import Pool
from functools import partial

def integrate(id,fun,range,dx,nproc):
    edges=np.linspace(range[0],range[1],nproc+1)
    xmin=edges[id]
    xmax=edges[id+1]
    print('I am ',id,' out of ',nproc,' with range ',xmin,xmax)
    nx=int(np.ceil((xmax-xmin)/dx))
    if (nx%2==0):
        nx=nx+1
    x=np.linspace(xmin,xmax,nx)
    y=fun(x)
    mydx=x[1]-x[0]
    my_sum=y[0]+y[-1]+4*np.sum(y[1::2])+2*np.sum(y[2:-1:2])
    return mydx*my_sum/3


if __name__=='__main__':
    nproc=4
    dx=1/256
    with Pool(nproc) as p:
        out=p.map(partial(integrate,fun=np.exp,range=[0,1],dx=dx,nproc=nproc),range(nproc))
    print('integral is ',np.sum(out))

    
