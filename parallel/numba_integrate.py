import numpy as np 
import numba as nb
import time

@nb.njit (parallel=True)
def integrate_exp(x0,x1,dx):
    n=int( (x1-x0)/dx)
    #n needs to be odd.  below code is a very fast way
    #of making sure that's true.  n&1 does bit-wise and
    #and so is zero if n is even, 1 if n odd.  ^ is xor
    #so (n&1)^1 is zero if n is odd, 1 if n even.  equivalent to
    #if iseven(n), n=n+1, but no if statements so very fast.
    n=n+((n&1)^1)
    dx=(x1-x0)/(n-1)
    tot=0.0
    for i in nb.prange(1,n-1):
        #now that I broke you in with the bit operators,
        #2+2*(n&1) will be 4 if n is odd, 2 if n is even
        tot=tot+(2+2*(i&1))*np.exp(x0+i*dx)
    tot=tot+np.exp(x0)+np.exp(x1)
    return dx*tot/3

@nb.njit (parallel=True)
def integrate_exp_v2(x0,x1,dx):
    n=int( (x1-x0)/dx)
    n=n+((n&1)^1)
    #tot0=0
    tot0=np.exp(x0)+np.exp(x1)
    dx=(x1-x0)/(n-1)
    tot=0.0
    for i in nb.prange(n//2):
        tot=tot+4*np.exp(x0+(2*i+1)*dx)
    tot2=0.0
    for i in nb.prange(n//2-1):
        tot2=tot2+2*np.exp(x0+(2*i+2)*dx)
    return dx*(tot+tot0+tot2)/3

def simpson(fun,x0,x1,dx):
    n=int( (x1-x0)/dx)
    n=n+((n&1)^1)
    x=np.linspace(x0,x1,n)
    dx=(x[1]-x[0])
    y=fun(x)
    tot=y[0]+y[-1]+4*np.sum(y[1::2])+2*np.sum(y[2:-1:2])
    return dx*tot/3

dx=0.000001
ans=integrate_exp(0,1,dx)
t1=time.time();ans=integrate_exp(0,1,dx);t2=time.time();dt=t2-t1;
print('we got ',ans,' expected ',np.exp(1)-np.exp(0),' in ',dt)
t1=time.time();ans=simpson(np.exp,0,1,dx);t2=time.time();dt2=t2-t1
print('vectorized answer is ',ans,' in ',dt2)
print('speed up is ',dt2/dt)


