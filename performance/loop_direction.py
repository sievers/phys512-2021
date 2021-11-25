import numpy as np
import numba as nb
import time


@nb.njit(parallel=True)
def mycopy(x):
    ans=np.empty(x.shape)
    nx=x.shape[0]
    ny=x.shape[1]
    if True:
        for i in nb.prange(nx):
            for j in range(ny):
                ans[i,j]=x[i,j]
    else:
        for j in nb.prange(ny):
            for i in range(nx):
                ans[i,j]=x[i,j]
    return ans




n=10000
x=np.random.rand(n,n)
for i in range(10):
    t1=time.time()
    y=mycopy(x)
    t2=time.time()
    print('copied in ',t2-t1,' seconds on iteration ',i)


