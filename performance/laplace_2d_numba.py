import numpy as np
import numba as nb
import time

@nb.njit(parallel=True)
def laplace_kernel(V):
    ans=np.empty(V.shape)
    n=V.shape[0]
    m=V.shape[1]
    for i in nb.prange(1,n-1):
        for j in range(1,m-1):
            ans[i,j]=V[i,j]-np.log(0.25*(np.exp(V[i,j-1])+np.exp(V[i,j+1])+np.exp(V[i-1,j])+np.exp(V[i+1,j])))
    return ans


n=1000
x=np.random.rand(n,n)

y1=laplace_kernel(x)
t1=time.time()
y1=laplace_kernel(x)
t2=time.time()
print('numba kernel took ',t2-t1)

t1=time.time()
y2=x-0.25*(np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1))
t2=time.time()
print('roll took ',t2-t1)
print(np.std( (y2-y1)[1:-1,1:-1]))

