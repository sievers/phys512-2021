import numpy as np
import time
import scipy as sp
niter=100

n=5000
x=np.random.rand(n,n)
t1=time.time()
for i in range(niter):
    y=sp.fft.rfft2(x,workers=4)
t2=time.time()
print('took ',t2-t1)
