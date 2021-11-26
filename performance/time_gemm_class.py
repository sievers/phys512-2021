import numpy as np
import time

n=5000
if (0):
    A=np.random.randn(n,n)


niter=10
for i in range(niter):
    t1=time.time()
    B=np.dot(A,A)
    t2=time.time()
    print("took ",t2-t1," seconds to multiply")
