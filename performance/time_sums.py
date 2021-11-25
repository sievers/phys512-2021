import numpy as np
import time

n=1000
a=np.random.rand(n,n)
b=np.random.rand(n,n)
c=np.empty([n,n])
t1=time.time()
for i in range(n):
    for j in range(n):
        c[i,j]=a[i,j]+b[i,j]
t2=time.time()
print('time to sum via loops is ',t2-t1)

t1=time.time()
c2=a+b
t2=time.time()
print('time to sum via numpy is ',t2-t1)
print('diff is ',np.std(c2-c))
