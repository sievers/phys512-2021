import numpy as np
import time
def time_fft(n,niter=20):
    x=np.random.randn(n)
    t1=time.time()
    for i in range(niter):
        y=np.fft.fft(x)
    t2=time.time()
    return (t2-t1)/niter


n=2**20
#pads=[1,2,14]
pads=np.arange(1,8)
nloop=10
t_ref=time_fft(n)
print('reference time is ',t_ref)
for pad in pads:
    t=time_fft(n+pad)
    print('for pad of ',pad,' time increase by a factor of ',t/t_ref)

