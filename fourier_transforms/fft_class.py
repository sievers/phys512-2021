import numpy as np
import time

def fft(x):
    if len(x)==1:
        return x
    xeven=x[::2]
    xodd=x[1::2]
    
    xe_ft=fft(xeven)
    xo_ft=fft(xodd)
    N=len(x)
    twid=np.exp(-2J*np.pi*np.arange(N//2)/N)
    first_half=xe_ft+twid*xo_ft
    second_half=xe_ft-twid*xo_ft
    return np.hstack([first_half,second_half])

x=np.random.randn(2**17)
t1=time.time()
xft=fft(x)
t2=time.time()
print(t2-t1)
#print(xft)
#print(np.fft.fft(x))
print('err is ',np.mean(np.abs(xft-np.fft.fft(x))))
