import numpy as np
import numba as nb
from scipy import fft
import timeit

def myfft(f):
    if len(f)==1:
        return f
    f_even=f[0::2]
    f_odd=f[1::2]
    N=len(f)
    kvec=np.arange(N//2)
    twid=np.exp(-2J*np.pi*kvec/N)
    
    ft_even=myfft(f_even)
    ft_odd=myfft(f_odd)
    return np.hstack([ft_even+twid*ft_odd,ft_even-twid*ft_odd])

@nb.njit
def nbfft(f):    
    #if (len(f)==1:
    #        return f
    #if len(f)==2:
    #    tmp=f[0]-f[1]
    #    f[0]=f[0]+f[1]
    #    f[1]=tmp
    #    return f
    if len(f)==4: #hard-wired FFT of four numbers.  saves us two rounds of recursion
        a=f[0]+f[1]+f[2]+f[3]
        b=f[0]-1j*f[1]-f[2]+1j*f[3]
        c=f[0]-f[1]+f[2]-f[3]
        d=f[0]+1j*f[1]-f[2]-1j*f[3]
        f=np.empty_like(f)
        f[0]=a
        f[1]=b
        f[2]=c
        f[3]=d
        return f
           
    f_even=f[0::2]
    f_odd=f[1::2]
    N=len(f)
    M=N//2
    #twid=np.empty(N//2)
    #twid[0]=1
    fac=np.exp(-2J*np.pi/N)
    #for i in np.arange(1,N//2):
    #    twid[i]=twid[i-1]*fac
    ft_even=nbfft(f_even)
    ft_odd=nbfft(f_odd)
    out=np.empty(N,dtype='complex')
    twid=1    
    for i in np.arange(M):
        out[i]=ft_even[i]+twid*ft_odd[i]
        out[i+M]=ft_even[i]-twid*ft_odd[i]
        twid=twid*fac
    return out


N=1024*1024
x=np.random.randn(N)+1J*np.random.randn(N)
t1=time.time()
xft=np.fft.fft(x)
t2=time.time()
print('numpy fft took ',t2-t1)

if (1==0):
    t1=time.time()
    xft2=myfft(x)
    t2=time.time()
    print('scat is ',np.std(xft-xft2),' in time ',t2-t1)


xft3=nbfft(x)
t1=time.time()
xft3=nbfft(x)
t2=time.time()
print('numscat is ',np.std(xft-xft3),' in time ',t2-t1)
