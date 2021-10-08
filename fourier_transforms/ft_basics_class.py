import numpy as np
from matplotlib import pyplot as plt
#definition is F(k)=sum f(x) exp(-2 pi i k x/N)

N=16
y=np.random.randn(N)
yft=np.fft.fft(y)
myft=np.zeros(N,dtype='complex')
x=np.arange(N)
for k in range(N):
    myft[k]=np.sum(y*np.exp(-2*np.pi*1J*k*x/N))
yback=np.zeros(N,dtype='complex')

k=np.arange(N)
for x in range(N):
    yback[x]=np.sum(myft*np.exp(2J*np.pi*k*x/N))/N

x=np.linspace(-10,10,1001)
sig=0.1
y=np.exp(-0.5*(x**2)/sig**2)
yft=np.fft.fft(y)
plt.ion()
plt.clf()
plt.plot(np.abs(yft))

y=np.ones(N)
#y[0]=1
yft=np.fft.fft(y)
plt.clf();plt.plot(np.abs(yft));plt.show()
