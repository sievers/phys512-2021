import numpy as np
from matplotlib import pyplot as plt

x=np.linspace(-5,15,2001)
f=np.exp(-0.5*(x-2)**2/0.35**2)
g=0*x
g[(x>-2)&(x<0)]=1
f=f/f.sum()
g=g/g.sum()

h=np.fft.irfft(np.fft.rfft(f)*np.fft.rfft(g),len(x))

plt.ion()
plt.clf()
plt.plot(x,f)
plt.plot(x,g)
plt.plot(x,h)
plt.show()
