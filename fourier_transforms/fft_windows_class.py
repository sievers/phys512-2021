import numpy as np


t=np.arange(1000)
y=np.cos(t*2*np.pi/len(t)*5.3)
yft=np.fft.rfft(y)
plt.clf()
plt.semilogy(np.abs(yft),'.')
#plt.plot(np.roll(y,len(y)//2))
plt.show()

