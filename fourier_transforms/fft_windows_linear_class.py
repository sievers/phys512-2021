import numpy as np


t=np.arange(1000)
tt=(t-len(t)//2)
tt=tt/len(tt)*2
win=0.5+0.5*np.cos(tt*np.pi)

y_org=np.random.randn(len(t))
y=y_org+(0.03*(t-np.mean(t)))
yft=np.fft.rfft(y)
yft_smooth=0.5*yft-0.25*np.roll(yft,1)-0.25*np.roll(yft,-1)
yback=np.fft.ifft(yft_smooth)
plt.figure(1)
plt.clf()
plt.plot(y*win)
plt.show()
plt.figure(2)
normfac=np.mean(win**2)

plt.clf()
plt.loglog(np.abs(yft))
plt.plot(np.abs(np.fft.fft(y_org)))
plt.plot(np.abs(yft_smooth)/np.sqrt(normfac))

plt.show()
