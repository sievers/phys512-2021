import numpy as np
from matplotlib import pyplot as plt
def gauss2d(dims,sig):
    xvec=np.fft.fftfreq(dims[0])*dims[0]
    yvec=np.fft.fftfreq(dims[1])*dims[1]
    
    rsqr=np.outer(xvec**2,np.ones(dims[1]))+np.outer(np.ones(dims[0]),yvec**2)
    return np.exp(-0.5*rsqr/sig**2)



myim=plt.imread('meerkat_small.png')

plt.ion()

red=myim[:,:,0]

kernel=gauss2d(red.shape,2.0)
redft=np.fft.fft2(red)
kernelft=np.fft.fft2(kernel)
red_smooth=np.fft.ifft2(kernelft*redft)
mynoise=1e-5*np.random.randn(red_smooth.shape[0],red_smooth.shape[1])


kft_inv=1/kernelft
thresh=10
mask=np.abs(kft_inv)>thresh*np.abs(kft_inv[0]) #only use bits of the kernel we think are OK
kft_inv[mask]=0 #and mask out everything we think we lost


plt.figure(1)
plt.clf()
plt.imshow(np.real(red_smooth))
plt.show()

redft2=np.fft.fft2(red_smooth+mynoise)
red_back=np.fft.ifft2(redft2/kernelft)

red_back2=np.fft.ifft2(redft2*kft_inv)
plt.figure(2)
plt.clf();
plt.imshow(np.real(red_back2),vmin=0,vmax=1)
plt.show()
