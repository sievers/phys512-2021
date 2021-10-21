import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=1000
npart=10
x=np.arange(n)



x_part=np.asarray((np.random.rand(npart)*n),dtype='int')
amps=2.0/np.random.rand(npart)
tau=5
kernel=np.exp(-x/tau)
kernel=kernel/kernel.sum()
y_true=np.zeros(n)
y_true[x_part]=amps
y_conv=np.fft.irfft(np.fft.rfft(kernel)*np.fft.rfft(y_true))
y_obs=y_conv+np.random.randn(n)

y_deconv=np.fft.irfft(np.fft.rfft(y_obs)/np.fft.rfft(kernel))

mf=np.fft.irfft(np.conj(np.fft.rfft(kernel))*np.fft.rfft(y_obs))
#plt.clf();plt.plot(y_deconv);plt.plot(mf);plt.plot(y_true);plt.show()
normfac=np.sum(kernel**2) #we've assumed white noise here
mf_norm=mf/normfac

plt.clf()
plt.plot(y_obs)
plt.show()
plt.savefig('part_detector_obs.png')

plt.clf();plt.plot(y_deconv);plt.show()
plt.savefig('part_detector_deconv.png')

plt.clf()
plt.plot(y_deconv)
plt.plot(y_obs)
plt.plot(y_true)
plt.legend(['Observed data','Deconvolved','Input Signal'])
plt.savefig('deconvolved_1d.png')
