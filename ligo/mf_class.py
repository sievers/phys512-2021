import numpy as np
from matplotlib import pyplot as plt

n=1000
npart=10

y_true=np.zeros(n)
for i in range(npart):
    t=int(np.random.rand(1)*n)
    y_true[t]=1/np.random.rand(1)

x=np.arange(n)
tau=10
kernel=np.exp(-x/tau)
y_nonoise=np.fft.irfft(np.fft.rfft(kernel)*np.fft.rfft(y_true))
y_obs=y_nonoise+np.random.randn(n)

plt.ion()
y_deconv=np.fft.irfft(np.fft.rfft(y_obs)/np.fft.rfft(kernel))

mf=np.fft.irfft(np.fft.rfft(y_obs)*np.conj(np.fft.rfft(kernel)))
