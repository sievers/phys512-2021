import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def generate_pulses(N,frac,soft=0.01):
    vec=np.random.rand(N)
    mask=vec>frac
    vec[mask]=0
    vec=vec/frac
    ans=(1/(vec+soft))**0.75
    ans[mask]=0
    return ans


N=1000
frac=0.02
y=generate_pulses(N,frac)

tau=N*frac*2 #set time constant to be close to expected spacing
x=np.arange(N)
kernel=np.exp(-x/tau)
#kernel=kernel/kernel.sum()
y_conv=np.fft.irfft(np.fft.rfft(y)*np.fft.rfft(kernel))

n=0.1*np.std(y_conv)
y_obs=y_conv+n*np.random.randn(N)
y_back=np.fft.irfft(np.fft.rfft(y_obs)/np.fft.rfft(kernel))





plt.figure(1)
plt.clf()
plt.plot(y)
plt.plot(y_obs)
plt.plot(y_conv)
plt.legend(['Truth','Noisy out','Noiseless out'])
plt.show()
plt.savefig('conv_signal_out.png')


plt.figure(2)
plt.clf()
plt.plot(y_back)
plt.plot(y,'.')
plt.show()
plt.legend(['Reconstructed','Truth'])
plt.savefig('conv_signal_reconstructed.png')
