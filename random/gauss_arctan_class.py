import numpy as np

y=np.linspace(-np.pi/2,np.pi/2,1001)
cents=(y[1:]+y[:-1])/2
#for gaussian, prob=exp(-0.5*x**2) goes to exp(-0.5*(tan(y))**2)/cos^2(y)
pp=np.exp(-0.5*np.tan(cents)**2)/np.cos(cents)**2
plt.clf()
plt.plot(cents,pp)
plt.show()

n=1000000
y=np.pi*(np.random.rand(n)-0.5)
h=np.random.rand(n)*1.22
accept=h<np.exp(-0.5*np.tan(y)**2)/np.cos(y)**2
t=np.tan(y[accept])
