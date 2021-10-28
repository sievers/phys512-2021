import numpy as np

def lorentz_dev(n):
    x=np.random.rand(n)
    return np.tan(np.pi*(x-0.5))

def gauss_from_lorentz(x):
    accept_prob=(1/2.23)*np.exp(-0.5*x**2)/(1/(1+x**2))     #ratio of gaussian pdf to lorentzian pdf
    assert(np.max(accept_prob)<=1)
    accept=np.random.rand(len(accept_prob))<accept_prob
    return x[accept]

n=1000000
y=lorentz_dev(n)


yy=y[np.abs(y)<20]
a,b=np.histogram(yy,200)
bb=0.5*(b[1:]+b[:-1])
plt.clf()
plt.plot(bb,a/a.max())
plt.plot(bb,1/(1+bb**2))
plt.plot(bb,np.exp(-0.5*bb**2))
plt.show()


z=gauss_from_lorentz(y)
print('accept fraction was ',len(z)/len(y))
zz=np.random.randn(len(z))
plt.clf();
z.sort()
zz.sort()
plt.plot(z)
plt.plot(zz)
plt.show()
