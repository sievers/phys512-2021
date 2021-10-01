import numpy as np
from matplotlib import pyplot as plt
def gauss(m,x):
    #get y=a+b*exp((x-x0)^2/2sig^2)
    a=m[0]
    b=m[1]
    x0=m[2]
    sig=m[3]
    expvec=np.exp(-0.5*(x-x0)**2/sig**2)
    y=a+b*expvec
    derivs=np.empty([len(x),len(m)])
    derivs[:,0]=1
    derivs[:,1]=expvec
    derivs[:,2]=b*(x-x0)*2/(2*sig**2)*expvec
    derivs[:,3]=b*(0.5*(x-x0)**2)*2/sig**3*expvec

    return y,derivs

def fit_newton(m,fun,x,y,niter=10):
    for i in range(niter):
        model,derivs=fun(m,x)
        r=y-model
        lhs=derivs.T@derivs
        rhs=derivs.T@r
        dm=np.linalg.inv(lhs)@rhs
        m=m+dm
        chisq=np.sum(r**2)
        print('on iteration ',i,' chisq is ',chisq,' with step ',dm)
    return m

x=np.linspace(-5,5,1001)
m_true=np.asarray([0.5,1.5,-0.5,1])
y_true,derivs=gauss(m_true,x)

plt.ion()
plt.clf()
plt.plot(x,y_true)
plt.show()
y=y_true+np.random.randn(len(x))
plt.plot(x,y,'.')

m0=m_true+np.random.randn(len(m_true))*1.5
m_fit=fit_newton(m0,gauss,x,y)
y_fit,derivs=gauss(m_fit,x)
plt.plot(x,y_fit)
