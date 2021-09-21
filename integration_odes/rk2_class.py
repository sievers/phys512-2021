import numpy as np
from matplotlib import pyplot as plt

def f(x,y):
    #dy/dx=-y has solution y=exp(-x)
    return -y

def sho(x,y):
    #y is now a vector
    #y[0] is y, but y[1] is y'
    #simple harmonic oscillator, y''=-y
    #y'=z
    #z'=-y
    
    return np.asarray([y[1],-y[0]])


def rk2_step(fun,x,y,h):
    k0=fun(x,y)*h
    k1=fun(x+h,y+k0)*h
    return 0.5*(k0+k1)

def rk2_step2(fun,x,y,h):
    k0=fun(x,y)*h
    k1=fun(x+h/2,y+k0/2)*h
    return k1

def rk3_step(fun,x,y,h):
    y1=rk2_step(fun,x,y,h)
    y2a=rk2_step(fun,x,y,h/2)
    y2b=rk2_step(fun,x+h/2,y+y2a,h/2)
    y2=y2a+y2b

    #y1=y_true+err
    #y2=y_true+err/4
    #4y2=4y_true+err
    #4y2-y1=3y_true
    #y_true=(4y2-y1)/3
    return (4*y2-y1)/3

def rk4_step(fun,x,y,h):
    k0=fun(x,y)*h
    k1=fun(x+h/2,y+k0/2)*h
    k2=fun(x+h/2,y+k1/2)*h
    k3=fun(x+h,y+k2)*h
    return (k0+2*k1+2*k2+k3)/6

nstep=1000
x=np.linspace(0,10,nstep+1)
if False:
    y=0*x #could be y=np.zeros(nstep+1) or y=np.empty(nstep+1)
    y[0]=1
    y_pred=np.exp(x)
    fun=f

else:
    y=np.zeros([2,nstep+1])
    y[0,0]=1
    y[1,0]=0
    y_pred=np.vstack([np.cos(x),-np.sin(x)])
    fun=sho

for i in range(nstep):
    h=x[i+1]-x[i]
    y[:,i+1]=y[:,i]+rk4_step(fun,x[i],y[:,i],h)



plt.ion()
plt.clf()
plt.plot(x,y[0,:])
plt.plot(x,y_pred[0,:])
print(np.std(y-y_pred))
