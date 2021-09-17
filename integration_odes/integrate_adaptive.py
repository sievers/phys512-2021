import numpy as np
def lorentz(x):
    return 1/(1+x**2)

def integrate_adaptive(fun,x0,x1,tol):
    print('integrating between ',x0,x1)
    #hardwire to use simpsons
    x=np.linspace(x0,x1,5)
    y=fun(x)
    dx=(x1-x0)/(len(x)-1)
    area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
    err=np.abs(area1-area2)
    if err<tol:
        return area2
    else:
        xmid=(x0+x1)/2
        left=integrate_adaptive(fun,x0,xmid,tol/2)
        right=integrate_adaptive(fun,xmid,x1,tol/2)
        return left+right


x0=-100
x1=100
if False:
    ans=integrate_adaptive(np.exp,x0,x1,1e-7)
    print(ans-(np.exp(x1)-np.exp(x0)))
else:
    ans=integrate_adaptive(lorentz,x0,x1,1e-7)
    print(ans-(np.arctan(x1)-np.arctan(x0)))

    
