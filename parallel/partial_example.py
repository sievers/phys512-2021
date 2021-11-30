import numpy as np
from functools import partial

def gauss(x,x0,sigma):
    return np.exp(-0.5*(x-x0)**2/sigma**2)


x0=0.5
sigma=2.0
myfun=partial(gauss,x0=x0,sigma=sigma)
x=1.5

print('expected answer is ',gauss(x,x0,sigma))
#note the lack of arguments
print('partial gave us ',myfun(x)) 
