import numpy as np

x=42
eps=2**-52
dx=eps**(1/3)

x2=x+dx
dx=x2-x

f1=np.exp(x)
f2=np.exp(x2)
deriv=(f2-f1)/dx
print('derivative is ',deriv,' with fractional error ',deriv/f1-1)


f0=np.exp(x-dx)
deriv=(f2-f0)/(2*dx)
print('derivative is ',deriv,'with new error ',(deriv/f1-1))
