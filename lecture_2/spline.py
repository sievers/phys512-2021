import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


xmin=-1
xmax=1
npt=9
x=np.linspace(xmin,xmax,npt)
y=np.tanh(x)
y[-2:]=y[-3]
xx=np.linspace(x[0],x[-1],2001)



spln=interpolate.splrep(x,y)
yy=interpolate.splev(xx,spln)

plt.clf()
plt.plot(x,y,'*')
plt.plot(xx,yy)
plt.plot(xx,np.tanh(xx))
plt.show()
