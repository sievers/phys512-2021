import numpy as np

x=np.linspace(0,1,41)
dx=x[1]-x[0]
y=np.sin(x)

area=dx*(y[0]/2+y[-1]/2+np.sum(y[1:-1]))
print(area-(np.cos(x[0])-np.cos(x[-1])))


yy=y[::2]
area1=(2*dx)*(yy[0]/2+yy[-1]/2+np.sum(yy[1:-1]))
area2=dx*(y[0]/2+y[-1]/2+np.sum(y[1:-1]))
guess=4/3*area2-1/3*area1
print(guess-(np.cos(x[0])-np.cos(x[-1])))
