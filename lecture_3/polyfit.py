import numpy as np
from matplotlib import pyplot as plt
npt=8
x=np.linspace(-1,1,npt)
y=np.exp(x)

X=np.empty([npt,npt])
print(X)
for i in range(npt):
    X[:,i]=x**i
Xinv=np.linalg.inv(X)
c=Xinv@y
y_pred=X@c
print('reconstructed error is ',np.std(y_pred-y))

xx=np.linspace(-1,1,npt*100)
XX=np.empty([len(xx),npt])
for i in range(npt):
    XX[:,i]=xx**i
yy=XX@c
ytrue=np.exp(xx)
plt.ion()
plt.plot(xx,yy)
plt.plot(xx,ytrue)
plt.plot(x,y,'*')
plt.show()
