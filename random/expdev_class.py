import numpy as np
from matplotlib import pyplot as plt


alpha=1

n=10000000
q=np.random.rand(n)
t=np.log(q)/(-alpha)

bins=np.linspace(0,20,101)
aa,bb=np.histogram(t,bins)

cents=(bb[1:]+bb[:-1])/2
pred=np.exp(-alpha*cents)
aa=aa/aa.sum()
pred=pred/pred.sum()
plt.ion()
plt.clf()
plt.plot(cents,aa,'*')
plt.plot(cents,pred,'r')
plt.show()
