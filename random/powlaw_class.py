import numpy as np

alpha=1.5
n=10000000
q=np.random.rand(n)
t=(q)**(1/(1-alpha))

bins=np.linspace(1,50,501)
aa,bb=np.histogram(t,bins)
aa=aa/aa.sum()

cents=0.5*(bins[1:]+bins[:-1])
pred=cents**(-alpha)
pred=pred/pred.sum()
plt.clf()
plt.plot(cents,aa,'*')
plt.plot(cents,pred,'r')
plt.show()
