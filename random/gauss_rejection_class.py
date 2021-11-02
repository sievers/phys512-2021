import numpy as np

def lorentzians(n):
    q=np.pi*(np.random.rand(n)-0.5)
    return np.tan(q)

n=10000000
t=lorentzians(n)
y=1.5/(1+t**2)*np.random.rand(n)*2


bins=np.linspace(-10,10,501)
aa,bb=np.histogram(t,bins)
aa=aa/aa.sum()

cents=0.5*(bins[1:]+bins[:-1])
pred=1/(1+cents**2)
pred=pred/pred.sum()
#plt.clf()
#plt.plot(cents,aa,'*')
#plt.plot(cents,pred,'r')
#plt.show()

mygauss=np.exp(-0.5*cents**2)
mylor=1/(1+cents**2)*3
#plt.clf()
#plt.plot(t,y,'.')
#plt.plot(cents,mylor,'b')
#plt.plot(cents,mygauss,'r')
#plt.show()
#plt.xlim(-10,10)
accept=y<np.exp(-0.5*t**2)
t_use=t[accept]

aa,bb=np.histogram(t_use,bins)
aa=aa/aa.sum()
pred=np.exp(-0.5*cents**2)
pred=pred/pred.sum()
plt.clf()
plt.plot(cents,aa,'*')
plt.plot(cents,pred,'r')
plt.show()
