import numpy as np
from mpi4py import MPI

def integrate(lims,fun,dx):
    x1=lims[0]
    x2=lims[1]
    print('integrating ',x1,x2)
    n=int(np.ceil((x2-x1)/dx))
    if (n&1)==0:
        n=n+1
    x=np.linspace(x1,x2,n)
    dx=x[1]-x[0]
    tot=fun(x[0])+fun(x[-1])
    tot=tot+4*np.sum(fun(x[1::2]))
    tot=tot+2*np.sum(fun(x[2:-1:2]))
    return dx*tot/3

comm=MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc=comm.Get_size()
print('I am ',myrank,' out of ',nproc)
x0=0.0
x1=1.0
dx=0.001

#break up the domain and decide which part
#this process owns
x=np.linspace(x0,x1,nproc+1)
myx0=x[myrank]
myx1=x[myrank+1]
myans=integrate([myx0,myx1],np.exp,dx)

#at this point each process knows its piece of 
#the answer.  Now accumulate onto process 0
if myrank==0:
    ans=myans
    for i in range(1,nproc):
        ans=ans+comm.recv(source=i)
    print('on process ',myrank,' integral is ',ans)
else:
    comm.send(myans,0)



