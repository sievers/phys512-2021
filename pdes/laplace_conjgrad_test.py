import numpy as np
from matplotlib import pyplot as plt

#we need to find the average of our neighbors, so
#write a routine that finds the sum (which we sometimes
#also need).  You can divide by 4 if you want the average.
def sum_neighbors(mat):
    tot=0
    for i in range(len(mat.shape)):
        tot=tot+np.roll(mat,1,axis=i)
        tot=tot+np.roll(mat,-1,axis=i)
    return tot

#del^2 operator is ourselves minus average
#of neighbors, so write a routine that does that as well
#note that we only solve for the potential away from the
#boundary conditions.  Since we are going to put any fixed
#boundary potentials on the right hand side of the equation,
#we do *not* want to include them here, which is giving us
#the left-hand side operator.  Rather than think carefully
#about which cells are used, we zero out the potential on 
#the boundary.  To avoid screwing up our input matrix, we 
#copy the potential into tmp and then zero the boundary there.
def apply_laplace(mat,mask):
    tmp=mat.copy()
    tmp[mask]=0
    #since the boundary is now masked, summing over the non-boundary
    #neighbors is now the same as summing over all neighbors.
    tot=sum_neighbors(tmp)
    tot[mask]=0
    return tmp-0.25*tot

#We're solving in the interior where rho=0, and V is defined
#on the boundaries.  If we know the potential in one or more neighboring cells,
#we'll need to include that info in setting up our equation.
#assume we're working with the cell V(x,y), and the V(x+1,y) 
#is given as part of our boundary conditions.  
#we re-write V(x,y)-0.25(V(x-1,y)+V(x+1,y)+V(x,y-1)+V(x,y+1))=0
#(which is true anywhere there is no charge)
#Since V(x+1,y) is fixed as part of the boundary conditions
#we put it on the right hand side.  This leaves us with
#V(x,y)-0.25(V(x-1,y)+V(x,y-1)+V(x,y+1))=0.25V(x+1,y)
#following further, the right hand side in Ax=b is the average of 
#neighbors where *only* the boundary conditions are included.  
#to make sure this is the case, we zero out the potential everywhere
#outside of the mask before averaging
def get_rhs(mat,mask):
    tmp=0*mat
    tmp[mask]=mat[mask]
    rhs=0.25*sum_neighbors(mat)
    rhs[mask]=0
    return rhs

#this is vanilla standard conjugate-gradient
def conjgrad(x,b,mask,niter=20,fun=apply_laplace,print_iter=20):
    #r=A@x-b
    r=b-fun(x,mask)
    p=r
    rr=np.sum(r*r)
    for iter in range(niter):
        Ap=fun(p,mask)
        pAp=np.sum(p*Ap)
        alpha=rr/pAp
        x=x+alpha*p
        r=r-alpha*Ap
        rr_new=np.sum(r*r)
        beta=rr_new/rr
        p=r+beta*p
        rr=rr_new
        if iter%print_iter==0:
            print('residual squared is ',rr)
            plt.clf()
            plt.imshow(x)
            plt.show()
            plt.pause(0.001)
    return x


plt.ion()

n=256
bc=np.zeros([n,n])
bc[n//4,n//4:3*n//4]=1
bc[3*n//4,n//4:3*n//4]=-1
mask=np.zeros([n,n],dtype='bool')
mask[:,0]=True
mask[:,-1]=True
mask[-1,:]=True
mask[0,:]=True
mask[n//4,n//4:3*n//4]=True
mask[3*n//4,n//4:3*n//4]=True

b=get_rhs(bc,mask)

V_raw=conjgrad(0*b,b,mask,niter=n*3)
V=V_raw.copy()
V[mask]=bc[mask]
rho=V-0.25*sum_neighbors(V)
rho_masked=rho.copy()
rho_masked[mask]=0
Q=np.sum(np.abs(rho))
Q_mask=np.sum(np.abs(rho_masked))
print('Charge fraction in region is ',Q_mask/Q)
