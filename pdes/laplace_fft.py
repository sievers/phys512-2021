import numpy as np
from matplotlib import pyplot as plt
import time

#make a greens function, the potential from a point charge at (0,0)
#we know at r=0, V[0]-V_ave,neighbors=Q[0].  If Q[0]=1, then V[0]=1+V_ave,neighbors
def greens(n,ndim=2):
    dx=np.arange(n)
    dx[n//2:]=dx[n//2:]-n
    if ndim==2:
        pot=np.zeros([n,n])
        xmat,ymat=np.meshgrid(dx,dx)
        dr=np.sqrt(xmat**2+ymat**2)
        dr[0,0]=1 #dial something in so we don't get errors.  we will overwrite later
        pot=np.log(dr)/2/np.pi  #in 2-D, potential looks like log(r) since a 2D point charge is a 3D line charge
        pot=pot-pot[n//2,n//2]  #set it so the potential at the edge goes to zero.  This is arbitrary
        pot[0,0]=pot[0,1]-0.25 #we know the Laplacian in 2D picks up rho/4 at the zero point
        return pot

#convolve the density with the Greens function to get the potential
def rho2pot(rho,kernelft):
    tmp=rho.copy()
    tmp=np.pad(tmp,(0,tmp.shape[0]))

    tmpft=np.fft.rfftn(tmp)
    tmp=np.fft.irfftn(tmpft*kernelft)
    if len(rho.shape)==2:
        tmp=tmp[:rho.shape[0],:rho.shape[1]]
        return tmp
    if len(rho.shape)==3:
        tmp=tmp[:rho.shape[0],:rho.shape[1],:rho.shape[2]]
        return tmp
    print("error in rho2pot - unexpected number of dimensions")
    assert(1==0)

#we are solving for the charge on the boundary that gives the potential 
#on the boundary.  In this case, Ax=b has x as the charge, b as the potential
#on the boundary, and A convolves the charge with the Green's function to get
#the potential from the current charge distribution. Because the DFT can be
#written as a matrix multiply, that makes the convolution a matrix multiply.

def rho2pot_masked(rho,mask,kernelft,return_mat=False):
    rhomat=np.zeros(mask.shape)
    rhomat[mask]=rho
    potmat=rho2pot(rhomat,kernelft)
    #Because we are solving for the charge only on the boundary, in general
    #we only want the potential from the current charge evaluated on the 
    #boundary.  However, at the end we will probably want to see the full
    #potential everywhere in space, setting return_mat=True will give that to you.
    if return_mat:
        return potmat
    else:

        return potmat[mask]


def cg(rhs,x0,mask,kernelft,niter,fun=rho2pot_masked,show_steps=False,step_pause=0.01):
    """cg(rhs,x0,mask,niter) - this runs a conjugate gradient solver to solve Ax=b where A
    is the Laplacian operator interpreted as a matrix, and b is the contribution from the 
    boundary conditions.  Incidentally, one could add charge into the region by adding it
    to b (the right-hand side or rhs variable).  Other than carrying around helper variables
    (like mask and kernelft), this is once again bog-standard conjugate gradient."""

    t1=time.time()
    Ax=fun(x0,mask,kernelft)
    r=rhs-Ax
    #print('sum here is ',np.sum(np.abs(r[mask])))
    p=r.copy()
    x=x0.copy()
    rsqr=np.sum(r*r)
    print('starting rsqr is ',rsqr)
    for k in range(niter):
        #Ap=ax_2d(p,mask)
        Ap=fun(p,mask,kernelft)
        alpha=np.sum(r*r)/np.sum(Ap*p)
        x=x+alpha*p
        if show_steps:            
            tmp=fun(x,mask,kernelft,True)
            plt.clf();
            plt.imshow(tmp,vmin=-2.1,vmax=2.1)
            plt.colorbar()
            plt.title('rsqr='+repr(rsqr)+' on iter '+repr(k+1))
            plt.savefig('laplace_iter_1024_'+repr(k+1)+'.png')
            plt.pause(step_pause)
        r=r-alpha*Ap
        rsqr_new=np.sum(r*r)
        beta=rsqr_new/rsqr
        p=r+beta*p
        rsqr=rsqr_new
        #print('rsqr on iter ',k,' is ',rsqr,np.sum(np.abs(r[mask])))
    t2=time.time()
    print('final rsqr is ',rsqr,' after ',t2-t1,' seconds')
    return x

plt.ion()

#do our usual setting up of boundary conditions with the potential held to zero on the edges
n=1024
bc=np.zeros([n,n])
mask=np.zeros([n,n],dtype='bool')
mask[0,:]=True
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True
bc[0,:]=0.0
bc[0,0]=0.0
bc[0,-1]=0.0
#This adds a bar in the interior held at fixed potential
bc[n//4:3*n//4,(2*n//5)]=2.0
mask[n//4:3*n//4,(2*n//5)]=True

bc[n//4:3*n//4,(3*n//5)]=-2.0
mask[n//4:3*n//4,(3*n//5)]=True

#get our green's function.  Why is the kernel being calculated for 2n?
kernel=greens(2*n,2)
kernelft=np.fft.rfft2(kernel)

#we're solving for the charge that gives us the potential, so 
#our right-hand side is the potential on the mask.  
rhs=bc[mask]
x0=0*rhs

#this should give us the charge on the boundary that matches the potential
rho_out=cg(rhs,x0,mask,kernelft,40,show_steps=True,step_pause=0.25)
#convert the charge on the boundary to the potential everywhere in space
#we should be done at this point!
pot=rho2pot_masked(rho_out,mask,kernelft,True)


