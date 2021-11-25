import numpy as np
import time
import ctypes


#pull a library in
mylib=ctypes.cdll.LoadLibrary("libhist2d_c.so")
#tell python which function from the library you want.  You can see what's in there
#on a unix system with the "nm" command.
hist2d_c=mylib.hist2d 
#now tell python what the arguments need to look like.
hist2d_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_long,ctypes.c_long,ctypes.c_long]




def hist_2d(xy,grid):
    ixy=np.asarray(np.round(xy),dtype='int')
    n=xy.shape[0]
    for i in range(n):
        grid[ixy[i,0],ixy[i,1]]=grid[ixy[i,0],ixy[i,1]]+1

npix=1000
npt=1000000
ndim=2
xy=np.random.rand(npt,ndim)*(npix-1)  #-1 is just to make sure we don't go off the edge

grid=np.zeros([npix,npix])
grid2=np.zeros([npix,npix])


#ipix=np.asarray(np.round(xy),dtype='int')
t1=time.time()
hist_2d(xy,grid)
#for i in range(npt):
#    grid[ipix[i,0],ipix[i,1]]=grid[ipix[i,0],ipix[i,1]]+1
t2=time.time()
rate1=(t2-t1)/npt
print('time per particle to project was ' + repr(rate1))


t1=time.time()
ixy=np.asarray(np.round(xy),dtype='int')
hist2d_c(ixy.ctypes.data,grid2.ctypes.data,npt,npix,npix)
t2=time.time()
rate2=(t2-t1)/npt
print('time per particle in C was ' + repr(rate2))

print("mean difference is " + repr(np.mean(np.abs(grid-grid2))))
print('speedup was ' + repr(rate1/rate2))
