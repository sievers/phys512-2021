import numpy as np
import numba as numbat



f=open('input_dec_20.txt')
lines=f.readlines()
f.close()

mydim=10
dx=mydim+2

id=1
block=np.loadtxt('input_dec_20_formatted.txt',skiprows=dx*id+1,max_rows=mydim)
nblock=len(lines)//dx
blocks=[None]*nblock
ids=np.zeros(nblock,dtype='int')
edges=np.zeros([nblock,4],dtype='int')
edges2=np.zeros([nblock,4],dtype='int')
vec1=2**np.arange(mydim)
vec2=2**np.arange(mydim-1,-1,-1)
for id in range(nblock):
    block=np.loadtxt('input_dec_20_formatted.txt',skiprows=dx*id+1,max_rows=mydim)
    blocks[id]=block
    ll=lines[id*dx]
    tags=ll.split()
    ids[id]=np.int(tags[1][:-1])
    edges[id,0]=np.dot(vec1,block[:,0])
    edges[id,1]=np.dot(vec1,block[:,-1])
    edges[id,2]=np.dot(vec1,block[0,:])
    edges[id,3]=np.dot(vec1,block[-1,:])
    
    edges2[id,0]=np.dot(vec2,block[:,0])
    edges2[id,1]=np.dot(vec2,block[:,-1])
    edges2[id,2]=np.dot(vec2,block[0,:])
    edges2[id,3]=np.dot(vec2,block[-1,:])


isedge=np.zeros(edges.shape,dtype='bool')
for i in range(nblock):
    for j in range(4):
        if np.sum(edges==edges[i,j])==1:
            if np.sum(edges2==edges[i,j])==0:
                isedge[i,j]=True


iscorner=np.sum(isedge,axis=1)==2
print(ids[iscorner],np.product(ids[iscorner]))
