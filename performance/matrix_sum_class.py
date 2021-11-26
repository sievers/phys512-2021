import numpy as np
import numba as nb
import time

@nb.njit
def sum_matrix(mat):
    n=mat.shape[0]
    m=mat.shape[1]
    tot=0.0
    for i in range(n):
        for j in range(m):
            tot=tot+mat[i,j]
    return tot

n=10000
mat=np.ones([n,n])
tot=sum_matrix(mat)
t1=time.time()
tot=sum_matrix(mat)
t2=time.time()
print(t2-t1)
