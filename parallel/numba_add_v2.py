import numpy as np
import numba as nb


@nb.njit(parallel=True)
def add(n):
    tot=np.zeros(1)
    val=-1
    for i in nb.prange(n):
        tot[0]=tot[0]+i
    return tot[0]
for i in range(10):
    print('sum is ',add(16*64*64))
