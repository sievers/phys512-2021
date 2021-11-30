import numpy as np
import numba as nb


@nb.njit(parallel=True)
def add(n):
    tot=1
    val=-1
    for i in nb.prange(n):
        tot=tot+i
    return tot
for i in range(10):
    print('sum is ',add(16*64*64))
