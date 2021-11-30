import numpy as np
import numba as nb


@nb.njit(parallel=True)
def assign(n):
    val=-1
    for i in nb.prange(n):
        val=i
    print('after loop, value is ',val)

for i in range(10):
    assign(16*64*64)
