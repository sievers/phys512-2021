import numpy as np
import numba as nb
import time

def sum_arrs(a,b,c,d,e):
    tot=np.zeros_like(a)
    tot=tot+a
    tot=tot+b
    tot=tot+c
    tot=tot+d
    tot=tot+e
    return tot

def sum_arrs_oneline(a,b,c,d,e):
    return a+b+c+d+e

@nb.njit
def sum_arrs_numba(a,b,c,d,e):
    tot=np.empty_like(a)
    n=len(a)
    for i in np.arange(n):
        tot[i]=a[i]+b[i]+c[i]+d[i]+e[i]


@nb.njit(parallel=True)
def sum_arrs_numba_par(a,b,c,d,e):
    n=len(a)
    tot=np.empty(n)
    for i in nb.prange(n):
        tot[i]=a[i]+b[i]+c[i]+d[i]+e[i]
    return tot


#@nb.njit(parallel=True)
#def sum_arrs_numba_par(a,b,c,d,e):
#    n=len(a)
#    print('n is ',n)
#    tot=np.empty(n)
#    for i in nb.prange(n):
#        tot[i]=a[i]+b[i]+c[i]+d[i]+e[i]


n=1000000*10
a=np.random.randn(n)
b=np.random.randn(n)
c=np.random.randn(n)
d=np.random.randn(n)
e=np.random.randn(n)

t1=time.time()
tot1=sum_arrs(a,b,c,d,e)
t2=time.time()
print('first sum took ',t2-t1)

for iter in range(10):
    t1=time.time()
    tot2=sum_arrs_oneline(a,b,c,d,e)
    t2=time.time()
    print('second sum took ',t2-t1)

tot3=sum_arrs_numba(a,b,c,d,e)
for iter in range(1000):
    t1=time.time()
    tot3=sum_arrs_numba(a,b,c,d,e)
    t2=time.time()
    print('numba version took ',t2-t1)

tot4=sum_arrs_numba_par(a,b,c,d,e)
for iter in range(10):
    t1=time.time()
    tot4=sum_arrs_numba_par(a,b,c,d,e)
    t2=time.time()
    nbytes=len(a)*8*6
    bw=nbytes/(t2-t1)/1e9
    print('numba parallel version took ',t2-t1,' with bandwidth ',bw)
