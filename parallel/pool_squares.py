import numpy as np
from multiprocessing import Pool
import os
import time

def square(x):
    pid=os.getpid()
    print('process ',pid,' is working on ',x)
    #time.sleep(3)
    return x*x

if __name__=='__main__':
    nthread=4
    n=2*nthread
    with Pool(nthread) as p:
        vec=p.map(square,range(n+1))
    print('squares are ',vec)
    print('sum of first ',n,' squares is: ',np.sum(vec))
    print('expected: ',n*(n+1)*(2*n+1)//6)
