import numpy as np

x=np.linspace(0.5,1,1001)

ord=40
A=np.zeros([len(x),ord+1])
for i in range(ord+1):
    A[:,i]=x**i

u,s,v=np.linalg.svd(A,0)
print(s.min(),s.max())


#A=np.polynomial.legendre.legvander(x,ord)
A=np.polynomial.chebyshev.chebvander(x,ord)
u,s,v=np.linalg.svd(A,0)
print('cheb: ',s.min(),s.max())
