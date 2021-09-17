import numpy as np
def get_legendre_weights(n):
    #y=Pc - we want to pick c so that Pc goes through y
    #c = P^-1 y (if P is invertible - which it is!)
    #because we oonly care about c_0, then we only need the first
    #fow of P^-1
    x=np.linspace(-1,1,n+1)
    P=np.polynomial.legendre.legvander(x,n)
    Pinv=np.linalg.inv(P)
    coeffs=Pinv[0,:]
    #coeffs=coeffs/coeffs.sum()*n
    return coeffs*n

coeffs=get_legendre_weights(5)

x=np.linspace(0,1,len(coeffs))
y=np.exp(x)
dx=x[1]-x[0]
my_int=np.sum(coeffs*y)*dx
print(my_int-(np.exp(x[-1])-np.exp(x[0])))


ord=10
x=np.linspace(-100,100,ord+1)
y=1/(1+x**2)
dx=x[1]-x[0]
coeffs=get_legendre_weights(ord)
my_int=np.sum(coeffs*y)*dx
pred=np.arctan(x[-1])-np.arctan(x[0])
print(my_int,pred)
