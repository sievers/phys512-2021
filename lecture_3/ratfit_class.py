import numpy as np

def rateval(x,p,q):
    top=0
    for i,par in enumerate(p):
        top=top+par*x**i
    bot=1
    for i,par in enumerate(q):
        bot=bot+par*x**(i+1)
    return top/bot

def ratfit(y,x,n,m):
    npt=len(x)
    assert(len(y)==npt)
    assert(n>=0)
    assert(m>=0)
    assert(n+1+m==npt)

    top_mat=np.empty([npt,n+1])
    bot_mat=np.empty([npt,m])
    for i in range(n+1):
        top_mat[:,i]=x**i
    for i in range(m):
        bot_mat[:,i]=y*x**(i+1)
    mat=np.hstack([top_mat,-bot_mat])
    print(mat)
    pars=np.linalg.inv(mat)@y
    p=pars[:n+1]
    q=pars[n+1:]
    return mat,p,q
    #return mat


x=np.arange(-2,3)
y=1/(1+x**2)
m=len(y)//2
n=len(y)-m-1

mat,p,q=ratfit(y,x,n,m)
#mat=ratfit(y,x,n,m)
assert(1==0)


x=np.linspace(-3,3)
#what would p,q be for y=1/1+x**2?
p=[1,2]
q=[0,1,-2]  #bottom = 1 + 0x + 1x^2, but we don't count the first 1

y=rateval(x,p,q)

