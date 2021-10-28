import copy as cp

class Complex:
    def __init__(self,real,im):
        self.r=real
        self.i=im
    def copy(self):
        #return Complex(self.r,self.i)
        #return copy.deepcopy(self)
        return cp.copy(self)

    def abs(self):
        return (self.r**2+self.i**2)**0.5
    def conj(self):
        return Complex(self.r,-self.i)
    def __add__(self,a):
        try:
            if isinstance(a,Complex):
                return Complex(self.r+a.r,self.i+a.i)
            else:
                return Complex(self.r+a,self.i)
        except:
            print('We dont know how to add whatever it is you are adding')
    def __radd__(self,a):
        return self+a
    def __repr__(self):
        if self.i>0:
            return repr(self.r)+'+'+repr(self.i)+'i'
        else:
            return repr(self.r)+'-'+repr(-self.i)+'i'
    def __mul__(self,a):
        if isinstance(a,Complex):
            return Complex(self.r*a.r-self.i*a.i,self.r*a.i+self.i*a.r)
        else:
            return Complex(self.r*a,self.i*a)
    def __rmul__(self,a):
        return self*a
    def double(self):
        self.r=2*self.r
        self.i=2*self.i
        self.isdouble=True

a=Complex(2,3)
b=a.conj()
print('absolute value is ',a.abs(),b.abs())
c=a.__add__(3)
d=3+a
