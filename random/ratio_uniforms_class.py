import numpy as np
from scipy.special import erf #to check our answers
n=100000
u=np.random.rand(n)
v=(2*np.random.rand(n)-1)*.8

r=v/u
#the below was my buggy line - note 0.*r**2 vs. 0.5*r**2
#accept=u<np.sqrt(np.exp(-0.*r**2))

#This line is what I was going for.  It is
#correct but unnecessarily slow
#accept=u<np.sqrt(np.exp(-0.5*r**2))

#might as well use the fast version:
accept=u<np.exp(-0.25*r**2)
t=r[accept]

#check the output
print('mean and std are ',t.mean(),np.std(t))
print('2 and 3-sigma fractions are ',np.mean(np.abs(t)<2),np.mean(np.abs(t)<3))
print('expected are ',erf(2/np.sqrt(2)),erf(3/np.sqrt(2)))
