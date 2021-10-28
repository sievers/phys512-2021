import numpy as np
from matplotlib import pyplot as plt
plt.ion()

class Particles:
    def __init__(self,n=2,soft=0.01):
        self.xy=np.empty([n,2])
        self.v=np.empty([n,2])
        self.m=np.empty(n)
        self.f=np.empty([n,2])
        self.n=n
        self.soft=soft
    def ic_gauss(self,sig=1.0):
        self.xy[:]=np.random.randn(self.n,2)*sig
        self.v[:]=0
        self.m[:]=1
    def get_forces(self):
        for i in range(self.n):
            dx=self.xy[i,0]-self.xy[:,0]
            dy=self.xy[i,1]-self.xy[:,1]
            r3=(dx**2+dy**2+self.soft**2)**1.5
            fx=(dx/r3).sum()
            fy=(dy/r3).sum()
            self.f[i,0]=-fx
            self.f[i,1]=-fy
    def update(self,dt):
        self.xy=self.xy+self.v*dt
        self.v=self.v+self.f*dt
        self.get_forces()
parts=Particles(1000)
parts.ic_gauss()
parts.get_forces()
dt=0.001
for i in range(1000):
    plt.clf()
    plt.plot(parts.xy[:,0],parts.xy[:,1],'.')
    parts.update(dt)
    plt.pause(0.001)
