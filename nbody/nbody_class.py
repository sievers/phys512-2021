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
        pot=0
        for i in range(self.n):
            dx=self.xy[i,0]-self.xy[:,0]
            dy=self.xy[i,1]-self.xy[:,1]
            rinv=(dx**2+dy**2+self.soft**2)**-0.5
            r3=rinv*rinv*rinv 
            fx=(dx*r3).sum()
            fy=(dy*r3).sum()
            self.f[i,0]=-fx
            self.f[i,1]=-fy
            pot=pot+np.sum(rinv)
        return -pot/2

    def update(self,dt):
        self.xy=self.xy+self.v*dt
        self.v=self.v+self.f*dt
        pot=self.get_forces()
        kin=0.5*np.sum(self.v**2)
        print('energies are ',pot,kin,kin+pot)
    def update_leapfrog(self,dt):
        self.xy=self.xy+self.v*dt
        kin1=0.5*np.sum(self.v**2) #kinetic at start
        pot=self.get_forces()
        self.v=self.v+self.f*dt
        kin2=0.5*np.sum(self.v**2) #kinetic at end
        kin=(kin1+kin2)/2
        print('energies are ',kin,pot,pot+kin)
    def update_o2(self,dt):
        xy=self.xy.copy()
        v=self.v.copy()
        pot=self.get_forces()
        kin=0.5*np.sum(v**2)
        print('energy is ',pot,kin,kin+pot)
        xy_new=self.xy+self.v*dt
        v_new=self.v+self.f*dt
        v_ave=(v+v_new)/2
        x_ave=(xy+xy_new)/2
        self.x=x_ave
        pot=self.get_forces()
        self.v=v+dt*self.f
    def update_rk4(self,dt):
        v=self.v.copy()
        x=self.xy.copy()

        kin=np.sum(v**2)
        pot=self.get_forces()
        h1x=self.v*dt
        h1v=self.f*dt

        self.xy=x+h1x/2
        self.v=v+h1v/2
        self.get_forces()
        h2x=self.v*dt
        h2v=self.f*dt

        self.xy=x+h2x/2
        self.v=v+h2v/2
        self.get_forces()
        h3x=self.v*dt
        h3v=self.f*dt

        self.xy=x+h3x
        self.v=v+h3v
        self.get_forces()
        h4x=self.v*dt
        h4v=self.f*dt

        self.xy=x+(h1x+h2x*2+h3x*2+h4x)/6
        self.v=v+(h1v+h2v*2+h3v*2+h4v)/6
        print(kin,pot,kin+pot)
        return kin,pot

parts=Particles(10000)
parts.ic_gauss()
parts.get_forces()
osamp=5
dt=parts.soft**1.5
dt2=parts.soft/np.sqrt(parts.n)
print('two timesteps are ',dt,dt2)

dt=np.min([dt,dt2])
print(dt)



for i in range(1000):
    plt.clf()
    plt.plot(parts.xy[:,0],parts.xy[:,1],'.')
    for j in range(osamp):
        parts.update_leapfrog(dt/osamp)
    plt.pause(0.001)
