import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
plt.ion()


def make_window(n):
    x=np.linspace(-np.pi,np.pi,n)
    return 0.5+0.5*np.cos(x)

def make_flat_window(n,m):
    tmp=make_window(m)
    win=np.ones(n)
    mm=m//2
    win[:mm]=tmp[:mm]
    win[-mm:]=tmp[-mm:]
    return win

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc



#fnames=glob.glob("[HL]-*.hdf5")
#fname=fnames[0]
fname='H-H1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
strain,dt,utc=read_file(fname)

#th,tl=read_template('GW150914_4_template.hdf5')
template_name='GW150914_4_template.hdf5'
th,tl=read_template(template_name)

n=len(strain)
#win=make_window(n)
win=make_flat_window(n,n//5)
sft=np.fft.rfft(win*strain)

Nft=np.abs(sft)**2
for i in range(10):
    Nft=(Nft+np.roll(Nft,1)+np.roll(Nft,-1))/3

sft_white=sft/np.sqrt(Nft)
tft_white=np.fft.rfft(th*win)/np.sqrt(Nft)
t_white=np.fft.irfft(tft_white)

xcorr1=np.fft.irfft(sft*np.fft.rfft(th*win))
xcorr2=np.fft.irfft(sft_white*np.conj(tft_white))
