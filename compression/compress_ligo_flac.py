import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
plt.ion()
import os

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
fname='../ligo/H-H1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
strain,dt,utc=read_file(fname)


mynoise=np.std(np.diff(strain))
nbit_noise=5
scale_fac=mynoise/(2**nbit_noise)
#noise_level=np.median(np.abs(np.diff(strain)))
#nbit=7 
#scale_fac=noise_level/2**nbit
istrain=np.asarray(strain/scale_fac,dtype='int')

if np.max(np.abs(istrain))<2**15:
    print('we can use 2-byte integers')
    tmp=np.asarray(istrain,dtype='int16')
    f=open('ligo_16bit.raw','w')
    tmp.tofile(f)
    f.close()
    os.system('flac -8 -f --endian=little --bps=16 --sign=signed --channels=1 --sample-rate=44100 ligo_16bit.raw')

#ERROR: for encoding a raw file you must specify a value for --endian, --sign, --channels, --bps, and --sample-rate
    
