#!/usr/bin/env python
#_*_ encoding:utf-8 _*_

import os
import numpy as np
import matplotlib as mpl
mpl.use('agg')
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import scipy.fftpack as fp
import warnings
warnings.filterwarnings("ignore")

#-------------------------------------------------------------------#

#基本参数
potim = 1.0 
nsw = 2000
nkpts = 64
nbands = 234
kpoint = [1,1,1,1,1,33,33,33]
band = [186,188,192,193,194,188,192,193]  
path_D = './'   
#-------------------------------------------------------------------#

Engy = np.loadtxt(path_D+'energy.dat').reshape(-1,nkpts,nbands)[:nsw]

tmin = 0
tmax = tmin + nsw * potim


#-------------------------------------------------------------------#

fig = plt.figure()
fig.set_size_inches(4.8,3.6) 
ax = plt.subplot()

omega = 33.3564 * 1E3 * fp.fftfreq(nsw,potim)
np.savetxt('omega.dat',omega)
for ii in range(len(band)):
    k = kpoint[ii]
    b = band[ii]
    en = Engy[:,k-1,b-1]-np.average(Engy[:,k-1,b-1])
    en =  np.correlate(en, en, 'full')[en.size-1:] /en.size
    fft = fp.fft(en/en[0])
    FFT = np.abs(fft)
    ax.plot(omega[:300],FFT[:300])


ax.legend(ax.get_lines(),[r'VBM@$\Gamma$',r'VBM@Z',r'VBM@X',r'CBM@$\mathrm{\mathsf{\Gamma}}$',r'CBM@Z',r'VBM@Z/2',r'VBM@P',r'CBM@Z/2'], 
          loc='upper right', fontsize=8, ncol=2, bbox_to_anchor=(0.95,0.95))

ax.set_xlim(0,1000)
ax.set_ylim(0,400)
ax.set_ybound(lower=0)
ax.set_xlabel('Wavenumber [cm$^-$$^1$]',fontsize=16,labelpad=5)
ax.set_ylabel('Amplitude [arb. unit]',fontsize=16,labelpad=5)


plt.tight_layout(pad=0.4)
plt.savefig('fft.png',dpi=400)
