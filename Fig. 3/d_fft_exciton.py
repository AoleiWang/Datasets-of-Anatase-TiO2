#!/usr/bin/env python
#_*_ encoding:utf-8 _*_

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.fftpack as fp
import warnings
warnings.filterwarnings("ignore")

#-------------------------------------------------------------------#

potim = 1.0 
nsw = 1000
label = ['Bright exciton','Dark exciton']
a = np.zeros((nsw,2))
a[:,0] = np.loadtxt('en_bright.dat')[:,1]
a[:,1] = np.loadtxt('en_dark.dat')

#-------------------------------------------------------------------#
tmin = 0
tmax = tmin + nsw * potim
#-------------------------------------------------------------------#

fig, (ax) = plt.subplots(1)
fig.set_size_inches(4.8,3.6)

omega = 33.3564 * 1E3 * fp.fftfreq(nsw,potim)
for ii in range(2):
    en = a[:,ii]-np.average(a[:,ii])
    en =  np.correlate(en, en, 'full')[en.size:] /en.size
    fft = fp.fft(en/en[0])
    FFT = np.abs(fft)
    ax.plot(omega[:300],FFT[:300])

ax.set_xlim(0,1000)
ax.set_ybound(lower=0)
ax.set_xlabel('Wavenumber [cm$^-$$^1$]',fontsize=16,labelpad=5)
ax.set_ylabel('Amplitude [arb. unit]',fontsize=16,labelpad=5)
ax.legend(label)


plt.tight_layout(pad=0.4)
plt.subplots_adjust(wspace=0.35)
plt.savefig('fft.png', dpi=400)
