#!/usr/bin/env python

import os
import re
import numpy as np
from optparse import OptionParser

import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
mpl.use('agg')
import matplotlib.pyplot as plt
mpl.rcParams['axes.unicode_minus'] = False

def mesh():
    index_arr = np.arange(27)
    x, y, z = np.meshgrid(range(3),range(3),range(3), indexing='ij')
    arr_3d = np.stack((x,y,z), axis=-1)
    arr_3d = arr_3d.reshape((27, 3))
    result_arr = arr_3d[index_arr] - 1
    
    return result_arr

'''
Use matplotlib to plot band structure
'''
Bands = np.loadtxt('gw_wannier90_band.dat')

pi = 3.1415926 
efermi = 3.53457
nspin = 1
nkpts = 593
nbands = 22

inps  = [line for line in open('POSCAR_primitivecell') if line.strip()]
cell = np.array([line.split() for line in inps[2:5]], dtype=float)
icell = np.linalg.inv(cell).T * 2 * np.pi
b1, b2, b3 = np.linalg.norm(icell, axis=1)  
kpts  = ['Z','G','X','P','N','G']
kcoo  = np.array([[0.5,0.5,-0.5],[0,0,0],[0,0,0.5],[0.25,0.25,0.25],[0,0.5,0],[0,0,0]])
kpath = Bands[:nkpts,0]
bands = Bands[:,1].reshape(nbands,nkpts) - efermi

kpt_bounds = [0.0,0.65334634,1.82305,2.14973,2.97583,3.86612]
xx = np.linalg.norm(np.dot(np.diff(kcoo,axis=0),icell),axis=1)
xx = np.insert(xx, 0, 0)

width = 3.2
height = 3.0
ymin = -1.5
ymax = 5

fig = plt.figure()
fig.set_size_inches(width, height)
ax = plt.subplot(111)

for Ispin in range(nspin):
    for Iband in range(nbands):
        ax.plot(kpath, bands[Iband, :], lw=1.0,
                        alpha=0.8, ls='-',
                        color='black',zorder=1)

#pripp = np.loadtxt('pripp-dark.dat').reshape(64,8,8,8,8)  #dark exciton
pripp = np.loadtxt('pripp-bright.dat').reshape(64,8,8,8,8)   #bright exciton

pric = np.sum(pripp,axis=(3,4)).reshape(512,-1)
priv = np.sum(pripp,axis=(1,2)).reshape(512,-1)

prik = np.loadtxt('coff4dk.txt').reshape(512,3)

ppo=np.zeros((nkpts,nbands),dtype=float)
for ii in range(nkpts):
    kk = 0
    for jj in range(5):
        if kpath[ii] >=np.cumsum(xx)[jj] and kpath[ii] < np.cumsum(xx)[jj+1]:
            kk = jj
            break
    ss = kcoo[kk] + (kpath[ii]-np.cumsum(xx)[kk])/xx[kk+1]*(kcoo[kk+1]-kcoo[kk])
    delt = ss + mesh() 
    for ll in range(512):
        delta = np.linalg.norm(delt - prik[ll],axis=1)
        delta2 = np.linalg.norm(-delt - prik[ll],axis=1)
        if np.min(delta) < 0.00171 or np.min(delta2) < 0.00171:
            ppo[ii,12:14] += pric[ll,:2]
            ppo[ii,7:12] += priv[ll,:-3][::-1]
            continue


fac = 200
sumc = np.sum(ppo[:,12:])
sumv = np.sum(ppo[:,:12])
scal = sumc/sumv

ppo[:100,11] += ppo[:100,10]   #band degeneracy
for Iband in range(7,14):
    if Iband > 11:
        ax.scatter(kpath, bands[Iband, :], c='red',s=ppo[:,Iband]*fac, lw=0.0,
                        alpha=1.0,zorder=3)
    else:
        ax.scatter(kpath, bands[Iband, :], c='blue',s=ppo[:,Iband]*fac*scal, lw=0.0,
                        alpha=1.0,zorder=3)
           
for bd in kpt_bounds[1:-1]:
    ax.axvline(x=bd, ls='-', color='k', lw=0.5, alpha=0.5,zorder=2)

ax.set_ylabel('Energy [eV]', # fontsize='small',
        labelpad=5)
ax.set_ylim(ymin, ymax)
ax.set_xlim(kpath.min(), kpath.max())

ax.set_xticks(kpt_bounds)

kname = [x.upper() for x in kpts]
for ii in range(len(kname)):
    if kname[ii] == 'G':
        kname[ii] = r'$\mathrm{\mathsf{\Gamma}}$'
    else:
        kname[ii] = r'$\mathrm{\mathsf{%s}}$' % kname[ii]
ax.set_xticklabels(kname)

ax.yaxis.set_minor_locator(AutoMinorLocator(2))

plt.tight_layout(pad=0.20)
plt.savefig('band-bright.eps')
#plt.savefig('band-dark.eps')
