#!/usr/bin/env python
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

a = np.loadtxt('band.dat').reshape(7,7,9)
b = np.sum(a,axis=(0,1))/np.sum(a)
b = b[[0,1,3,4]]

labels = ['c1v1-c1v1', 'c1v1-c1v2', 'c1v2-c1v1', 'c1v2-c1v2']

fig, ax = plt.subplots()

ax.bar(labels, b)

ax.set_xlabel('Basis')
ax.set_ylabel('Populaton')

plt.savefig('band.png')
