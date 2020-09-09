# -*- coding: utf-8 -*-
"""
@author: Yanina Corsaro
"""

import numpy as np
import matplotlib.pyplot as plt
from sen import mi_funcion_sen
from scipy.fftpack import fft, fftfreq

def twiddle_factor(k, n, N):
    Wk = np.cos(2*np.pi*k*n/N ) + np.sin(2*np.pi*k*n/N)*1j
    return Wk

N  = 100 # muestras
fs = 100 # Hz
a0 = 1       # Volts
p0 = np.pi/4 # radianes
f0 = 10    # Hz
dt = 1/(fs)

tt, y= mi_funcion_sen( vmax=a0 , dc=0 , ff=f0 , ph=p0, nn=N , fs=fs )
frq = np.linspace( start= 0, stop= fs ,num = N)
#frq1 = np.fft.fftfreq(N, dt)
'''
fft = np.fft.fft(y, N)
fftabs=np.abs(fft)
fftangle=np.angle(fft)
'''
fft = fft(y)
fftabs=np.abs(fft)
fftangle=np.angle(fft)

dft = []
for k in range(0,N):
    res = 0
    for n in range(0,N):
        res = res + twiddle_factor(k, n, N)*y[n]
    dft.append(res) 
dft = np.array(dft)
  
xfabs=np.abs(dft)
xfangle=np.angle(dft)

# grafico módulo de la dft
fig, (ax, ax1) = plt.subplots(1, 2)
ax.plot(frq, xfabs, "-", frq, fftabs, ":")
ax.set(xlabel='frecuencia', ylabel='módulo',title='DFT módulo')
ax.grid()
plt.show()

ax1.plot(frq, xfangle, "-", frq, fftangle, ":")
ax1.set(xlabel='frecuencia', ylabel='fase',title='DFT fase')
ax1.grid()
plt.show()








