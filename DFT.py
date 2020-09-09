# -*- coding: utf-8 -*-
"""
@author: Yanina Corsaro
"""

import numpy as np
import matplotlib.pyplot as plt
from sen import mi_funcion_sen

def twiddle_factor(k, n, N):
    Wk = np.cos(2*np.pi*k*n/N ) + np.sin(2*np.pi*k*n/N)*1j
    return Wk

N  = 100 # muestras
fs = 100 # Hz
a0 = 1       # Volts
p0 = 0# radianes
f0 = 10    # Hz
dt = 1/(fs)

tt, y= mi_funcion_sen( vmax=a0 , dc=0 , ff=f0 , ph=p0, nn=N , fs=fs )
frq = np.linspace( start= 0, stop= fs ,num = N)

dft = []
for k in range(0,N):
    res = 0
    for n in range(0,N):
        res = res + twiddle_factor(k, n, N)*y[n]
    dft.append(res) 
  
xfabs=np.abs(dft)
xfangle=np.angle(dft)

for i in range(0,N):
    # aca chequeamos que el módulo no sea muy cercano a cero
    if xfabs[i]<0.1:
        xfangle[i]=0

# grafico señal
fig3, ax3 = plt.subplots()
ax3.plot(tt, y)
ax3.set(xlabel='tiempo (segundos)', ylabel='amplitud (V)',title='señal: senoidal')
ax3.grid()
plt.show()

# grafico módulo de la dft
fig, ax = plt.subplots()
ax.plot(frq, xfabs)
ax.set(xlabel='frecuencia', ylabel='módulo',title='DFT módulo')
ax.grid()
plt.show()

# grafico fase de la dft
fig2, ax2 = plt.subplots()
ax2.plot(frq, xfangle)
ax2.set(xlabel='frecuencia', ylabel='fase',title='DFT fase')
ax2.grid()
plt.show()





