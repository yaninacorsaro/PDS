import matplotlib.pyplot as plt
import numpy as np


def mi_funcion_sen( vmax , dc , ff , ph, nn , fs ):
     
    tt = np.linspace( start= 0, stop= nn/fs ,num = nn)
    xx = vmax*np.sin(tt*2*np.pi*ff+ph) + dc
    return tt, xx

"""
# test
tt, xx = mi_funcion_sen( vmax = 1 , dc = 0 , ff = 1 , ph = 0, nn = 1000 , fs=100 )
fig, ax = plt.subplots()
ax.plot(tt, xx)
ax.set(xlabel='time (s)', ylabel='voltage (V)',title='sen function')
ax.grid()
plt.show()
"""

