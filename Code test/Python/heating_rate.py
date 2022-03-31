

import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

params={'axes.titlesize' : 12, 'axes.labelsize' : 12, 'lines.linewidth' : 1.8,
        'lines.markersize' : 2, 'xtick.labelsize' : 12, 'ytick.labelsize' : 12,
        'font.size': 10,'legend.fontsize': 12, 'legend.handlelength' : 1.5,
        'legend.borderpad' : 0.1,'legend.labelspacing' : 0.1, 'axes.linewidth' : 1,
        'figure.autolayout': True}
plt.rcParams.update(params)

A = np.logspace(-1,1,100)

K = 1./(2*A*np.sqrt(np.pi))*(-3+(A+3)*np.arctan(np.sqrt(A))/np.sqrt(A))


plt.figure()
plt.loglog(A,1./K,color="r")
plt.xlabel(r"$A$")
plt.ylabel(r"$\kappa$")



