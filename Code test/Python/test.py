import numpy as np
import matplotlib.pyplot as plt
import plasmapy as pp
plt.close("all")



x = np.logspace(-2, 2, 100)
F = pp.dispersion.dispersionfunction.plasma_dispersion_func(x)

rF = np.real(F)
iF = np.imag(F)

approx=-1./x
approx2 = -1./x -1./(2*x**3) +1j*np.pi*np.exp(-x**2)

approx_s = -np.sqrt(2)*x +1j*np.sqrt(np.pi)



plt.figure()
plt.ylim(-2,2)

plt.axhline(1,color="grey",linestyle="--",linewidth=0.9)
# plt.semilogx(x,rF,color="r")
# plt.semilogx(x,iF,color="b")

# plt.semilogx(x,approx,color="k")

plt.semilogx(x,np.real(approx2)/rF,color="green")
plt.semilogx(x,np.imag(approx2)/iF,color="k")

plt.semilogx(x,np.real(approx_s)/rF,color="r")
plt.semilogx(x,np.imag(approx_s)/iF,color="b")

