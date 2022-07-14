
# import some libraries

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.interpolate import griddata


# read in data

f0 = open('convolve_raw_0.dat', 'r')
f1 = open('convolve_raw_1.dat', 'r')

data0 = np.loadtxt(f0, skiprows=0, delimiter=None)
data1 = np.loadtxt(f1, skiprows=0, delimiter=None)

f0.close()
f1.close()


# examine raw data

# print min/max

print ('x0 = ', min(data0[:,0]), max(data0[:,0]))
print ('y0 = ', min(data0[:,1]), max(data0[:,1]))
print ('d0 = ', min(data0[:,2]), max(data0[:,2]))

print ('x1 = ', min(data1[:,0]), max(data1[:,0]))
print ('y1 = ', min(data1[:,1]), max(data1[:,1]))
print ('d1 = ', min(data1[:,2]), max(data1[:,2]))


# plot raw data

fig = plt.figure(figsize=(6, 10))

rows = 5
rows = 4
columns = 2

grid = plt.GridSpec(rows, columns, wspace = 0.30, hspace = 0.30, \
                    width_ratios=[1,1])

plt.subplot(grid[0,0])
plt.scatter(data0[:,0], data0[:,1]*1.0e3, c=np.log(data0[:,2]), s=1, cmap='jet')
plt.colorbar()

plt.subplot(grid[0,1])
plt.scatter(data1[:,0], data1[:,1]*1.0e3, c=np.log(data1[:,2]), s=1, cmap='jet')
plt.colorbar()


plt.subplot(grid[1,0])
plt.tripcolor(data0[:,0], data0[:,1]*1.0e3, np.log(data0[:,2]), cmap='jet')
plt.colorbar()


plt.subplot(grid[1,1])
plt.tripcolor(data1[:,0], data1[:,1]*1.0e3, np.log(data1[:,2]), cmap='jet')
plt.colorbar()


# interpolate to regular 2d grid

numcols, numrows = 250, 250

x0 = np.linspace(min(data0[:,0]), max(data0[:,0]), numcols)
y0 = np.linspace(min(data0[:,1]), max(data0[:,1]), numrows)
x0, y0 = np.meshgrid(x0, y0)
z0 = griddata((data0[:,0],data0[:,1]), np.log(data0[:,2]), 
              (x0,y0), method='linear', rescale=True)
z0 = np.exp(z0)

x1 = np.linspace(min(data1[:,0]), max(data1[:,0]), numcols)
y1 = np.linspace(min(data1[:,1]), max(data1[:,1]), numrows)
x1, y1 = np.meshgrid(x1, y1)
z1 = griddata((data1[:,0],data1[:,1]), np.log(data1[:,2]), 
              (x1,y1), method='linear', rescale=True)
z1 = np.exp(z1)


#print ('z0 = ', min(z0), max(z0))
#print ('z1 = ', min(z1), max(z1))


# plot interpolated data

plt.subplot(grid[2,0])
#plt.contourf(x0, y0*1.0e3, np.log(z0), 500, cmap='jet', zorder = 0)
plt.contourf(x0, y0*1.0e3, np.log(z0), 500, cmap='jet')#, vmin=-15, vmax=15)#, zorder = 0)
#plt.contourf(x0, y0*1.0e3, np.log(z0), cmap='jet', vmin=-10, vmax=10)
plt.colorbar()

plt.subplot(grid[2,1])
#plt.contourf(x1, y1*1.0e3, np.log(z1), 500, cmap='jet', zorder = 0)
plt.contourf(x1, y1*1.0e3, np.log(z1), 500, cmap='jet')#, vmin=-15, vmax=15)#, zorder = 0)
#plt.contourf(x1, y1*1.0e3, np.log(z1), cmap='jet', vmin=-10, vmax=10)
plt.colorbar()


# perform convolution


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
# the Scharr operator

scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + i*Gy


grad0 = signal.convolve2d(np.log(z0), scharr, boundary='symm', mode='same')
grad1 = signal.convolve2d(np.log(z1), scharr, boundary='symm', mode='same')


plt.subplot(grid[3,0])
plt.imshow(np.absolute(grad0[::-1,:]), cmap='jet')
plt.colorbar()

plt.subplot(grid[3,1])
plt.imshow(np.absolute(grad1[::-1,:]), cmap='jet')
plt.colorbar()


#plt.subplot(grid[4,0])
#plt.imshow(np.angle(grad0[::-1,:]), cmap='hsv') # hsv is cyclic, like angles
#plt.colorbar()

#plt.subplot(grid[4,1])
#plt.imshow(np.angle(grad1[::-1,:]), cmap='hsv') # hsv is cyclic, like angles
#plt.colorbar()


plt.savefig("../figs/convolution.png", bbox_inches='tight')


plt.show()



