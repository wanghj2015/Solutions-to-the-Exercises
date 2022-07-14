

# plot for 1d simulations


import numpy as np
import matplotlib.pyplot as plt


# read in data

d1 = open('../src/density.dat.1', 'rb')
d2 = open('../src/density.dat.2', 'rb')
#d1 = open('../src/density.dat.3', 'rb')
#d2 = open('../src/density.dat.4', 'rb')

v1 = open('../src/velocity.dat.1', 'rb')
v2 = open('../src/velocity.dat.2', 'rb')
#v1 = open('../src/velocity.dat.3', 'rb')
#v2 = open('../src/velocity.dat.4', 'rb')

rho1 = np.fromfile(d1, dtype=np.float64)
rho2 = np.fromfile(d2, dtype=np.float64)

vel1 = np.fromfile(v1, dtype=np.float64)
vel2 = np.fromfile(v2, dtype=np.float64)

d1.close()
d2.close()

v1.close()
v2.close()


n1 = len(rho1)
n2 = len(rho2)


nframe1 = n1//1000
nframe2 = n2//1000


print ('nframe1 = ', nframe1)
print ('nframe2 = ', nframe2)


nframe = min(nframe1, nframe2)


rows = 6
columns = 1


for i in range(nframe):

    fig = plt.figure(figsize=(8, 8))

    grid = plt.GridSpec(rows, columns, wspace = 0.20, hspace = 0.00)

    plt.subplot(grid[0,0])
    plt.plot(range(0, 1000), rho1[i*1000:(i+1)*1000]) 
    plt.ylabel("density 1")

    plt.subplot(grid[1,0])
    plt.plot(range(0, 1000), rho2[i*1000:(i+1)*1000]) 
    plt.ylabel("density 2")

    plt.subplot(grid[2,0])
    plt.plot(range(0, 1000), rho1[i*1000:(i+1)*1000] - rho2[i*1000:(i+1)*1000]) 
    plt.ylabel("density diff")


    plt.subplot(grid[3,0])
    plt.plot(range(0, 1000), vel1[i*1000:(i+1)*1000])
    plt.ylabel("velocity 1")

    plt.subplot(grid[4,0])
    plt.plot(range(0, 1000), vel2[i*1000:(i+1)*1000])
    plt.ylabel("velocity 2")

    plt.subplot(grid[5,0])
    plt.plot(range(0, 1000), vel1[i*1000:(i+1)*1000] - vel2[i*1000:(i+1)*1000])
    plt.ylabel("velocity diff")


    #plt.xlim(0, 1000)
    #plt.ylim(-1, 30)

    plt.xlabel("meters")


    plt.savefig("../movie/case1_{0:03d}.png".format(i))
    #plt.savefig("../movie/case2_{0:03d}.png".format(i))


    #plt.show()

    plt.clf()
    plt.close()




