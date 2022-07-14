

# Ray tracing for QP ionosphere model


# import some libraries
import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt


# define the quasi-parabolic (QP) model
def qp_model(Nm, rb, rm, r):
    ym = rm-rb
    rt = rm*rb/(rb-ym)
    if (rb < r and r < rt): 
       Ne = Nm * (1 - ((r-rm)*rb/ym/r)**2)
    else: 
       Ne = 0
    return Ne


# define the ray tracing range equation
def ray_tracing(f, fc, beta0, rb, rm, r0, r):
    ym = rm-rb
    F = f/fc
    A = 1 - 1/F**2 + (rb/(F*ym))**2
    B = -2*rm*rb**2/F**2/ym**2
    C = (rb*rm/(F*ym))**2 - r0**2*np.cos(beta0)**2

    rt = -(B + np.sqrt(B**2-4*A*C))/2/A

    if (r0 < r and r < rb): 
       beta = np.arccos(r0/r*np.cos(beta0))
       D = r0 * (beta-beta0)
    elif (rb < r and r < rt): 
       X = A*r**2 + B*r + C
       Xb = rb**2 - r0**2*np.cos(beta0)**2
       beta = np.arccos(r0/rb*np.cos(beta0))
       D = r0 * (beta-beta0) + \
           r0**2*np.cos(beta0)/np.sqrt(C) * \
           np.log((r*(2*C+rb*B+2*np.sqrt(C*Xb))) / \
                  (rb*(2*C+r*B+2*np.sqrt(C*X))))
    else:
       D = 0

    return D


# https://stackoverflow.com/questions/42426095/matplotlib-contour-contourf-of-concave-non-gridded-data

def apply_mask(triang, x, y, alpha=0.4):
    # Mask triangles with sidelength bigger some alpha
    triangles = triang.triangles
    # Mask off unwanted triangles.
    xtri = x[triangles] - np.roll(x[triangles], 1, axis=1)
    ytri = y[triangles] - np.roll(y[triangles], 1, axis=1)
    maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
    # apply masking
    triang.set_mask(maxi > alpha)


# define some ionosphere parameters

re = 6.3712e6

Nm = 1.0e12
rm = 300.0e3 + re
ym = np.array([20, 50, 100, 200, 250])
ym = ym*1.0e3

rb = rm - ym

rt0 = rm*rb/(rb-ym)


np.set_printoptions(precision=2)


print ('ym  = ', ym/1000)
print ('rb  = ', (rb-re)/1000)
print ('rt0 = ', (rt0-re)/1000)


# critical frequency
fc = 1/(2*np.pi)*np.sqrt(Nm*(1.602e-19)**2/8.854e-12/9.109e-31)

print ('fc = ', '%.2f' % (fc*1.0e-6))


# define different cases

# operating frequencies
f = np.array([5, 10, 15, 20])
f = f * 1.0e6

# elevation angles
beta0 = np.array([5, 10, 15, 20])
beta0 = beta0 * np.pi/180


# ro is used as the radius of the earth in ray tacying calculation
r0 = re


# define ray path 

npts  = 100
npts2 = npts*2-1
npts3 = npts*npts2

D  = np.zeros(npts)
Ne = np.zeros(npts)

x1 = np.zeros(npts2)
y1 = np.zeros(npts2)

theta1 = np.zeros(npts2)

x2 = np.zeros(npts2)
y2 = np.zeros(npts2)

r3 = np.zeros(npts3)
theta3 = np.zeros(npts3)

ne3 = np.zeros(npts3)


# plot ray path

rows = 2
columns = 2

grid = plt.GridSpec(rows, columns, wspace = 0.20, hspace = 0.25)


# 1) loop for ionospheres
for i in range(5):

    plt.figure(figsize=(12, 6))

    # 2) loop for frequencies
    for j in  range(4):

        # ionosphere heatmap

        r2     = np.linspace(rb[i]+1e-4, rt0[i]-1e-4, num=npts)
        theta2 = np.pi/180*np.linspace(-15, 15, num=npts2)

        x2[0:npts]     = r2*np.sin(theta2[0:npts])
        x2[npts:npts2] = np.flip(r2[:npts-1])*np.sin(theta2[npts:npts2])

        y2[0:npts]     = r2*np.cos(theta2[0:npts])
        y2[npts:npts2] = np.flip(r2[:npts-1])*np.cos(theta2[npts:npts2])

        #print ('x2 = ', min(x2)/1000, max(x2)/1000)
        #print ('y2 = ', min(y2)/1000, max(y2)/1000)

        x1[0:npts]     = r0*np.sin(theta2[0:npts])
        x1[npts:npts2] = r0*np.sin(theta2[npts:npts2])

        y1[0:npts]     = r0*np.cos(theta2[0:npts])
        y1[npts:npts2] = r0*np.cos(theta2[npts:npts2])

        for l in range(npts):
            Ne[l] = qp_model(Nm, rb[i], rm, r2[l])

        for m in range(npts):
            r3    [m*npts2:(m+1)*npts2] = r2[m]
            theta3[m*npts2:(m+1)*npts2] = theta2[:]
            ne3   [m*npts2:(m+1)*npts2] = Ne[m]

        x3 = r3*np.sin(theta3)
        y3 = r3*np.cos(theta3)

        x3 = x3/1000
        y3 = (y3-r0)/1000

        #print ('x3 = ', min(x3)/1000, max(x3)/1000)
        #print ('y3 = ', min(y3)/1000, max(y3)/1000)

        triang = tri.Triangulation(x3, y3)


        apply_mask(triang, x3, y3, alpha=1.0e2)


        plt.subplot(grid[j//2,j%2])

        plt.tripcolor(triang, (ne3), shading='flat', cmap='viridis')
        plt.colorbar()

        plt.plot(x1/1000, (y1-r0)/1000)


        #plt.show()


        # 3) loop for elevation angles
        for k in range(4):

            # compute some parameters

            F = f[j]/fc
            A = 1 - 1/F**2 + (rb[i]/(F*ym[i]))**2
            B = -2*rm*rb[i]**2/F**2/ym[i]**2
            C = (rb[i]*rm/(F*ym[i]))**2 - r0**2*np.cos(beta0[k])**2

            rt = -(B + np.sqrt(B**2-4*A*C))/2/A
            rt_max = -B/2/A
            beta_p = np.arccos(np.sqrt(-B/2/r0**2*(rm+B/2/A)))

            print ('rt     = ', '%.2f' % ((rt-r0)/1000    ))
            print ('rt_max = ', '%.2f' % ((rt_max-r0)/1000))
            print ('beta_p = ', '%.2f' % (beta_p*180/np.pi))


            # ray tracing

            r = np.linspace(r0+1e-4, rt-1e-4, num=npts)
            for l in range(npts):
                D[l] = ray_tracing(f[j], fc, beta0[k], rb[i], rm, re, r[l])

            #print ('D = ', D)

            theta = D/r0

            #print ('theta = ', theta*180/np.pi)

            theta1[0:npts] = theta[0:npts]
            for l in range(npts-1):
                dtheta = theta[npts-1-l] - theta[npts-2-l]
                theta1[npts+l] = theta1[npts+l-1] + dtheta

            #print ('theta1 = ', theta1*180/np.pi)

            # all rays initiated from theta=-10 degrees on the earth's surface
            theta1 = theta1 - 10*np.pi/180


            x2[0:npts]     = r*np.sin(theta1[0:npts])
            x2[npts:npts2] = np.flip(r[:npts-1])*np.sin(theta1[npts:npts2])

            y2[0:npts]     = r*np.cos(theta1[0:npts])
            y2[npts:npts2] = np.flip(r[:npts-1])*np.cos(theta1[npts:npts2])


            plt.plot(x2/1000, (y2-r0)/1000, 
                     label=r"$\beta_0={:d}$".format(int(beta0[k]*180/np.pi)))
            plt.legend(loc="upper right", fontsize=7)


            plt.annotate(r"$f={:d}$".format(int(f[j]*1.0e-6)),
                        xy=(0.02, 0.92), xycoords="axes fraction", fontsize=10)


    plt.savefig("../figs/ray_tracing_{}.png".format(i), bbox_inches='tight')
    plt.show()


