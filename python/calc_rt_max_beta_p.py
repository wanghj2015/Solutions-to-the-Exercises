
# compute ray tracing parameters: rt, rt_max, beta_p


import numpy as np


# define some ionosphere parameters

re = 6.3712e6

Nm = 10.0**12
rm = 300.0e3 + re
ym = np.array([20.0, 50.0, 100.0, 200.0, 250.0])
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


# compute ray tracing parameters: rt, rt_max, beta_p


# loop for ionospheres
for i in range(5):
    # loop for frequencies
    for j in  range(4):
        F = f[j]/fc
        A = 1 - 1/F**2 + (rb[i]/(F*ym[i]))**2
        B = -2*rm*rb[i]**2/F**2/ym[i]**2
        C = (rb[i]*rm/(F*ym[i]))**2 - r0**2*np.cos(beta0)**2

        rt = -(B + np.sqrt(B**2-4*A*C))/2/A
        rt_max = -B/2/A
        beta_p = np.arccos(np.sqrt(-B/2/r0**2*(rm+B/2/A)))


        #print ('f, beta0 ', f[j]*1.0e-6, beta0*180/np.pi)
        #print ('rt     = ', (rt-re)/1000    )
        #print ('rt_max = ', '%.2f' % ((rt_max-r0)/1000))
        #print ('beta_p = ', '%.2f' % (beta_p*180/np.pi))


        # output for latex table
        print ('{:d}'.format(int(f[j]*1.0e-6)), '&', \
               ' & '.join(map(str, np.round((rt-re)/1000,2))), \
               '&', '%.2f' % ((rt_max-r0)/1000), \
               '&', '%.2f' % (beta_p*180/np.pi), '\\\\')



