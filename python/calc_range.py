
# calculate range for plane earth and ionosphere


import numpy as np


re = 6.3712e6

Nm = 1.0e12
rm = 300.0e3 + re
ym = np.array([20, 50, 100, 200, 250])
ym = ym*1.0e3

rb = rm - ym

rt = rm*rb/(rb-ym)


np.set_printoptions(precision=2)


print ('ym = ', ym/1000)
print ('rb = ', (rb-re)/1000)
print ('rt = ', (rt-re)/1000)


# calculate virtual height

# critical frequency
fc = 1/(2*np.pi)*np.sqrt(Nm*(1.602e-19)**2/8.854e-12/9.109e-31)


print ('fc = ', fc)


# equivalent vertical incidence frequency for calculating $h'_max$
#fv = (1.0 - 1.0e-16) * fc
fv = (1.0 - 1.0e-12) * fc

hb = rb - re
hv = hb + 0.5*ym*fv/fc*np.log((fc+fv)/(fc-fv))


print ('hb = ', hb/1000)
print ('hv = ', hv/1000)


# incidence angles
theta = np.array([25, 30, 35, 40, 45])

# operation frequency
fo = fv / np.cos(theta * np.pi / 180)
 

print ('fo = ', fo*1.0e-6)


for i in range(5):
    D = 2 * hv * np.tan(theta[i] * np.pi / 180)

    print ('D = ', D/1000)


