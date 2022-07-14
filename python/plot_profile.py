
# plot QP ionosphere profiles


# import some libraries
import numpy as np
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


# plot profiles

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


# divide r (rb <= r <= rt) to 'npts' 
npts = 100
Ne = np.zeros(npts)

plt.figure(figsize=(4, 6))

for i in range(5):
    r = np.linspace(rb[i], rt[i], num=npts)
    for k in range(npts):
        Ne[k] = qp_model(Nm, rb[i], rm, r[k])

    plt.plot(Ne[1:npts-1], (r[1:npts-1]-re)/1000, 
             label=r"$y_m={:d}$".format(int(ym[i]/1000)))
    plt.legend(loc="lower right", fontsize=7)

plt.xscale('log')
plt.ylim(50, 300)

plt.xlabel("electron density [1/m^3]")
plt.ylabel("altitude [km]")
 
plt.savefig("../figs/qp_profile.pdf", bbox_inches='tight')

plt.show()


