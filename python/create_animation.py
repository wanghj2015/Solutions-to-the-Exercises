

# create animation


import glob
import imageio


images = []

filenames = sorted(glob.glob('../movie/case1_*.png'))
#filenames = sorted(glob.glob('../movie/case2_*.png'))

for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave('./case1.gif', images)
#imageio.mimsave('./case2.gif', images)



