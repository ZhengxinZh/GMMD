
import torch
from torch import nn
import numpy as np
from PIL import Image
import sys
# import matplotlib.pyplot as plt


def load_img(fn='img/heart.png', size=200, max_samples=4000):
    r"""Returns x,y of black pixels (between -1 and 1)
    """
    pic = np.array(Image.open(fn).resize((size,size)).convert('L'))
    y_inv, x = np.nonzero(pic<=128)
    y = size - y_inv - 1
    if max_samples and x.size > max_samples:
        ixsel = np.random.choice(x.size, max_samples, replace=False)
        x, y = x[ixsel], y[ixsel]
    return np.stack((x, y), 1) / size * 2 - 1

x=load_img()
# plt.scatter(x[:,0],x[:,1],c = x[:,1])
# plt.show()
np.save('data/heart',x)

# rotated
# x=load_img()
radian = np.pi/3
rotate_mat  = np.matrix([[np.cos(radian),np.sin(radian)],[-np.sin(radian),np.cos(radian)]])
y=np.array(np.matmul(x,rotate_mat))
np.save('data/heart_rotate',y)

# scaled
z = 3*x
np.save('data/heart_scale',z)

# 3d
radian = np.pi/3
angle = np.pi/3
embed_mat  = np.matrix([[np.cos(radian)*np.cos(angle),np.sin(radian)*np.cos(angle),np.sin(angle)],[-np.cos(radian)*np.sin(angle),-np.sin(radian)*np.sin(angle),np.cos(angle)]])
vec=np.cross(embed_mat[0,:],embed_mat[1,:])
embed_mat[1,:]=vec
w = np.array(np.matmul(x,embed_mat))
np.save('data/heart_embed',w)



# colorImage  = Image.open("cat/cat.png")
# rotated     = colorImage.rotate(60)
# rotated.show()


