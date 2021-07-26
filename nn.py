#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import os
from PIL import Image

from MMD_kernels import norm_square, gaussian_kernel


epsilon = 1.024*(0.5**float(sys.argv[1]))
xname = sys.argv[2]
yname = sys.argv[3]


# prepare data
X_train = np.load('data/'+xname+'.npy')
Y_train = np.load('data/'+yname+'.npy')
sample_x, dimension_x = np.shape(X_train)
sample_y, dimension_y = np.shape(Y_train)
X_tensor = torch.from_numpy(X_train)
Y_tensor = torch.from_numpy(Y_train)

# # prepare validation data
# X_validate = np.load('data/spiral_validation.npy')
# Y_validate = np.load('data/wave_validation.npy')
# sample_val, dimval_x = np.shape(X_validate)
# sample_val, dimval_y = np.shape(Y_validate)
# X_valten = torch.from_numpy(X_validate)
# Y_valten = torch.from_numpy(Y_validate)



# batch_size is batch size;
# hidden_layer is hidden dimension;
# sigma is the gaussian kernel bandwidth;
# epsilon is the multiplier on distortion
# mx ,my are multiplier on MMDs
hidden_layer = 200
epochs = 3000
batch_size = 400
sigma_x = 10
sigma_y = 10

mx = 1
my = 1

def distortion(X,FX,Y,GY):
    delta1 = torch.mean( torch.abs(  norm_square(X,X) -  norm_square(FX,FX)  )  )
    delta2 = torch.mean( torch.abs(  norm_square(Y,Y) -  norm_square(GY,GY)  )  )
    delta3 = torch.mean( torch.abs(  norm_square(X,GY) -  norm_square(FX,Y)  )  )
    return (delta1 + delta2 + 2*delta3)


class generator_x_y(nn.Module):
    def __init__(self):
        super(generator_x_y, self).__init__()
        
        self.lin1 = nn.Linear(dimension_x, hidden_layer)
        self.lin2 = nn.Linear(hidden_layer, hidden_layer)
        self.lin3 = nn.Linear(hidden_layer, hidden_layer)

        # self.lin1 = nn.utils.spectral_norm(nn.Linear(dimension_x, hidden_layer))
        # self.lin2 = nn.utils.spectral_norm(nn.Linear(hidden_layer, hidden_layer))
        # self.lin3 = nn.utils.spectral_norm(nn.Linear(hidden_layer, hidden_layer))
        self.lin_end = nn.Linear(hidden_layer, dimension_y)
        
    def forward(self, input):
        y = Fn.leaky_relu(self.lin1(input))
        y = Fn.leaky_relu(self.lin2(y))
        y = Fn.leaky_relu(self.lin3(y))
        y = self.lin_end(y)
        
        return y



class generator_y_x(nn.Module):
    def __init__(self):
        super(generator_y_x, self).__init__()
               
        self.lin1 = nn.Linear(dimension_y, hidden_layer)
        self.lin2 = nn.Linear(hidden_layer, hidden_layer)
        self.lin3 = nn.Linear(hidden_layer, hidden_layer)

        # self.lin1 = nn.utils.spectral_norm(nn.Linear(dimension_y, hidden_layer))
        # self.lin2 = nn.utils.spectral_norm(nn.Linear(hidden_layer, hidden_layer))
        # self.lin3 = nn.utils.spectral_norm(nn.Linear(hidden_layer, hidden_layer))
        self.lin_end = nn.Linear(hidden_layer, dimension_x)
        
    def forward(self, input):
        x = Fn.leaky_relu(self.lin1(input))
        x = Fn.leaky_relu(self.lin2(x))
        x = Fn.leaky_relu(self.lin3(x))
        x = self.lin_end(x)
        
        return x


dtype = torch.DoubleTensor
F = generator_x_y().type(dtype)
G = generator_y_x().type(dtype)

sigma_x = torch.median( norm_square(X_tensor, X_tensor) )
sigma_y = torch.median( norm_square(Y_tensor, Y_tensor) )

G_optim = optim.Adam(G.parameters(), lr=1e-3)
F_optim = optim.Adam(F.parameters(), lr=1e-3)

for epoch in range(epochs):
    x_train_dataloader = DataLoader(X_tensor, batch_size=batch_size, shuffle=True)
    y_train_dataloader = DataLoader(Y_tensor, batch_size=batch_size, shuffle=True)
    print('Epoch number: {0}'.format(epoch))
    i=0
    for X_batch, Y_batch in zip(x_train_dataloader,y_train_dataloader):
        FX = F(X_batch)
        GY = G(Y_batch)
        MMD_x = torch.mean( gaussian_kernel(sigma_x, X_batch, X_batch) ) + torch.mean( gaussian_kernel(sigma_x, GY, GY) ) - 2*torch.mean( gaussian_kernel(sigma_x, GY, X_batch) )
        MMD_y = torch.mean( gaussian_kernel(sigma_y, Y_batch, Y_batch) ) + torch.mean( gaussian_kernel(sigma_y, FX, FX) ) - 2*torch.mean( gaussian_kernel(sigma_y, FX, Y_batch) )
        loss_function = mx*MMD_x + my*MMD_y + epsilon*distortion(X_batch,FX,Y_batch,GY)
        
        G_optim.zero_grad()
        F_optim.zero_grad()
        
        loss_function.backward()
        G_optim.step()
        F_optim.step()

        if i % 10 == 0:
            # print('**Batch number: {0}**'.format(i))
            print('loss: {0}'.format(loss_function.item()))

        i+=1
    # FXval = F(X_valten)
    # GYval = G(Y_valten)
    # MMDval_x = torch.mean( gaussian_kernel(sigma_x, X_valten, X_valten) ) + torch.mean( gaussian_kernel(sigma_x, GYval, GYval) ) - 2*torch.mean( gaussian_kernel(sigma_x, GYval, X_valten) )
    # MMDval_y = torch.mean( gaussian_kernel(sigma_y, Y_valten, Y_valten) ) + torch.mean( gaussian_kernel(sigma_y, FXval, FXval) ) - 2*torch.mean( gaussian_kernel(sigma_y, FXval, Y_valten) )
    # validation_loss = mx*MMDval_x + my*MMDval_y + epsilon*distortion(X_valten,FXval,Y_valten,GYval)
    # print('validation loss: {0}'.format(validation_loss.item()))
    
    # if epoch % 50 == 0:
    #     fx=F(X_tensor).detach().numpy()
    #     gy=G(Y_tensor).detach().numpy()
    #     plt.scatter(fx[:,0],fx[:,1],c = fx[:,1])
    #     plt.show()
    #     plt.scatter(gy[:,0],gy[:,1],c = gy[:,1])
    #     plt.show()

    #     # 3d
    #     gytest=G(Y_tensor).detach().numpy()
    #     ax = plt.axes(projection='3d')
    #     ax.scatter3D(gytest[:,0], gytest[:,1], gytest[:,2], c=gytest[:,2])
    #     plt.show()
    #     # 2d
    #     fxtest=F(X_tensor).detach().numpy()
    #     plt.scatter(fxtest[:,0],fxtest[:,1],c = fxtest[:,1])
    #     plt.show()


# log for results
result_path = 'results/'+xname+'_'+yname+'/'

FX = F(X_tensor)
GY = G(Y_tensor)
MMD_x = torch.mean( gaussian_kernel(sigma_x, X_tensor, X_tensor) ) + torch.mean( gaussian_kernel(sigma_x, GY, GY) ) - 2*torch.mean( gaussian_kernel(sigma_x, GY, X_tensor) )
MMD_y = torch.mean( gaussian_kernel(sigma_y, Y_tensor, Y_tensor) ) + torch.mean( gaussian_kernel(sigma_y, FX, FX) ) - 2*torch.mean( gaussian_kernel(sigma_y, FX, Y_tensor) )
delta = distortion(X_tensor,FX,Y_tensor,GY)
gmmd = MMD_x+MMD_y+epsilon*delta
results = np.array([[epsilon,gmmd.item(),MMD_x.item(),MMD_y.item(),delta.item()]])

logfile = np.load(result_path+xname+'_'+yname+'.npy')
logfile = np.append(logfile, results, axis=0)
np.save(result_path+xname+'_'+yname,logfile)

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# results
fx=F(X_tensor).detach().numpy()
gy=G(Y_tensor).detach().numpy()
# plt.scatter(fx[:,0],fx[:,1],c = fx[:,1])
# plt.show()
# ax = plt.axes(projection='3d')
# ax.scatter3D(gy[:,0], gy[:,1], gy[:,2], c=gy[:,2])
# plt.show()

# # consistency
fgy=F(G(Y_tensor)).detach().numpy()
gfx=G(F(X_tensor)).detach().numpy()
# plt.scatter(fx[:,0],fx[:,1],c = fx[:,1])
# plt.show()
# ax = plt.axes(projection='3d')
# ax.scatter3D(gy[:,0], gy[:,1], gy[:,2], c=gy[:,2])
# plt.show()



np.save(result_path+'fx_{e}'.format(e=epsilon),fx)
np.save(result_path+'gy_{e}'.format(e=epsilon),gy)
np.save(result_path+'fgy_{e}'.format(e=epsilon),fgy)
np.save(result_path+'gfx_{e}'.format(e=epsilon),gfx)
