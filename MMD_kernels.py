#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import sys
import torch

def norm_square(X1,X2):
    batch_size_1 = X1.size(0)
    batch_size_2 = X2.size(0)

    # norm1 = (torch.sum(torch.square(X1),1)).t()
    # norm1 = norm1.expand(batch_size_1, batch_size_2)
    # norm2 = (torch.sum(torch.square(X2),1)).t()
    # norm2 = norm2.expand(batch_size_2, batch_size_1)
    # square =  norm1- 2*torch.mm(X1, X2.t()) + norm2.t()
    # square = torch.abs(square)

    expand_x1 = X1.expand(batch_size_2,batch_size_1,-1)
    expand_x1 = torch.transpose(expand_x1,0,1)
    expand_x2 = X2.expand(batch_size_1,batch_size_2,-1)
    square = torch.sum(torch.square(expand_x1-expand_x2),2)

    return square


# Define the kernel functions for X and Y
# X1: batch_size * dimension of X
# sigma: band_width
def gaussian_kernel(sigma,X1,X2):
    square = norm_square(X1,X2)
    val=torch.exp(-square/sigma) + torch.exp(-square/(sigma*0.25))  + torch.exp(-square/(sigma*0.05))  + torch.exp(-square/(sigma*4))  + torch.exp(-square/(sigma*20))  + torch.exp(-square/(sigma*0.01)) + torch.exp(-square/(sigma*0.001))+torch.exp(-square/(sigma*0.0001))+ torch.exp(-square/(sigma*100))+ torch.exp(-square/(sigma*1000))
    return val


# rational quadratic
# a: parameter
def rational_kernel(a,X1,X2):
    square = norm_square(X1,X2)
    val = torch.pow(1+square/(2*a), -a)
    return val