#Functions use
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import inspect
import numpy as np

def norm2(X):
    """"Vectorized implementation of norm squared. X is collection of N x,y points, and
        returns Y, where Y[i,j] = norm(X[i],X[j])**2. X has shape N*zdim, Y has shape NxN.
    """

    #Use formula ||(x-y)||**2 = ||x||**2 + ||y||**2 - 2x.T*y. in 1d this is just the
    #familiar formula (x-y)**2 expanded.

    X_norm = torch.sum(X ** 2, axis = -1) #gives x[i]**2 for each i

    #This First part is "matrix mult" viewed as addition
    tem1 = X_norm[:,None] + X_norm[None,:] #gives x[i]**2 + x[j]**2 for each i,j

    tem2 = - 2 * torch.mm(X, X.T) #last part of above formula

    return tem1 + tem2


def rbf(NORM2,sig=1.0):
    """"Assumes have already computed norm2(X). So can call as: rbf(norm2(X),sig)
    """

    assert(sig>0)
    assert NORM2.shape[0] == NORM2.shape[1]

    return torch.exp(-1/(2*sig)*NORM2)


#Need gradient of log_density, or score function, for N(zdim)
def score(X,inv_cov_mat,loc_t):

    tem = -1*torch.mm(inv_cov_mat, (X-loc_t).T).T #broadcasting

    return tem

    #Need gradient of kernel wrt FIRST argment

#Grad kernel works just passing in x,y,sig, OR if pass in, additionally, x,y,sig and array of
#precomputed rbf_xy with an index i,j that corresponds to rbf_xy[i,j,sig]=rbf(x,y,sig)
def grad_kernel(Particles,rbf_xy=None,sig=1.0):
    """"Since gradient of kernel depends on z=rbf(x,y) we just compute it once
        elsewhere and pass it in here. IMPORTANT: Computed wrt first argument x.
    """

    x_y = (Particles[:,None]-Particles[None,:]) #This is collection of all: Particles[i]-Particles[j] over all i,j

    rbf_xy=torch.unsqueeze(rbf_xy,-1)#make N*N*1 for broadcasting, since x_y is N*N*zdim

    return (-1/sig)*rbf_xy*(x_y) #


    #syntax of decorator: dec(grad_kernel(X,...)) = grad_kernel(X[i],X[j],...)

    #g = (-1/sig)*rbf_xy*(x-y) formula given all the values.

#Now have 1,2,3 so essentially have phi

def phi(Particles,inv_cov_mat,loc_t,NORM2,N,sig):

    K = rbf(NORM2,sig=sig) #kernel

    S = score(Particles,inv_cov_mat,loc_t).view(N,-1) #score

    DK = grad_kernel(Particles,rbf_xy=K,sig=sig).view(N,N,-1) #derivative of kernel


            #First term             #Second term
    return (1/N)*(torch.mm(K.T,S) + torch.sum(DK,axis=0))
