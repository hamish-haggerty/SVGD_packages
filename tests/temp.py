from Base_Stein.SVGD_functions import *


i,j=0,0
N=2
zdim=2
sig=1.0
component = 0

#i) compute rbf(x,y)
X = torch.rand(N,zdim)


NORM2 = norm2(X)
K = rbf(NORM2,sig=sig)

print(X)
print(K)
input()


#Issue here is we compute rbf as rbf(norm2(X)); i.e. pairwise
#To compute rbf(x+h,y) we need to form the tensor with [x+h,y] in it
#ii) compute rbf(x+h,y)


tol=1e-4
h = torch.zeros(zdim)
h[component]=tol


X_h_plus = torch.stack((X[i]+h,X[j]))
NORM2_h = norm2(X_h_plus)
K_h_plus = rbf(NORM2_h,sig=sig)[0,1]

X_h_minus = torch.stack((X[i]-h,X[j]))
NORM2_h = norm2(X_h_minus)
K_h_minus = rbf(NORM2_h,sig=sig)[0,1]

print(K_h_plus)
print(K_h_minus)
input()


#iii) compute numerical gradient: using finite difference formula: (f(x+h)-f(x-h))/(2h)
num_grad = (K_h_plus-K_h_minus)/(2*tol)

print(num_grad)
input()

#iv) compute analytic gradient
grad_comp = grad_kernel(X,rbf_xy=K,sig=sig)[i,j][component] #do d/dxi K(x1,...,xi,...,xn,y). Remember grad is a vector

print(grad_comp)
input()

#v) compute relative error percentage
RE = abs(num_grad-grad_comp)/(grad_comp)

print(num_grad)
print(grad_comp)


print(RE)
print('RE above...')
