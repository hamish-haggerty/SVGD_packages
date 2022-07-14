#TODO: Write tests for the other functions in SVGD_functions.
#TODO: Remove redundancy in the code within this test
#This might be helpful for the latter: https://stackoverflow.com/questions/51739589/how-to-share-parametrized-arguments-across-multiple-test-functions

#Test all the basic functions defined in SVGD_functions
from Base_Stein.SVGD_functions import *
import pytest

torch.set_default_dtype(torch.float64) #double precision

@pytest.mark.parametrize('i,j,N,zdim',
[(0,0,1,1),(1,1,10,1),(2,3,15,2)])

def test1_norm2(i,j,N,zdim):
    """Test norm2 function with several different values of: N=number of particles,zdim=dim of each
        particle and i,j the index."""

    X = torch.rand(N,zdim)

    assert norm2(X)[i,j] - torch.linalg.vector_norm(X[i]-X[j])**2 < 1e-06


@pytest.mark.parametrize('i,j,N,zdim,sig',
[(0,0,1,1,1.0),(1,1,10,1,0.5),(2,3,15,2,1.01)])

def test1_rbf(i,j,N,zdim,sig):

    X = torch.rand(N,zdim)
    NORM2 = norm2(X)

    assert rbf(NORM2,sig=sig)[i,j] - torch.exp((-1/(2*sig)*NORM2[i,j])) < 1e-6

@pytest.mark.parametrize('N,zdim',
[(1,1),(10,1),(10,2)])
def test1_score(N,zdim):
    #Only tested for basic case of identity covariance matrix in which case score(x)=-x


    #Only tested for this basic case of zero mean and identity covariance matric
    loc = torch.zeros(zdim,1)
    loc_t = loc.T
    cov_mat=torch.eye(zdim)
    inv_cov_mat = torch.inverse(cov_mat)
    Particles = torch.rand(N,zdim)

    torch.mm(inv_cov_mat, (Particles-loc_t).T).T

    assert torch.all(score(Particles,inv_cov_mat,loc_t) == -Particles)


def numerical_grad_kernel(X,i,j,N,zdim,sig,component,tol=1e-4):
    """Get numerical partial derivative of K[X[i],X[j]] with respect to `component' variable of first vector argument X[i].
        i.e. write out in full: X[i]=(v1,...,v_component,...) and want numerical approx to: (d/dv_component) K(X[i],X[j])
    """

    h = torch.zeros(zdim)
    h[component]=tol #i.e. (0,0,...,tol,0,...,0)

    X_h_plus = torch.stack((X[i]+h,X[j]))
    NORM2_h = norm2(X_h_plus)
    K_h_plus = rbf(NORM2_h,sig=sig)[0,1] #i.e. K(xi1,...,xi_component+tol,...,y)

    X_h_minus = torch.stack((X[i]-h,X[j]))
    NORM2_h = norm2(X_h_minus)
    K_h_minus = rbf(NORM2_h,sig=sig)[0,1] #i.e. K(xi1,...,xi_component-tol,...,y)

    return (K_h_plus-K_h_minus).item()/(2*tol)


#Note: redundancy here
@pytest.mark.parametrize('i,j,N,zdim,sig',
[(0,0,1,1,1.0),(1,1,10,5,0.5),(2,3,15,6,1.01),(5,5,20,7,0.01)])
def test_numerically_grad_kernel(i,j,N,zdim,sig):
    """Numerically compute gradient of kernel and check that it is
        close to the analytic gradients. See numerical_grad_kernel(...) above.
    """

    X = torch.rand(N,zdim) #random X
    for component in range(zdim): #i.e. check the partial derivative (d/dv) K(X[i],X[j]) where v=X[i][component] a real number

        #get numerical partial derivative with respect to 'component' variable v of X[i], in rbf(X[i],X[j])
        num_grad = numerical_grad_kernel(X,i,j,N,zdim,sig,component,tol=6*1e-6)

        #analytic derivative
        NORM2 = norm2(X)
        K = rbf(NORM2,sig=sig)
        grad = grad_kernel(X,rbf_xy=K,sig=sig)[i,j][component]

        abs_grad = abs(grad)
        abs_num_grad = abs(num_grad)

        #compute relative error
        if abs_grad>0:
            RE = abs(grad-num_grad)/(max(abs_grad,abs_num_grad))
            assert RE<1e-6 #want nice small relative error

        else:
            assert abs(num_grad)<1e-6 #want nice small relative error
