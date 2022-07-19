from Base_Stein.SVGD_functions import *

class base_SVGD:
    """Base SVGD object - input N number of particles, zdim size of normal dist.
        Also inits shape of normal dist, and most importantly inits the particles randomly.
        Has method to compute the gradient and dynamically update sig.
    """

    def __init__(self,N=100,zdim=1,device='cpu'):

        self.N=N #number of particles
        self.zdim=zdim #dimensionality of particles

        self.loc=torch.zeros(zdim,1).to(device) #mean of normal distribution
        self.loc_t = self.loc.T

        self.cov_mat=torch.eye(zdim).to(device) #covariance
        self.inv_cov_mat = torch.inverse(self.cov_mat)

        #self._Particles = 10*torch.rand(self.N,self.zdim)-5 #initialize particles
        self._Particles = None

    @property
    def Particles(self):
        return self._Particles

    @Particles.setter
    def Particles(self, val):
        self._Particles = val


    @property
    def sig(self):
        #Get pairwise distance between particles (used to compute sig, kernel width adaptively on each iteration)

        d2 = self.d2 #get d2
        d = d2**0.5 #Problem: may be nan after this step. e.g. if d2 <=0 through rounding errors
        d=torch.nan_to_num(d,posinf=float('inf'),neginf=-float('inf')) #replace nan but leave inf values
        sig = 0.5*torch.median(d)**2 / math.log(self.N) #Adaptively

        return sig #Dynamically set sig

    def get_gradient(self):

        Particles=self.Particles
        N=self.N

        self.d2 = norm2(Particles) #set d2

        sig=self.sig #Inside sig we get d2

        #Getting ugly. Clearly phi only depends on the class
        g = phi(Particles=Particles,inv_cov_mat=self.inv_cov_mat,\
                loc_t=self.loc_t,NORM2=self.d2,N=N,sig=sig
               )

        return g

class trainable_SVGD(base_SVGD):

    def __init__(self,N=100,zdim=1,device='cpu',eta=0.001,beta1=0.9,beta2=0.999,eps=1e-8):
        super().__init__(N,zdim,device)

        self.eta=eta #initial learning rate

        self.beta1=beta1 #Adam specific hps
        self.beta2=beta2
        self.eps=eps

        self._m=torch.zeros(N,zdim)
        self._v=torch.zeros(N,zdim)

        self._x=None

    #Open problem: Reduce having to repeat this twice for m and v respectively.
    #There is obvious redundancy here.
    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        self._m = value

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = value

    def AdamStep(self,gradient):
        """Runs one step of Adam, given gradient.
        """

        #Current values, which all get updated
        Particles = self.Particles

        m = self.m
        v = self.v

        #hps
        eta = self.eta
        eps = self.eps
        beta1 = self.beta1
        beta2 = self.beta2

        #Update m and v
        self.m = beta1*m + (1-beta1)*gradient
        self.v = beta2*v + (1-beta2)*gradient.pow(2)

        mhat = self.m/(1-beta1)
        vhat = self.v/(1-beta2)

        #One step of Adam
        self.Particles = self.Particles + eta/(vhat.pow(0.5) + eps) * mhat
        return self.Particles,self.m,self.v

    def SgdStep(self,gradient):
        eta = self.eta
        self.Particles = self.Particles + eta*gradient
        return self.Particles
