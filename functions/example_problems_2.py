import numpy as np

from smt.problems.problem import Problem

class ALOSDim(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[1, 2, 3], types=int)
        self.options.declare("name", "ALOSDim", types=str)

    def _setup(self):
        self.xlimits[:, 0] = 0.0
        self.xlimits[:, 1] = 1.0

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape

        dim = self.options["ndim"]
        a = 30. if dim == 1 else 21.
        b = 0.9 if dim == 1 else 0.7

        y = np.zeros((ne, 1), complex)
        if kx is None:
            y[:,0] = np.sin(a*(x[:,0] - 0.9)**4)*np.cos(2.*(x[:,0] - 0.9)) + (x[:,0] - b)/2.
            if dim > 1:
                y[:,0] += 2.*x[:,1]*x[:,1]*np.sin(x[:,0]*x[:,1])
            if dim > 2:
                y[:,0] += 3.*x[:,2]*x[:,2]*np.sin(x[:,0]*x[:,1]*x[:,2])
        elif kx == 0:
            y[:,0] = np.cos(a*(x[:,0] - 0.9)**4)*np.cos(2.*(x[:,0] - 0.9))*(4*a*(x[:,0] - 0.9)**3)
            y[:,0] += -np.sin(a*(x[:,0] - 0.9)**4)*np.sin(2.*(x[:,0] - 0.9))*2.
            y[:,0] += 0.5
            if dim > 1:
                y[:,0] += 2.*x[:,1]*x[:,1]*x[:,1]*np.cos(x[:,0]*x[:,1])
            if dim > 2:
                y[:,0] += 3.*x[:,1]*x[:,2]*x[:,2]*x[:,2]*np.cos(x[:,0]*x[:,1]*x[:,2])
        elif kx == 1:
            y[:,0] = 0.
            if dim > 1:
                y[:,0] += 4.*x[:,1]*np.sin(x[:,0]*x[:,1]) + 2.*x[:,1]*x[:,1]*x[:,0]*np.cos(x[:,0]*x[:,1])
            if dim > 2:
                y[:,0] += 3.*x[:,0]*x[:,2]*x[:,2]*x[:,2]*np.cos(x[:,0]*x[:,1]*x[:,2])
            
        elif kx == 2:
            y[:,0] = 0.
            if dim > 2:
                y[:,0] += 6.*x[:,2]*np.sin(x[:,0]*x[:,1]*x[:,2]) +  3.*x[:,0]*x[:,1]*x[:,2]*x[:,2]*np.cos(x[:,0]*x[:,1]*x[:,2])

        return y

class ScalingExpSine(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, types=int)
        self.options.declare("name", "ScalingExpSine", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -2.0
        self.xlimits[:, 1] = 2.0

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape

        dim = self.options["ndim"]
        pi = np.pi
        
        y = np.zeros((ne, 1), complex)
        if kx is None:
            for i in range(dim):
                y[:,0] += np.exp(-0.1*x[:,i])*np.sin(0.5*pi*x[:,i])
        
        elif kx is not None:
            y[:,0] = -0.1*np.exp(-0.1*x[:,kx])*np.sin(0.5*pi*x[:,kx])
            y[:,0] += 0.5*pi*np.exp(-0.1*x[:,kx])*np.cos(0.5*pi*x[:,kx])
            
        y /= dim
        return y
    

class MixedSine(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int)
        self.options.declare("name", "MixedSine", types=str)

    def _setup(self):
        self.xlimits[:, 0] = 0.3
        self.xlimits[:, 1] = 1.0

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape

        y = np.zeros((ne, 1), complex)
        if kx is None:
            y[:,0] = np.sin(1./(x[:,0]*x[:,1]))
        else:
            y[:,0] = -1./(x[:,0]*x[:,1]*x[:,kx])*np.cos(1./(x[:,0]*x[:,1]))
        return y
    
class SimpleConvex(Problem):
    def _initialize(self):
        self.options.declare("ndim", 1, values=[1], types=int)
        self.options.declare("order", 4, types=int)
        self.options.declare("name", "SimpleConvex", types=str)

    def _setup(self):
        self.xlimits[:, 0] = 0.0
        self.xlimits[:, 1] = 1.0

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape
        ord = self.options["order"]

        y = np.zeros((ne, 1), complex)
        if kx is None:
            y[:,0] = x[:,0]**ord
        else:
            y[:,0] = ord*x[:,0]**(ord-1)
        return y

# From Eldred Sandia Report 2009

# outputs limit state function based on material properties and rectangular cross-section shape
# maybe do limit state - some cost function? e.g. 0.25ln(x) or 0.0005 P ln(x)
class ShortColumn(Problem):
    def _initialize(self):
        self.options.declare("name", "ShortColumn", types=str)
        self.options.declare("ndim", 4, types=int)
        self.options.declare("lncost", False, types=bool) # include ln cost function
        self.options.declare("P_bounds", [200, 800], types=list) # 500 \pm 3 sigma, sigma = 100 
        self.options.declare("M_bounds", [800, 3200], types=list) # 2000 \pm 3 sigma, sigma = 400 
        self.options.declare("Y_bounds", [40.93909, 538.03018], types=list) # lognormal  5 \pm 0.5 sigma, sigma = 100 
        self.options.declare("b_bounds", [0.5, 10.], types=list) 
        self.options.declare("h_bounds", [10., 20.], types=list)

    def _setup(self):
        assert self.options["ndim"] > 3, "ndim must be 3 or more, with and without b and/or h"

        # Axial Force P 
        self.xlimits[0, 0] = self.options["P_bounds"][0]
        self.xlimits[0, 1] = self.options["P_bounds"][1]

        # Bending Moment M
        self.xlimits[1, 0] = self.options["M_bounds"][0]
        self.xlimits[1, 1] = self.options["M_bounds"][1]

        # yield stress Y
        self.xlimits[2, 0] = self.options["Y_bounds"][0]
        self.xlimits[2, 1] = self.options["Y_bounds"][1]

        # width b
        self.xlimits[3, 0] = self.options["b_bounds"][0]
        self.xlimits[3, 1] = self.options["b_bounds"][1]

        # depth h
        if self.options["ndim"] == 5:
            self.xlimits[4, 0] = self.options["h_bounds"][0]
            self.xlimits[4, 1] = self.options["h_bounds"][1]

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape

        P = x[:,0]
        M = x[:,1]
        Y = x[:,2]
        b = x[:,3]
        h = 15.
        if nx > 4:
            h = x[:,4]

        y = np.zeros((ne, 1), complex)
        if kx is None:
            y[:, 0] = 1. - 4*M/(b*h*h*Y) - P*P/(b*b*h*h*Y*Y)

            y[:, 0] -= 0.0003*P*np.log(b)
            y[:, 0] += 0.5*(h-15.)*np.exp(-(h-18.)**2)
        else:
            if kx == 0:
                y[:, 0] = -2*P/(b*b*h*h*Y*Y)
                y[:, 0] -= 0.0003*np.log(b)
            elif kx  == 1:
                y[:, 0] = -4/(b*h*h*Y)
            elif kx  == 2:
                y[:, 0] = 4*M/(b*h*h*Y*Y) + 2*P*P/(b*b*h*h*Y*Y*Y)
            elif kx  == 3:
                y[:, 0] = 4*M/(b*b*h*h*Y) + 2*P*P/(b*b*b*h*h*Y*Y)
                y[:, 0] -= 0.0003*P/b
            elif kx  == 4:
                y[:, 0] = 8*M/(b*h*h*h*Y) + 2*P*P/(b*b*h*h*h*Y*Y)
                work = np.exp(-h**2 + 36.*h - 324)
                y[:, 0] += 0.5*work*(-2*h*h + 66*h - 539)



        return -10*y


# 1 uq, 2 dv version
class ShortColumn1U(Problem):
    def _initialize(self):
        self.options.declare("name", "ShortColumn1U", types=str)
        self.options.declare("ndim", 3, types=int)
        self.options.declare("lncost", False, types=bool) # include ln cost function
        self.options.declare("P_bounds", [200, 800], types=list) # 500 \pm 3 sigma, sigma = 100 
        self.options.declare("b_bounds", [0.5, 10.], types=list) 
        self.options.declare("h_bounds", [10., 20.], types=list)

    def _setup(self):
        # Axial Force P 
        self.xlimits[0, 0] = self.options["P_bounds"][0]
        self.xlimits[0, 1] = self.options["P_bounds"][1]

        # width b
        self.xlimits[1, 0] = self.options["b_bounds"][0]
        self.xlimits[1, 1] = self.options["b_bounds"][1]

        # depth h
        self.xlimits[2, 0] = self.options["h_bounds"][0]
        self.xlimits[2, 1] = self.options["h_bounds"][1]

        self.func_wrap = ShortColumn(ndim=5,
                                    lncost=self.options["lncost"],
                                    P_bounds=self.options["P_bounds"],
                                    b_bounds=self.options["b_bounds"],
                                    h_bounds=self.options["h_bounds"],
                                    ) 

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape

        M = 2000.
        Y = 100.
        P = x[:,0]
        b = x[:,1]
        h = x[:,2]
      
        x_w = np.zeros([x.shape[0], 5])
        x_w[:,0] = P
        x_w[:,1] = M
        x_w[:,2] = Y
        x_w[:,3] = b
        x_w[:,4] = h

        if kx is None:
            kx_w = None
        if kx == 0:
            kx_w = 0
        if kx == 1:
            kx_w = 3
        if kx == 2:
            kx_w = 4

        y = self.func_wrap(x_w, kx_w)

        return y







if __name__ == '__main__':

    x_init = 0.
    # N = [5, 3, 2]
    # xlimits = np.array([[-1., 1.],
    #            [-1., 1.],
    #            [-1., 1.],
    #            [-1., 1.]])
    # pdfs =  [x_init, 'uniform', 'uniform', 'uniform']
    # samp1 = CollocationSampler(np.array([x_init]), N=N,
    #                             xlimits=xlimits, 
    #                             probability_functions=pdfs, 
    #                             retain_uncertain_points=True)
    
    import matplotlib.pyplot as plt
    from smt.problems import Rosenbrock
    from smt.sampling_methods import LHS
    from utils.sutils import convert_to_smt_grads
    from scipy.stats import norm

    plt.rcParams['font.size'] = '15'

    ndir = 150
    nu = 5000
    ## ShortCol1U
    func = ShortColumn1U(ndim=3)
    xlimits = func.xlimits
    x1 = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
    x2 = np.linspace(xlimits[2][0], xlimits[2][1], ndir)
    samp = LHS(xlimits=xlimits[0:1], criterion='maximin')
    area = xlimits[0][1] - xlimits[0][0]
    x0 = samp(nu)
    pdf = norm(500, 100)
    weight = pdf.pdf(x0)
    X1, X2 = np.meshgrid(x1, x2)

    TF = np.zeros([ndir, ndir])
    for i in range(len(x1)):
        for j in range(len(x2)):
            xi = np.zeros([nu,3])
            xi[:,1] = x1[i]
            xi[:,2] = x2[j]
            xi[:,0] = x0[:,0]
            eval = func(xi)
            weval = np.dot(eval.T, weight)*area
            TF[i,j] = weval/nu

    # # Plot the target function
    plt.contourf(X1, X2, TF, levels=15, label=f'True')
    plt.xlabel(r"$x_{d,1}$")
    plt.ylabel(r"$x_{d,2}$")
    plt.colorbar()
    #plt.legend(loc=1)
    plt.savefig(f"./2dshortcol1u.png", bbox_inches="tight")
    plt.clf()
    import pdb; pdb.set_trace()

    # # 1d ALOS
    # func1 = ALOSDim(ndim=1)
    # xlimits1 = func1.xlimits
    # x1 = np.linspace(xlimits1[0][0], xlimits1[0][1], ndir)
    # x1 = np.atleast_2d(x1).T
    # TF1 = func1(x1)

    # # Plot the target function
    # plt.plot(x1, TF1, "-k", label=f'True')
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"$f$")
    # plt.title(r"ALOS Function 1D")
    # #plt.legend(loc=1)
    # plt.savefig(f"./1dALOS.pdf", bbox_inches="tight")
    # plt.clf()


    # # 2d ALOS
    # func2 = ALOSDim(ndim=2)
    # xlimits2 = func2.xlimits
    # x1 = np.linspace(xlimits2[0][0], xlimits2[0][1], ndir)
    # x2 = np.linspace(xlimits2[1][0], xlimits2[1][1], ndir)
    # X1, X2 = np.meshgrid(x1, x2)
    # combx = np.concatenate((X1.reshape((ndir*ndir, 1)), X2.reshape((ndir*ndir, 1))), axis=1)
    # TF2 = func2(combx)
    # TF2 = TF2.reshape((ndir, ndir))

    # # Plot the target function
    # plt.contourf(X1, X2, TF2, levels=30, label=f'True')
    # plt.xlabel(r"$x_1$")
    # plt.ylabel(r"$x_2$")
    # plt.title(r"ALOS Function 2D")
    # plt.colorbar()
    # #plt.legend(loc=1)
    # plt.savefig(f"./2dALOS.pdf", bbox_inches="tight")
    # plt.clf()

    # # 1d ExpSin
    # func1 = ScalingExpSine(ndim=1)
    # xlimits1 = func1.xlimits*10
    # x1 = np.linspace(xlimits1[0][0], xlimits1[0][1], ndir)
    # x1 = np.atleast_2d(x1).T
    # TF1 = func1(x1)

    # # Plot the target function
    # plt.plot(x1, TF1, "-k", label=f'True')
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"$f$")
    # plt.title(r"ExpSine Function 1D")
    # #plt.legend(loc=1)
    # plt.savefig(f"./1dES.pdf", bbox_inches="tight")
    # plt.clf()


    # # 2d ExpSin
    # func2 = ScalingExpSine(ndim=2)
    # xlimits2 = func2.xlimits*10
    # x1 = np.linspace(xlimits2[0][0], xlimits2[0][1], ndir)
    # x2 = np.linspace(xlimits2[1][0], xlimits2[1][1], ndir)
    # X1, X2 = np.meshgrid(x1, x2)
    # combx = np.concatenate((X1.reshape((ndir*ndir, 1)), X2.reshape((ndir*ndir, 1))), axis=1)
    # TF2 = func2(combx)
    # TF2 = TF2.reshape((ndir, ndir))

    # # Plot the target function
    # plt.contourf(X1, X2, TF2, levels=30, label=f'True')
    # plt.xlabel(r"$x_1$")
    # plt.ylabel(r"$x_2$")
    # plt.title(r"ExpSine Function 2D")
    # plt.colorbar()
    # #plt.legend(loc=1)
    # plt.savefig(f"./2dES.pdf", bbox_inches="tight")
    # plt.clf()


    # #contour
    # plt.rcParams['font.size'] = '16'
    # ndir = 150
    # func = MixedSine(ndim=2)
    # xlimits = func.xlimits
    # x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
    # y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)

    # X, Y = np.meshgrid(x, y)
    # Za = np.zeros([ndir, ndir])
    # Zk = np.zeros([ndir, ndir])
    # F  = np.zeros([ndir, ndir])
    # FK  = np.zeros([ndir, ndir])
    # TF = np.zeros([ndir, ndir])

    # for i in range(ndir):
    #     for j in range(ndir):
    #         xi = np.zeros([1,2])
    #         xi[0,0] = x[i]
    #         xi[0,1] = y[j]
    #         TF[j,i] = func(xi)

    # # Plot original function
    # cs = plt.contourf(X, Y, TF, levels = 40)
    # plt.colorbar(cs, aspect=20)
    # plt.xlabel(r"$x_1$")
    # plt.ylabel(r"$x_2$")
    # #plt.legend(loc=1)
    # plt.savefig(f"mixedsine_true.pdf", bbox_inches="tight")

    # plt.clf()

    # import pdb; pdb.set_trace()