import numpy as np
# import csv
import os
from scipy.optimize import leastsq
from smt.problems.problem import Problem
from scipy.spatial.distance import cdist
# FROM WINSLOW ET AL., Turns2D CFD predictions for NACA0012 airfoil, empirical model
class NacaAOALiftModel(Problem):
    def _initialize(self):
        self.options.declare("name", "NacaAOALiftModel", types=str)
        self.options.declare("ndim", 2, types=int) # by default, 2 inputs, but fix_RE reduces this
        self.options.declare("fix_RE", 3e5, types=(type(None), float)) # if none, treat as uncertain param
        
    def _setup(self):
        # customize uncertainty bounds after setup
        # matching rosenbrock domain for now for eval loc
        # AOA
        self.xlimits[0, 0] = 0.
        self.xlimits[0, 1] = 20.
        # AOA perturn
        self.xlimits[1, 0] = -3.
        self.xlimits[1, 1] = 3.
        # RE
        if self.options["ndim"] == 3:
            self.xlimits[2, 0] = 1e5
            self.xlimits[2, 1] = 1e6

        var_list = [0, 1]
        if self.options["fix_RE"] is None:
            var_list.append(2)


        # correct it
        self.options["ndim"] = len(var_list)
        self.xlimits = self.xlimits[var_list, :]
        self.var_list = var_list

        filename3 = os.path.join(os.path.dirname(__file__), 're3data.csv')
        self.AOA3Data = np.append(np.zeros([1,2]), np.genfromtxt(filename3, delimiter=','), axis=0)
        self.bx31 = 11.2; self.bx32 = 14.8
        # self.AOALData = np.append(np.zeros([1,2]), np.genfromtxt('reldata.csv', delimiter=','), axis=0)
        # self.bxl1 = 9.; self.bxl2 = 13.
        # self.AOAHData = np.append(np.zeros([1,2]), np.genfromtxt('rehdata.csv', delimiter=','), axis=0)
        # self.bxh1 = 14.; self.bxh2 = 17.

        self.xc = np.array([(self.bx31 - self.xlimits[0,0])*0.9, (self.bx32 - self.bx31)/2. + self.bx31, (self.xlimits[0,1] - self.bx32 )/4. + self.bx32])
        self.rho = 4.

        bin1 = np.argwhere(self.AOA3Data[:,0] < self.bx31).flatten()
        bin2 = np.argwhere(np.logical_and(self.AOA3Data[:,0] <= self.bx32, self.AOA3Data[:,0] >= self.bx31)).flatten()
        bin3 = np.argwhere(self.AOA3Data[:,0] > self.bx32).flatten()

        func31 = lambda p31 : self._fitter0(self.AOA3Data[bin1,0],  p31) - self.AOA3Data[bin1,1]
        func32 = lambda p32 : self._fitter1(self.AOA3Data[bin2,0],  p32) - self.AOA3Data[bin2,1]
        func33 = lambda p33 : self._fitter2(self.AOA3Data[bin3,0],  p33) - self.AOA3Data[bin3,1]

        # breakpoint()
        init1 = np.ones(3)
        init2 = np.ones(4)
        init3 = np.ones(3)
        init2[2] *= 13
        init2[3] *= 0.8
        self.param31 = leastsq(func31, init1)[0]
        self.param32 = leastsq(func32, init2)[0]
        self.param33 = leastsq(func33, init3)[0]


        # import pdb; pdb.set_trace()

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

        # get inputs
        # ex = x[:,0]

        # add perturbations
        ex = x[:,0] + x[:,1]

        if self.options["fix_RE"] is None:
            ey = x[:,self.var_list.index(2)]

        
        y = np.zeros((ne, 1), complex)
        mindist = np.min(cdist(np.atleast_2d(ex).T, np.atleast_2d(self.xc).T), axis=1)

        # breakpoint()
        y0 = self._fitter0(ex, self.param31)
        y1 = self._fitter1(ex, self.param32)
        y2 = self._fitter2(ex, self.param33)
        
        rho = self.rho
        wx0 = ex - self.xc[0]
        wx1 = ex - self.xc[1]
        wx2 = ex - self.xc[2]
        d0 = np.sqrt(wx0**2 + 1e-10)
        d1 = np.sqrt(wx1**2 + 1e-10)
        d2 = np.sqrt(wx2**2 + 1e-10)
        expfac0 = np.exp(-rho*(d0.flatten() - mindist))
        expfac1 = np.exp(-rho*(d1.flatten() - mindist))
        expfac2 = np.exp(-rho*(d2.flatten() - mindist))
        denom = expfac0 + expfac1 + expfac2
        numer = y0.flatten()*expfac0 + y1.flatten()*expfac1 + y2.flatten()*expfac2

        if kx == None:
            
            y[:,0] = numer/denom

            # breakpoint()


        if kx is not None:
            g0 = self._fitter0grad(ex, self.param31)
            g1 = self._fitter1grad(ex, self.param32)
            g2 = self._fitter2grad(ex, self.param33)

            dd0 = wx0/d0
            dd1 = wx1/d1
            dd2 = wx2/d2

            dexpfac0 = -rho*expfac0*dd0
            dexpfac1 = -rho*expfac1*dd1
            dexpfac2 = -rho*expfac2*dd2

            dnumer = y0.flatten()*dexpfac0 + y1.flatten()*dexpfac1 + y2.flatten()*dexpfac2
            dnumer += g0.flatten()*expfac0 + g1.flatten()*expfac1 + g2.flatten()*expfac2

            ddenom = dexpfac0 + dexpfac1 + dexpfac2

            y[:,0] = dnumer/denom - numer*ddenom/(denom*denom)

            # ex = x + dx
            if kx == 0: #nominal

                y = y
            if kx == 1: #perturbation
                y = y

        return -y #maximize

    # three curve fits
    def _fitter0(self, x, param):
        
        # by1, by2, 
        a1 = param[0]
        b1 = param[1]
        c1 = param[2]

        # y = np.zeros_like(x)

        
        # quad 1
        # y[bin0] = a1*x[bin0]*x[bin0] + b1*x[bin0] + c1
        y = a1*x*x + b1*x + c1
        return y
    
    def _fitter1(self, x, param):
        ka = param[0]
        wa = param[1]
        xa = param[2]
        ya = param[3]
        # ra = param[4]

        
        work = x - xa

        y = ka*np.arctan(wa*work) + ya #*np.exp(-ra*work**2)

        return y

    def _fitter2(self, x, param):
        a2 = param[0]
        b2 = param[1]
        c2 = param[2]
    
        
    
        y = a2*x*x + b2*x + c2

        return y

    def _fitter0grad(self, x, param):
        # by1, by2, 
        a1 = param[0]
        b1 = param[1]
        c1 = param[2]

        y = 2*a1*x + b1
        
        return y
    def _fitter1grad(self, x, param):
        ka = param[0]
        wa = param[1]
        xa = param[2]
        ya = param[3]
        # ra = param[4]
        work = x - xa

        y = ka*wa/(1.+ work*work*wa*wa) #np.arctan(work) 
        
        return y
    def _fitter2grad(self, x, param):
        # by1, by2, 
        a2 = param[0]
        b2 = param[1]
        c2 = param[2]
    
        y = 2*a2*x + b2
        
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
    from matplotlib import colors
    from smt.problems import Rosenbrock
    from smt.sampling_methods import LHS
    from utils.sutils import convert_to_smt_grads
    from scipy.stats import norm

    plt.rcParams['font.size'] = '15'

    ndir =150
    nu = 5000

    
    # ## ShortCol1U
    # func = ShortColumn1U(ndim=3)
    # xlimits = func.xlimits

    ## UncertainEllipse
    # eval only
    func = NacaAOALiftModel()
    func = NacaAOALiftModel()
    # r = func.options["fix_radius"]
    # r = func.options["fix_radius"]
    # r = func.options["fix_radius"]

    if func.options["ndim"] == 2:
        xlimits = func.xlimits
        x0 = np.atleast_2d(np.linspace(xlimits[0][0], xlimits[0][1], ndir) ).T
        xp = np.atleast_2d(np.linspace(xlimits[1][0], xlimits[1][1], ndir) ).T
        x1 = np.append(x0, np.zeros_like(x0), axis=1)


        # x2 = np.append(x0.T, np.ones_like(x0).T, axis=1)
        # breakpoint()
        y1 = func(x1)
        y2 = np.zeros_like(y1)
        for i in range(ndir):
            xip = np.append(x0[i]*np.ones([ndir,1]), xp,  axis=1)
            y2[i] = np.mean(func(xip)) 

        # y2 = func(x2)
        # plt.scatter(func.AOA3Data[:,0], func.AOA3Data[:,1], marker='x', color='r')
        plt.plot(x1[:,0], y1)
        plt.plot(x1[:,0], y2)
        plt.axvline(func.bx31, linestyle='--', color='k')
        plt.axvline(func.bx32, linestyle='--', color='k')

        plt.savefig("aoaplot.png", dpi=1000, bbox_inches='tight')
        plt.clf()
        # full contour w/ perturb
        X, Y = np.meshgrid(x0, xp)
        FT = np.zeros([ndir,ndir])
        for i in range(ndir):
            for j in range(ndir):
                xi = np.zeros([1,2])
                xi[0,0] = x0[i]
                xi[0,1] = xp[j]
                FT[i,j] = func(xi)

        plt.axvline(func.bx31, linestyle='--', color='k')
        plt.axvline(func.bx32, linestyle='--', color='k')
        cs0 = plt.contourf( X.T, Y.T, FT, levels = 25)
        cbar = plt.colorbar(cs0, label=r'$f(x_d, x_u)$')
        plt.savefig("aoacontour.png", dpi=1000, bbox_inches='tight')
        plt.clf()
        breakpoint()
        
        print(func.param31)
        print(func.param32)
        print(func.param33)

    if func.options["ndim"] == 3:
        xlimits = func.xlimits
        x1 = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
        x2 = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
        samp = LHS(xlimits=xlimits[0:1], criterion='maximin')
        # area = xlimits[0][1] - xlimits[0][0]
        # x0 = samp(nu)
        # pdf = norm(500, 100)
        # weight = pdf.pdf(x0)
        X1, X2 = np.meshgrid(x1, x2)

        TF = np.zeros([ndir, ndir])
        for i in range(len(x1)):
            for j in range(len(x2)):
                xi = np.zeros([nu,2])
                xi[:,0] = x1[i]
                xi[:,1] = x2[j]
                # xi[:,0] = x0[:,0]
                eval = func(xi)
                # weval = np.dot(eval.T, weight)*area
                # TF[i,j] = weval/nu
                TF[i,j] = eval[0]

        # # Plot the target function
        vmin = np.min(TF) 
        vmax = np.max(TF)
        plt.contourf(X1, X2, TF.T, levels=25, label=f'True', cmap='RdBu_r',
                      norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
        # plt.xlabel(r"$x_{d,1}$")
        # plt.ylabel(r"$x_{d,2}$")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        # vmin = np.min(TF) 
        # vmax = np.max(TF)
        # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        plt.colorbar(label=r"$F(x,y)$")
        # # import pdb; pdb.set_trace()
        # # minindx = np.argmin(TF.T, keepdims=True)
        # minind = np.where(TF.T == np.min(TF.T))
        # plt.scatter(x1[minind[1]][0], x2[minind[0]][0], color='r')
        # plt.scatter(9., 15., color='b')
        #plt.legend(loc=1)
        plt.savefig(f"./2dellipse.png", bbox_inches="tight")
        plt.clf()
        import pdb; pdb.set_trace()