from io import BufferedRandom
import numpy as np
import copy

from matplotlib import pyplot as plt
from smt.utils.options_dictionary import OptionsDictionary
from smt.sampling_methods import LHS
from smt.surrogate_models import GEKPLS
from smt.utils.checks import check_support, check_nx, ensure_2d_array
from surrogate.pougrad import POUSurrogate, POUHessian
from scipy.linalg import lstsq, eig
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import Bounds
from scipy.integrate import nquad
from scipy.stats import qmc
from utils.sutils import print_rc_plots, standardization2, linear, quadratic, quadraticSolve, quadraticSolveHOnly, symMatfromVec, maxEigenEstimate, boxIntersect

"""Base Class for Adaptive Sampling Criteria Functions"""
class ASCriteria():
    def __init__(self, model, **kwargs):
        """
        Constructor for a class encompassing a refinement criteria function for adaptive sampling or cross-validation

        Parameters
        ----------
        model : smt SurrogateModel object
            Surrogate model as needed to evaluate the criteria

        **kwargs : named arguments
            Set of options that can be optionally set; each option must have been declared.
        """
        self.name = 'Base'

        # copy the surrogate model object
        self.model = copy.deepcopy(model)

        # get the size of the training set
        kx = 0
        self.dim = self.model.training_points[None][kx][0].shape[1]
        self.ntr = self.model.training_points[None][kx][0].shape[0]

        # if not isinstance(self.model, POUHessian):
        #     trxs = self.model.training_points[None][0][0]
        #     trfs = self.model.training_points[None][0][1]
        #     (
        #         trx,
        #         trf,
        #         X_offset,
        #         y_mean,
        #         X_scale,
        #         y_std,
        #     ) = standardization2(trxs, trfs, self.bounds)

        # else:
        #     trx = self.model.X_norma[0:self.ntr]#model.training_points[None][0][0]
        self.trx = self.model.training_points[None][kx][0]

        self.supports = supports = {}
        supports["obj_derivatives"] = False
        supports["uses_constraints"] = False
        supports["rescaling"] = False

        # set options
        self.options = OptionsDictionary()
        self._init_options()
        self.options.declare("print_iter", True, types=bool)
        self.options.declare(
            "print_rc_plots", 
            False, 
            types=bool,
            desc="Print plots of the RC function if 1D or 2D"
            )
        self.options.declare(
            "print_energy", 
            True, 
            types=bool,
            desc="Compute and print RC energy values upon refinement"
            )
        self.options.declare(
            "return_rescaled", 
            False, 
            types=bool,
            desc="If supported, return RC output scaled back to original function"
            )
        self.options.declare(
            "improve", 
            0, 
            types=int,
            desc="Number of points to generate before retraining"
            )

        self.options.declare(
            "multistart", 
            1, 
            types=int,
            desc="number of optimizations to try per point"
            )
        
        self.options.declare(
            "sub_index", 
            None, 
            desc="indices of dimensions to refine over, mostly useful for pre and post as"
            )


        self.options.update(kwargs)

        self.opt = True
        self.condict = () #for constrained optimization

        # note that evaluate and eval_grad will behave the same no matter what,
        # but this will help with multistart and post processing steps
        dim_t = self.model.training_points[None][0][0].shape[1]
        self.sub_ind = np.arange(0, dim_t).tolist()
        self.fix_ind = []
        self.fix_val = []
        if self.options['sub_index'] is not None:
            self.sub_ind = self.options['sub_index']
            self.fix_ind = [x for x in np.arange(0, dim_t) if x not in self.sub_ind]
            self.fix_val = [0.]*len(self.fix_ind)

        self.initialize(self.model)

    # set static variables if we're only refining over a subspace
    # do nothing otherwise
    def set_static(self, x):
        len_given = x.shape[0]
        dim_t = len(self.sub_ind) + len(self.fix_ind)
        dim_r = len(self.fix_ind)
        if len_given == dim_t:
            self.fix_val = x[self.fix_ind]
        elif len_given == dim_r:
            self.fix_val = x
        else:
            ValueError(f'Invalid number of inputs given ({len_given} != total dim {dim_t}, {len_given} != reduced dim {dim_r})')

        return 


    def pre_asopt(self, bounds, dir=0):
        
        xc, bounds_m = self._pre_asopt(bounds, dir)
        # NOTE: xc not doing anything here

        # ### FD CHECK
        h = 1e-6
        zero = 0.5*np.ones([1,bounds.shape[0]])
        step = 0.5*np.ones([1,bounds.shape[0]])
        step[0,0] += h
        ad = self.eval_grad(zero, bounds, dir)
        fd1 = (self.evaluate(step, bounds, dir) - self.evaluate(zero, bounds, dir))/h
        step = 0.5*np.ones([1,bounds.shape[0]])
        step[0,1] += h
        fd2 = (self.evaluate(step, bounds, dir) - self.evaluate(zero, bounds, dir))/h
        fd = [fd1, fd2]
        import pdb; pdb.set_trace()

        ### Get Reduced Space
        dim_r = len(self.sub_ind)
        bounds_r = bounds[self.sub_ind]

        ### Print Criteria Plots
        if(self.options["print_rc_plots"]):
            print_rc_plots(bounds_r, self.name, self, dir)

        ### Multistart
        sampling = LHS(xlimits=bounds_r, criterion='m')
        ntries = self.options["multistart"]
        if(ntries > 1):
            xc_r = sampling(ntries)
        else: 
            xc_r = np.random.rand(self.dim)*(bounds_r[:,1] - bounds_r[:,0]) + bounds_r[:,0]
            xc_r = np.array([xc_r])

        ### Batches

        ### Return Full Space
        xc = np.zeros([ntries, bounds.shape[0]])
        xc[:,self.sub_ind] = xc_r
        if len(self.sub_ind) != bounds.shape[0]:
            xc[:, self.fix_ind] = self.fix_val

        return xc, bounds_m

    def post_asopt(self, x, bounds, dir=0):

        ### Return Full Space
        xe = np.zeros([bounds.shape[0]])
        xe[self.sub_ind] = x
        if len(self.sub_ind) != bounds.shape[0]:
            xe[self.fix_ind] = self.fix_val

        x = self._post_asopt(xe, bounds, dir)

        self.trx = np.append(self.trx, np.array([x]), axis=0)
        return x

    def evaluate(self, x, bounds, dir=0):

        # _x = ensure_2d_array(x, 'x')
        # import pdb; pdb.set_trace()
        ans = self._evaluate(x, bounds, dir=dir)

        return ans
    
    def eval_grad(self, x, bounds, dir=0):

        # _x = ensure_2d_array(x, 'x')

        ans = self._eval_grad(x, bounds, dir=dir)

        return ans

    def eval_constraint(self, x, bounds, dir=0):

        # _x = ensure_2d_array(x, 'x')

        ans = self._eval_constraint(x, bounds, dir=dir)

        pass

    def eval_constraint_grad(self, x, bounds, dir=0):

        # _x = ensure_2d_array(x, 'x')

        ans = self._eval_constraint_grad(x, bounds, dir=dir)

        pass

    """
    Overwrite
    """
    def _init_options(self):
        pass

    def _pre_asopt(self, bounds, dir=0):
        return None, bounds

    def _post_asopt(self, x, bounds, dir=0):
        return x

    def initialize(self, model=None):
        pass

    def _evaluate(self, x, bounds, dir=0):
        pass

    def _eval_grad(self, x, bounds, dir=0):
        pass

    def _eval_constraint(self, x, bounds, dir=0):
        pass

    def _eval_constraint_grad(self, x, bounds, dir=0):
        pass

    
    def get_energy(self, xlimits, dir=0):
        """
        self determined stopping criteria, e.g. validation error

        Integrate over self.evaluate

        xfix needed if using sub index
        """

        self.pre_asopt(xlimits, dir=dir)

        n = xlimits.shape[0]
        unit_bounds = np.zeros([n,2])
        unit_bounds[:,1] = 1.
        sub_ind = np.arange(0, n).tolist()
        fix_ind = []
        xfix = None
        if self.options['sub_index'] is not None:
            sub_ind = self.options['sub_index']
            m_e = len(sub_ind)
            # grab values for other indices from x_init
            fix_ind = [x for x in np.arange(0, n).tolist() if x not in sub_ind]
            xfix = qmc.scale(np.array([self.fix_val]), xlimits[fix_ind,0], xlimits[fix_ind,1], reverse=True)#[fix_ind]

        def eval_eff(x, bounds, direction):
            x = np.atleast_1d(x)
            x_eff = np.zeros([n])
            x_eff[sub_ind] = x
            x_eff[fix_ind] = xfix

            y = self.evaluate(x_eff, bounds, direction)
            return y

        energy, d0 = nquad(eval_eff, unit_bounds[sub_ind,:], args=(xlimits, dir))

        return -energy

    
"""
A Continuous Leave-One-Out Cross Validation function
"""
class looCV(ASCriteria):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.dminmax = None


    def _init_options(self):
        self.options.declare("approx", False, types=bool)

    def initialize(self, model=None):

        # set up constraints
        self.condict = {
            "type":"ineq",
            "fun":self.eval_constraint,
            "args":[],
        }

        # in case the model gets updated externally by getxnew
        if(model is not None):
            self.model = copy.deepcopy(model)
            kx = 0
            self.dim = self.model.training_points[None][kx][0].shape[1]
            self.ntr = self.model.training_points[None][kx][0].shape[0]

        # Create a list of LOO surrogate models
        self.loosm = []
        for i in range(self.ntr):
            self.loosm.append(copy.deepcopy(self.model))
            self.loosm[i].options.update({"print_global":False})

            kx = 0

            # Give each LOO model its training data, and retrain if not approximating
            trx = self.loosm[i].training_points[None][kx][0]
            trf = self.loosm[i].training_points[None][kx][1]
            trg = []
                
            trx = np.delete(trx, i, 0)
            trf = np.delete(trf, i, 0)
            

            self.loosm[i].set_training_values(trx, trf)
            if(isinstance(self.model, GEKPLS) or isinstance(self.model, POUSurrogate)):
                for j in range(self.dim):
                    trg.append(self.loosm[i].training_points[None][j+1][1]) #TODO:make this optional
                for j in range(self.dim):
                    trg[j] = np.delete(trg[j], i, 0)
                for j in range(self.dim):
                    self.loosm[i].set_training_derivatives(trx, trg[j][:], j)

            if(self.options["approx"] == False):
                self.loosm[i].train()

        # Get the cluster threshold for exploration constraint
        dists = pdist(trx)
        dists = squareform(dists)
        mins = np.zeros(self.ntr)
        for i in range(self.ntr):
            ind = dists[i,:]
            ind = np.argsort(ind)
            mins[i] = dists[i,ind[1]]
        self.dminmax = max(mins)

    #TODO: This could be a variety of possible LOO-averaging functions
    def _evaluate(self, x, dir=0):
        
        if(len(x.shape) != 2):
            x = np.array([x])

        # evaluate the point for the original model
        #import pdb; pdb.set_trace()
        M = self.model.predict_values(x).flatten()

        # now evaluate the point for each LOO model and sum
        y = 0
        for i in range(self.ntr):
            Mm = self.loosm[i].predict_values(x).flatten()
            y += (1/self.ntr)*((M-Mm)**2)
        
        ans = -np.sqrt(y)

        return ans # to work with optimizers

    # if only using local optimization, start the optimizer at the worst LOO point
    def _pre_asopt(self, bounds, dir=0):
        t0 = self.model.training_points[None][0][0]
        #import pdb; pdb.set_trace()
        diff = np.zeros(self.ntr)

        for i in range(self.ntr):
            M = self.model.predict_values(t0[[i]]).flatten()
            Mm = self.loosm[i].predict_values(t0[[i]]).flatten()
            diff[i] = abs(M - Mm)

        ind = np.argmax(diff)

        return np.array([t0[ind]]), None

    def _post_asopt(self, x, bounds, dir=0):

        return x
        

    def _eval_constraint(self, x, dir=0):
        t0 = self.model.training_points[None][0][0]

        con = np.zeros(self.ntr)
        for i in range(self.ntr):
            con[i] = np.linalg.norm(x - t0[i])

        return con - 0.5*self.dminmax


# Hessian estimation and direction criteria

class HessianFit(ASCriteria):
    def __init__(self, model, grad, **kwargs):

        self.bads = None
        self.bad_eigs = None
        self.bad_list = None
        self.bad_nbhd = None
        self.bad_dirs = None
        self.dminmax = None
        self.grad = grad

        super().__init__(model, **kwargs)

        
        
    def _init_options(self):
        #options: neighborhood, surrogate, exact
        self.options.declare("hessian", "neighborhood", types=str)

        #options: honly, full, arnoldi
        self.options.declare("interp", "arnoldi", types=str)

        #options: distance, variance, random
        self.options.declare("criteria", "distance", types=str)

        #options: linear, quadratic
        self.options.declare("error", "linear", types=str)

        self.options.declare("improve", 0, types=int)

        #number of closest points to evaluate nonlinearity measure
        self.options.declare("neval", self.dim*2+1, types=int)

        #perturb the optimal result in a random orthogonal direction
        self.options.declare("perturb", False, types=bool)
        
    def initialize(self, model=None, grad=None):
        
        # set up constraints
        self.condict = {
            "type":"ineq",
            "fun":self.eval_constraint,
            "args":[],
        }


        if(model is not None):
            self.model = copy.deepcopy(model)
            kx = 0
            self.dim = self.model.training_points[None][kx][0].shape[1]
            self.ntr = self.model.training_points[None][kx][0].shape[0]

        if(grad is not None):
            self.grad = grad

        trx = self.model.training_points[None][0][0]
        trf = self.model.training_points[None][0][1]
        trg = np.zeros_like(trx)
        trg = self.grad
        if(isinstance(self.model, GEKPLS)):
            for j in range(self.dim):
                trg[:,j] = self.model.training_points[None][j+1][1].flatten()
        dists = pdist(trx)
        dists = squareform(dists)

        neval = self.options["neval"]
        if(self.options["interp"] == "arnoldi"):
            neval = self.dim




        # 1. Estimate the Hessian (or its principal eigenpair) about each point
        hess = []
        nbhd = []
        indn = []
        pts = []

        # 1a. Determine the neighborhood to fit the Hessian for each point/evaluate the error
        # along with minimum distances
        mins = np.zeros(self.ntr)
        for i in range(self.ntr):
            ind = dists[i,:]
            ind = np.argsort(ind)
            pts.append(np.array(trx[ind,:]))
            indn.append(ind)
            mins[i] = dists[i,ind[1]]
        self.dminmax = max(mins)
        lmax = np.amax(dists)

        if(self.options["hessian"] == "neighborhood"):        
            for i in range(self.ntr):
                if(self.options["interp"] == "full"):
                    fh, gh, Hh = quadraticSolve(trx[i,:], trx[indn[i][1:neval+1],:], \
                                            trf[i], trf[indn[i][1:neval+1]], \
                                            trg[i,:], trg[indn[i][1:neval+1],:])

                if(self.options["interp"] == "honly"):
                    Hh = quadraticSolveHOnly(trx[i,:], trx[indn[i][1:neval+1],:], \
                                            trf[i], trf[indn[i][1:neval+1]], \
                                            trg[i,:], trg[indn[i][1:neval+1],:])
                    fh = trf[i]
                    gh = trg[i,:]

                if(self.options["interp"] == "full" or self.options["interp"] == "honly"):
                    hess.append(np.zeros([self.dim, self.dim]))
                    for j in range(self.dim):
                        for k in range(self.dim):
                            hess[i][j,k] = Hh[symMatfromVec(j,k,self.dim)]
                
                else: #arnoldi
                    evalm, evecm = maxEigenEstimate(trx[i,:], trx[indn[i][1:neval],:], \
                                                    trg[i,:], trg[indn[i][1:neval],:])

                    hess.append([evalm, evecm])

        if(self.options["hessian"] == "surrogate"):
            # 1a. Get the hessian as determined by the surrogate
            # central difference scheme
            h = 1e-5
            for i in range(self.ntr):
                hess.append(np.zeros((self.dim, self.dim)))
            
            for j in range(self.dim):
                xsp = np.copy(trx)
                xsm = np.copy(trx)
                xsp[:,j] += h
                xsm[:,j] -= h

                for k in range(self.dim):
                    hj = np.zeros(self.dim)
                    hj = self.model.predict_derivatives(xsp, k)
                    hj -= self.model.predict_derivatives(xsm, k)
                    for l in range(len(hess)):
                        hess[l][j,k] = hj[l]/h




        # 2. For every point, sum the discrepancies between the linear (quadratic)
        # prediction in the neighborhood and the observation value
        err = np.zeros(self.ntr)

        # using the original neval here
        for i in range(self.ntr):
            #ind = indn[i]
            ind = dists[i,:]
            ind = np.argsort(ind)
            for key in ind[1:self.options["neval"]]:
                if(self.options["error"] == "linear" or self.options["interp"] == "arnoldi"):
                    fh = linear(trx[key], trx[i], trf[i], trg[:][i])
                else:
                    fh = quadratic(trx[key], trx[i], trf[i], trg[:][i], hess[i])
                err[i] += abs(trf[key] - fh)
            err[i] /= self.options["neval"]
            nbhd.append(ind[1:neval])

        emax = max(err)
        for i in range(self.ntr):
            # ADDING A DISTANCE PENALTY TERM
            err[i] /= emax
            err[i] *= 1. - mins[i]/lmax
            err[i] += mins[i]/self.dminmax

        # 2a. Pick some percentage of the "worst" points, and their principal Hessian directions
        badlist = np.argsort(err)
        badlist = badlist[-self.nnew:]
        bads = trx[badlist]
        bad_nbhd = np.zeros([bads.shape[0], self.options["neval"]-1], dtype=int)
        for i in range(bads.shape[0]):
            bad_nbhd[i,:] = nbhd[badlist[i]]





        # 3. Generate a criteria for each bad point

        # 3a. Take the highest eigenvalue/vector of each Hessian
        opt_dir = []
        opt_val = []
        if(self.options["interp"] == "arnoldi"):
            for i in badlist:
                opt_dir.append(hess[i][1])
                opt_val.append(hess[i][0])
        else:
            for i in badlist:
                H = hess[i]
                eigvals, eigvecs = eig(H)
                o = np.argsort(abs(eigvals))
                opt_dir.append(eigvecs[:,o[-1]])
                opt_val.append(eigvals[o[-1]])

        # we have what we need
        self.bad_list = badlist
        self.bad_eigs = opt_val
        self.bads = bads
        self.bad_nbhd = bad_nbhd
        self.bad_dirs = opt_dir
        

    def _evaluate(self, x, dir=0):
        
        xc = self.bads[dir]
        xdir = self.bad_dirs[dir]
        trx = self.model.training_points[None][0][0]
        #trx = self.bads
        m, n = trx.shape

        # x is alpha, the distance from xc along xdir, x(alpha) = xc + alpha*xdir

        # we can use either distance, or kriging variance

        xeval = xc + x*xdir

        if(self.options["criteria"] == "distance"):
            sum = 0
            for i in range(m):
                sum += np.linalg.norm(xeval-trx[i])**2
            
            ans = -sum


        elif(self.options["criteria"] == "variance"):
            
            ans = -self.model.predict_variances(np.array([xeval]))[0,0]

        else:
            print("Invalid Criteria Option")

        return ans 




    def _pre_asopt(self, bounds, dir=0):
        xc = self.bads[dir]
        gc = self.grad[self.bad_list[dir],:]
        eig = self.bad_eigs[dir]
        xdir = self.bad_dirs[dir]
        trx = self.model.training_points[None][0][0]
        nbhd = trx[self.bad_nbhd[dir],:]
        dists = pdist(np.append(np.array([xc]), nbhd, axis=0))
        dists = squareform(dists)
        B = max(np.delete(dists[0,:],[0,0]))
        #B = 0.8*B
        #import pdb; pdb.set_trace()

        # find a cluster threshold (max of minimum distances, Aute 2013)
        mins = np.zeros(dists.shape[0])
        for i in range(dists.shape[0]):
            mins[i] = min(np.delete(dists[i,:], [i,i]))
        S = 0.5*max(mins)
        S = 0.5*mins[0]
        if(self.options["criteria"] == "variance"):
            S = 0.01*max(mins)


        # check if we need to limit further based on bounds
        p0, p1 = boxIntersect(xc, xdir, bounds)
        bp = min(B, p1)
        bm = max(-B, p0)

        # choose the direction to go

        # if concave up, move up the gradient, if concave down, move down the gradient
        work = np.dot(gc, xdir)
        adir = np.sign(work)*np.sign(np.real(eig))
        if(adir > 0):
            bm = min(adir*S, p1)-0.01#p0)
            bp = bm+0.01
        else:
            bp = max(adir*S, p0)+0.01#p1)
            bm = bp-0.01

        #import pdb; pdb.set_trace()
        return np.array([adir*S]), Bounds(bm, bp)




    def _post_asopt(self, x, bounds, dir=0):

        # transform back to regular coordinates

        xc = self.bads[dir]
        xdir = self.bad_dirs[dir]
        eig = self.bad_eigs[dir]

        xeval = xc + x*xdir

        # generate random vector and orthogonalize, if we want to perturb
        if(self.options["perturb"]):
            trx = self.model.training_points[None][0][0]
            nbhd = trx[self.bad_nbhd[dir],:]
            dists = pdist(np.append(np.array([xc]), nbhd, axis=0))
            dists = squareform(dists)
            B = max(np.delete(dists[0,:],[0,0]))
            xrand = np.random.randn(self.dim)
            xrand -= xrand.dot(xdir)*xdir/np.linalg.norm(xdir)**2
            
            p0, p1 = boxIntersect(xeval, xrand, bounds)

            bp = min(B, p1)
            bm = max(-B, p0)
            alpha = np.random.rand(1)*(bp-bm) + bm
            xeval += alpha*xrand

        return xeval

    def _eval_constraint(self, x, dir=0):
        xc = self.bads[dir]
        xdir = self.bad_dirs[dir]
        trx = self.model.training_points[None][0][0]
        nbhd = trx[self.bad_nbhd[dir],:]
        m, n = nbhd.shape

        xeval = xc + x*xdir

        con = np.zeros(m)
        for i in range(m):
            con[i] = np.linalg.norm(xeval - nbhd[i])

        return con - 0.5*self.dminmax












# TEAD Method with exact gradients

class TEAD(ASCriteria):
    def __init__(self, model, grad, bounds, **kwargs):

        self.cand = None
        self.dminmax = None
        self.lmax = None
        self.grad = grad
        self.bounds = bounds
        self.bad_list = None
        self.bads = None
        self.bad_nbhd = None

        super().__init__(model, **kwargs)

        self.opt = False #no optimization performed for this
        
    def _init_options(self):
        #number of candidates to consider
        self.options.declare("ncand", self.dim*50, types=int)

        #source of gradient
        self.options.declare("gradexact", False, types=bool)

        #number of points to pick
        self.options.declare("improve", 0, types=int)
        
        #number of closest points to evaluate nonlinearity measure
        self.options.declare("neval", 1, types=int)

    def initialize(self, model=None, grad=None):
        
        if(model is not None):
            self.model = copy.deepcopy(model)
            kx = 0
            self.dim = self.model.training_points[None][kx][0].shape[1]
            self.ntr = self.model.training_points[None][kx][0].shape[0]

        if(grad is not None):
            self.grad = grad

        trx = self.model.training_points[None][0][0]
        trf = self.model.training_points[None][0][1]
        trg = np.zeros_like(trx)
        if(self.options["gradexact"]):
            trg = self.grad
            if(isinstance(self.model, GEKPLS)):
                for j in range(self.dim):
                    trg[:,j] = self.model.training_points[None][j+1][1].flatten()
        else:
            for j in range(self.dim):
                trg[:,j] = self.model.predict_derivatives(trx, j)[:,0]
        

        # 1. Generate candidate points, determine reference distances and neighborhoods
        sampling = LHS(xlimits=self.bounds, criterion='m')
        ncand = self.options["ncand"]
        self.cand = sampling(ncand)
        dists = cdist(self.cand, trx)

        mins = np.zeros(ncand)
        nbhd = np.zeros([ncand, self.options["neval"]], dtype=int)
        for i in range(ncand):
            ind = dists[i,:]
            ind = np.argsort(ind)
            mins[i] = dists[i,ind[0]]
            nbhd[i] = ind[0:self.options["neval"]]
        self.dminmax = max(mins)
        self.lmax = np.amax(dists)

        # 2. For every candidate point, sum the discrepancies between the linear (quadratic)
        # prediction in the neighborhood and the surrogate value at the candidate point
        lerr = np.zeros(ncand)
        err = np.zeros(ncand)
        for i in range(ncand):
            for key in nbhd[i,0:self.options["neval"]]:
                fh = linear(self.cand[i], trx[key], trf[key], trg[:][key])
                lerr[i] += abs(model.predict_values(self.cand[[i],:]) - fh)

            lerr[i] /= self.options["neval"]

        emax = max(lerr)
        for i in range(ncand):
            # ADDING A DISTANCE PENALTY TERM
            lerr[i] /= emax
            w = 1. - mins[i]/self.lmax
            err[i] += mins[i]/self.dminmax + w*lerr[i]

        # 2a. Pick some percentage of the "worst" points
        badlist = np.argsort(err)
        badlist = badlist[-self.nnew:]
        bads = self.cand[badlist]
        bad_nbhd = np.zeros([bads.shape[0], self.options["neval"]], dtype=int)
        for i in range(bads.shape[0]):
            bad_nbhd[i,:] = nbhd[badlist[i]]

        # we have what we need
        self.bad_list = badlist
        self.bads = bads
        self.bad_nbhd = bad_nbhd

    def _post_asopt(self, x, bounds, dir=0):

        return self.bads[dir]




    