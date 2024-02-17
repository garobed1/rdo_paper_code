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
from utils.stat_comps import _mu_sigma_comp, _mu_sigma_grad
from utils.error import _gen_var_lists
from utils.sutils import convert_to_smt_grads, print_rc_plots, standardization2, linear, quadratic, quadraticSolve, quadraticSolveHOnly, symMatfromVec, maxEigenEstimate, boxIntersect

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
        self.options.declare(
            "pdf_weight",
            None,
            desc="pdf list to weight criteria by"
        )
        self.options.declare(
            "eta_weight",
            None,
            desc="for energy calc, account for mean+stdev implementation"
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

        # pdfs
        self.pdfs = [1.]*dim_t
        u_b = np.zeros([dim_t, 2])
        u_b[:,1] = 1.0
        if self.options['pdf_weight'] is not None:
            pdf_list, uncert_list, static_list, scales, pdf_name_list = _gen_var_lists(self.options['pdf_weight'], u_b)
            for i in range(len(pdf_list)):
                if not isinstance(pdf_list[i], float):
                    self.pdfs[i] = pdf_list[i]
        self.pdf_name_list = pdf_name_list

        # flag for speeding up energy calculations if possible
        self.energy_mode = False
        self.D_cache = None

        dim_u = len(self.sub_ind)
        xlimits_u = np.zeros([dim_u,2])
        xlimits_u[:,1] = 1.
        samp = LHS(xlimits = xlimits_u)
        e_x = None
        if rank == 0:
            e_x = samp(5000*dim_u)
        self.e_x = comm.bcast(e_x)

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

        ### FD CHECK
        # if dir == 1:
        #     h = 1e-6
        #     zero = 0.5*np.ones([1,bounds.shape[0]])
        #     zero[0,0] = self.trx[-1,0]-0.01
        #     zero[0,1] = 0.65
        #     step = 0.5*np.ones([1,bounds.shape[0]])
        #     step[0,0] = self.trx[-1,0]-0.01
        #     step[0,1] = 0.65
        #     step[0,0] += h
        #     ad = self.eval_grad(zero, bounds, dir)
        #     fd1 = (self.evaluate(step, bounds, dir) - self.evaluate(zero, bounds, dir))/h
        #     step = 0.5*np.ones([1,bounds.shape[0]])
        #     step[0,0] = self.trx[-1,0]-0.01
        #     step[0,1] = 0.65
        #     step[0,1] += h
        #     fd2 = (self.evaluate(step, bounds, dir) - self.evaluate(zero, bounds, dir))/h
        #     fd = [fd1, fd2]
        #     import pdb; pdb.set_trace()

        ### Get Reduced Space
        dim_r = len(self.sub_ind)
        bounds_r = bounds[self.sub_ind]

        ### Print Criteria Plots
        if(self.options["print_rc_plots"]):
            print_rc_plots(bounds_r, self.name, self, dir)

        ### Multistart
        sampling = LHS(xlimits=bounds_r, criterion='m')
        ntries = self.options["multistart"]
        
        xc_r = None
        if rank == 0:
            if(ntries > 1):
                xc_r = sampling(ntries)
            else: 
                xc_r = np.random.rand(dim_r)*(bounds_r[:,1] - bounds_r[:,0]) + bounds_r[:,0]
                xc_r = np.array([xc_r])
        xc_r = comm.bcast(xc_r)

        ### Batches

        ### Return Full Space
        xc = np.zeros([ntries, bounds.shape[0]])
        xc[:,self.sub_ind] = xc_r
        if len(self.sub_ind) != bounds.shape[0]:
            xc[:, self.fix_ind] = self.fix_val

        return xc, bounds_m

    def post_asopt(self, x, bounds, dir=0):

        ### Return Full Space
        xe = np.zeros([x.shape[0], bounds.shape[0]])
        xe[:, self.sub_ind] = x
        if len(self.sub_ind) != bounds.shape[0]:
            xe[:,self.fix_ind] = self.fix_val

        x = self._post_asopt(xe, bounds, dir)

        self.trx = np.append(self.trx, x, axis=0)
        return x

    def evaluate(self, x, bounds, dir=0):

        # _x = ensure_2d_array(x, 'x')
        ans = self._evaluate(x, bounds, dir=dir)

        # apply pdf weightings if present
        weight = np.ones_like(ans)
        xw = np.atleast_2d(x)
        area = 1.0
        for j in range(xw.shape[1]):
            if isinstance(self.pdfs[j], float) or j not in self.sub_ind:
                weight *= 1.0
            else:

                try:
                    if self.pdf_name_list[j] == 'uniform' or self.pdf_name_list[j] == 'beta':
                        weight *= self.pdfs[j].pdf(xw[:,j:j+1])[:,0]
                    else:
                        weight *= self.pdfs[j].pdf(qmc.scale(xw[:,j:j+1], bounds[j,0], bounds[j,1]))[:,0]
                except:
                    import pdb; pdb.set_trace()
                area *= bounds[j,1] - bounds[j,0]
        ans_w = ans*weight*area
        return ans_w
    
    def eval_grad(self, x, bounds, dir=0):

        # _x = ensure_2d_array(x, 'x')

        ans = self._eval_grad(x, bounds, dir=dir)
        ans_w = ans
        if self.options["pdf_weight"]:
            h = 1e-8
            ans_f = self._evaluate(x, bounds, dir=dir)
            weight = np.ones_like(ans_f)
            xw = np.atleast_2d(x)
            dweight = np.zeros_like(xw)
            area = 1.0
            for j in range(xw.shape[1]):
                if isinstance(self.pdfs[j], float) or j not in self.sub_ind:
                    weight *= 1.0
                    dweight *= 1.0
                else:
                    weight *= self.pdfs[j].pdf(qmc.scale(xw[:,j:j+1], bounds[j,0], bounds[j,1]))[:,0]
                    area *= bounds[j,1] - bounds[j,0]
                    for k in range(xw.shape[1]):
                        if isinstance(self.pdfs[k], float):
                            dweight[:,j] *= self.pdfs[k]
                        else:
                            if j != k:
                                dweight[:,j] *= self.pdfs[j].pdf(qmc.scale(xw[:,j:j+1], bounds[j,0], bounds[j,1]))[:,0]
                            else:
                                fac = 1.0
                                step = xw[:,j:j+1] + h
                                if step > 1.0: # just in case
                                    fac = -1.0
                                    step = xw[:,j:j+1] - h
                                dweight[:,j] *= fac*(self.pdfs[j].pdf(qmc.scale(step, bounds[j,0], bounds[j,1]))[:,0] -
                                            self.pdfs[j].pdf(qmc.scale(xw[:,j:j+1], bounds[j,0], bounds[j,1]))[:,0])/h
        
            ans_w = np.einsum('i,ij->ij', ans_f, np.atleast_2d(dweight)) + np.einsum('i,ij->ij', weight, np.atleast_2d(ans))
            ans_w *= area
        return ans_w

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
            x = np.atleast_2d(x)
            x_eff = np.zeros([x.shape[0], n])
            x_eff[:,sub_ind] = x
            x_eff[:,fix_ind] = xfix

            y = self.evaluate(x_eff, bounds, direction)
            
            return y

        self.energy_mode = True
        # print(f"PAST EN PREP {rank}", flush = True)
        # energy, d0 = nquad(eval_eff, unit_bounds[sub_ind,:], args=(xlimits, dir))
        res = eval_eff(self.e_x, xlimits, dir)

        # account for mean plus stdev
        if self.options["eta_weight"] is not None:
            # get original crit
            scrit = HessianRefine(self.model, convert_to_smt_grads(self.model), xlimits, sub_index=sub_ind, 
                                  pdf_weight=self.options["pdf_weight"], neval=self.options['neval'], rho=self.rho, 
                                  rscale=self.options['rscale'],  scale_by_volume=False, 
                                  return_rescaled=True, min_contribution=1e-14, 
                                  print_rc_plots=False)

            def eval_eff2(x, bounds, direction):
                x = np.atleast_2d(x)
                x_eff = np.zeros([x.shape[0], n])
                x_eff[:,sub_ind] = x
                x_eff[:,fix_ind] = xfix

                y = scrit.evaluate(x_eff, bounds, direction)

                return y

            eta = self.options["eta_weight"]
            mpart = np.sum(res, axis=0)/self.e_x.shape[0]
            # Wu = 
            x_eff = np.zeros([self.e_x.shape[0], n])
            x_eff[:,sub_ind] = self.e_x
            x_eff[:,fix_ind] = xfix
            exs = qmc.scale(x_eff, xlimits[:,0], xlimits[:,1])
            pdf_list, uncert_list, static_list, scales, pdf_name_list = _gen_var_lists(self.options['pdf_weight'], xlimits)
            stats, vals = _mu_sigma_comp(self.model.predict_values, exs.shape[0], exs, xlimits, scales[sub_ind], pdf_list, tf = None, weights=None)
            gstats, gvals = _mu_sigma_grad(self.model.predict_derivatives, exs.shape[0], exs, xlimits, scales[sub_ind], fix_ind, pdf_list, tf = vals, weights=None)
            mn = stats[0]
            dmn = gstats[0]
            Wu = vals - mn
            work = gvals[:,fix_ind] -dmn
            dWu = np.linalg.norm(work, axis= 1)*np.sign(work).flatten()

            res2 = eval_eff2(self.e_x, xlimits, dir)
            spart = abs(np.dot(res, dWu)/self.e_x.shape[0] + np.dot(res2, Wu)/self.e_x.shape[0])

            energy = eta*mpart - (1.-eta)*spart
            # breakpoint()
        # print(f"PAST EN EVAL {rank}", flush = True)
        else:
            term = np.sum(res, axis=0)/self.e_x.shape[0]
            if isinstance(term, np.ndarray):
                energy = -np.linalg.norm(term[sub_ind])
            else:
                energy = term
        
        # multiply by volume ?
        # vol = 1
        # for i in sub_ind:
        #     vol *= (xlimits[i,1] - xlimits[i,0])
        # energy *= vol

        self.energy_mode = False
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




    












"""
BAD WORKAROUND INCOMING




"""







from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist, squareform
# from scipy.optimize import Bounds
from utils.sutils import divide_cases, innerMatrixProduct, quadraticSolveHOnly, symMatfromVec, estimate_pou_volume,  standardization2, gen_dist_func_nb



class HessianRefine(ASCriteria):
    def __init__(self, model, grad, bounds, **kwargs):


        self.grad = grad
        self.bounds = bounds
        self.Mc = None

        super().__init__(model, **kwargs)
        self.name = 'POUHESS'

        self.scaler = 0

        self.supports["obj_derivatives"] = True  
        
    def _init_options(self):
        declare = self.options.declare

        declare(
            "rscale", 
            0.5, 
            types=float,
            desc="scaling for error model hyperparameter"
        )

        declare(
            "rho",
            None,
            desc="Distance scaling parameter"
        )
        declare(
            "neval", 
            3, 
            types=int,
            desc="number of closest points to evaluate hessian estimate"
        )

        declare(
            "scale_by_cond", 
            False, 
            types=bool,
            desc="scale criteria in a cell by the condition number of the hess approx matrix"
        )
        declare(
            "scale_by_volume", 
            True, 
            types=bool,
            desc="scale criteria in a cell by the approximate volume of the cell"
        )
        declare(
            "out_of_bounds", 
            0, 
            types=(int, float),
            desc="allow optimizer to go out of bounds, then snap inside if it goes there"
        )
        declare(
            "min_contribution",
            1e-13,
            types=(float),
            desc="If not 0.0, use to determine a ball-point radius for which POU numerator contributes greater than it. Then, use a KDTree to query points in that ball during evaluation"
        )

        declare(
            "min_points",
            # size-1,
            9,
            # 4,
            types=int,
            desc="Minimum number of points to compute distances to if min_contribution is active"
        )


    def initialize(self, model=None, grad=None):
        
        # set up constraints
        # self.condict = {
        #     "type":"ineq",
        #     "fun":self.eval_constraint,
        #     "args":[],
        # }
        self.supports["rescaling"] = True

        if(model is not None):
            self.model = copy.deepcopy(model)
            kx = 0
            self.dim = self.model.training_points[None][kx][0].shape[1]
            self.ntr = self.model.training_points[None][kx][0].shape[0]

        if(grad is not None):
            self.grad = grad

        #NOTE: Slicing here because GEKPLS appends grad approx
        if not isinstance(self.model, POUHessian):
            trxs = self.model.training_points[None][0][0]
            trfs = self.model.training_points[None][0][1]
            trg = np.zeros_like(trxs)
            (
                trx,
                trf,
                X_offset,
                y_mean,
                X_scale,
                y_std,
            ) = standardization2(trxs, trfs, self.bounds)

            trg = self.grad*(X_scale/y_std)

            self.x_off = X_offset
            self.x_sca = X_scale
            self.y_off = y_mean
            self.y_sca = y_std

        else:
            trx = self.model.X_norma[0:self.ntr]#model.training_points[None][0][0]
            trf = self.model.y_norma[0:self.ntr]#training_points[None][0][1]
            trg = np.zeros_like(trx)
            # if(isinstance(self.model, GEKPLS)):
            #     for j in range(self.dim):
            #         trg[:,j] = self.model.g_norma[:,j].flatten()
            # else:
            trg = self.grad*(self.model.X_scale/self.model.y_std)

            self.x_off = self.model.X_offset
            self.x_sca = self.model.X_scale
            self.y_off = self.model.y_mean
            self.y_sca = self.model.y_std

        # Determine rho for the error model
        if self.options["rho"] is not None:
            self.rho = self.options["rho"]
        else:
            self.rho = self.options['rscale']*pow(self.ntr, 1./self.dim)
        self.rho2 = 5000. # rho to use for energy calc

        # Generate kd tree for nearest neighbors lookup
        self.tree = KDTree(trx)

        # Check if the trained surrogate model has hessian data
        try:
            self.H = model.h
            self.Mc = model.Mc
        except:
            indn = []
            nstencil = self.options["neval"]
            for i in range(self.ntr):
                dists, ind = self.tree.query(trx[i], nstencil)
                indn.append(ind)
            hess = []
            mcs = np.zeros(self.ntr)
            for i in range(self.ntr):
                Hh, mc = quadraticSolveHOnly(trx[i,:], trx[indn[i][1:nstencil],:], \
                                         trf[i], trf[indn[i][1:nstencil]], \
                                         trg[i,:], trg[indn[i][1:nstencil],:], return_cond=True)

                hess.append(np.zeros([self.dim, self.dim]))
                mcs[i] = mc
                for j in range(self.dim):
                    for k in range(self.dim):
                        hess[i][j,k] = Hh[symMatfromVec(j,k,self.dim)]

            # self.h = hess
            # self.Mc = mcs
            self.H = comm.allreduce(hess)
            self.Mc = comm.allreduce(mcs)

        m, n = trx.shape

        # factor in cell volume
        fakebounds = copy.deepcopy(self.bounds)
        fakebounds[:,0] = 0.
        fakebounds[:,1] = 1.
        if self.options["scale_by_volume"]:
            self.dV = estimate_pou_volume(trx, fakebounds)
        else:
            self.dV = np.ones(trx.shape[0])

    # Assumption is that the quadratic terms are the error
    def _evaluate(self, x, bounds, dir=0):
        X_cont = np.atleast_2d(x)
        cap = self.options["min_contribution"]
        cmin = self.options["min_points"]
        numeval = X_cont.shape[0]
        cases = divide_cases(numeval, size)
        try:
            delta = self.model.options["delta"]
        except:
            delta = 1e-10

        Mc = np.ones(self.ntr)
        if self.options["scale_by_cond"]:
            Mc = self.Mc

        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)

        # exhaustive search for closest sample point, for regularization
        # import pdb; pdb.set_trace()
        # D = cdist(np.array([x]), trx)
        # distf = gen_dist_func_nb(parallel=True)
        distf = cdist
        # if self.energy_mode and  self.D_cache is None:
        #     diff = trx.shape[0] - self.D_cache.shape[1]

        #     if diff > 0:
        #         self.D_cache = np.hstack([self.D_cache, distf(X_cont[cases[rank],:], trx[-diff:,:])])
        #     # import pdb; pdb.set_trace()

        #     D = self.D_cache

        # else:
        D = distf(X_cont[cases[rank],:], trx)

        # neighbors
        neighbors_all, ball_rad = self.neighbors_func(X_cont, self.rho, cap, cmin, self.ntr, cases)

        mindist_p = np.min(D, axis=1)
        mindist = np.zeros(numeval)
        c = 0
        for k in cases[rank]:
            mindist[k] = mindist_p[c]
            c += 1

        mindist = comm.allreduce(mindist)

        D = None
        fac_all = self.dV*Mc
        y_ = np.zeros(numeval)
        if self.energy_mode:
            # y_ = np.zeros([numeval, X_cont.shape[1]])#self.higher_terms(X_cont[0,:] - trx, None, self.H).shape[1]])
            c = 0
            print(f"PAST EN EVAL PREP {rank}", flush = True)
            for k in cases[rank]:
            # for k in range(numeval):
                
                neighbors = neighbors_all
                if ball_rad:
                    # neighbors = neighbors_all[k]
                    neighbors = neighbors_all[c]
                    xc = trx[neighbors,:]
                fac = fac_all[neighbors]

                # work = X_cont[k,:] - trx[:self.ntr,:]
                # work = X_cont[k,:] - xc
                # # dist = np.sqrt(D[k,:]**2 + delta)#np.sqrt(D[0][i] + delta)
                # dist = np.sqrt(np.einsum('ij,ij->i',work,work) + delta)#np.sqrt(D[0][i] + delta)
                # # local = np.einsum('ij,i->ij', self.higher_terms(work, None, self.H), fac) # NEWNEWNEW
                # local = np.einsum('ij,i->ij', self.higher_terms(work, None, self.H[neighbors]), fac) # NEWNEWNEW
                # expfac = np.exp(-self.rho2*(dist-mindist[k]))
                # numer = np.einsum('ij,i->j', local, expfac)
                # denom = np.sum(expfac)

                # work = X_cont[k,:] - trx[:self.ntr,:]
                work = X_cont[k,:] - xc
                # dist = np.sqrt(D[k,:]**2 + delta)#np.sqrt(D[0][i] + delta)
                dist = np.sqrt(np.einsum('ij,ij->i',work,work) + delta)
                # local = self.higher_terms(work, None, self.H)*fac # NEWNEWNEW
                local = self.higher_terms(work, None, self.H[neighbors])*fac # NEWNEWNEW
                expfac = np.exp(-self.rho2*(dist-mindist[k]))
                numer = np.dot(local, expfac)
                denom = np.sum(expfac)
        
                y_[k] = numer/denom
                c += 1
            
            print(f"PAST EN EVAL LOOP {rank}", flush = True)
        else: 
            c = 0
            for k in cases[rank]:
            # for k in range(numeval):
            # for k in prange(numeval):
                neighbors = neighbors_all
                if ball_rad:
                    # neighbors = neighbors_all[k]
                    neighbors = neighbors_all[c]
                    # try:
                    xc = trx[neighbors,:]
                    # except:
                    #     import pdb; pdb.set_trace()
                fac = fac_all[neighbors]

                # work = X_cont[k,:] - trx[:self.ntr,:]
                work = X_cont[k,:] - xc
                # dist = np.sqrt(D[k,:]**2 + delta)#np.sqrt(D[0][i] + delta)
                dist = np.sqrt(np.einsum('ij,ij->i',work,work) + delta)
                # local = self.higher_terms(work, None, self.H)*fac # NEWNEWNEW
                local = self.higher_terms(work, None, self.H[neighbors])*fac # NEWNEWNEW
                expfac = np.exp(-self.rho*(dist-mindist[k]))
                numer = np.dot(local, expfac)
                denom = np.sum(expfac)
        
                y_[k] = numer/denom
                c += 1
        # y_ = pou_crit_loop(X_cont, D, trx, fac, mindist, delta, self.energy_mode, self.higher_terms, self.H, self.rho)
        
        y_ = comm.allreduce(y_)
        ans = -abs(y_)

        
        """
        from stack: do this
        
        Your comment on the question indicates that you're in the special case of a binary integer linear programming problem. For these problems, a standard approach is to find an optimal solution, add a constraint to eliminate that particular solution, and then reoptimize to find another optimal solution.
        
        For example, if your first optimal solution has binary variables with values x1=1
        , x2=0, x3=1
        
        , then you can add the constraint
        
        (1−x1)+x2+(1−x3)≥1
        
        to eliminate the solution x1=1
        , x2=0, x3=1.
        
        """
        # if self.energy_mode:
        #     import pdb; pdb.set_trace()
        # for batches, loop over already added points to prevent clustering
        # this should only work for 
        for i in range(dir):
            ind = self.ntr + i
            work = x - trx[ind]
            # dirdist = np.sqrt(np.dot(work, work)) 
            dirdist = np.linalg.norm(work) 
            # ans += 1./(np.dot(work, work) + 1e-10)
            ans += np.exp(-self.rho*(dirdist + delta))

        return ans 
    







    # @njit(parallel=True)
    def _eval_grad(self, x, bounds, dir=0):
        X_cont = np.atleast_2d(x)
        numeval = X_cont.shape[0]
        cap = self.options["min_contribution"]
        cmin = self.options["min_points"]
        cases = divide_cases(numeval, size)
        dim = X_cont.shape[1]
        try:
            delta = self.model.options["delta"]
        except:
            delta = 1e-10

        Mc = np.ones(self.ntr)
        if self.options["scale_by_cond"]:
            Mc = self.Mc

        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        # exhaustive search for closest sample point, for regularization
        # D = cdist(X_cont, trx)
        # mindist = np.min(D, axis=1)

        D = cdist(X_cont[cases[rank],:], trx)

        neighbors_all, ball_rad = self.neighbors_func(X_cont, self.rho, cap, cmin, trx.shape[0], cases)
        mindist_p = np.min(D, axis=1)
        mindist = np.zeros(numeval)
        c = 0
        for k in cases[rank]:
            mindist[k] = mindist_p[c]
            c += 1

        mindist = comm.allreduce(mindist)

        fac_all = self.dV*Mc

        y_ = np.zeros(numeval)
        dy_ = np.zeros([numeval, dim])
        c = 0
        for k in cases[rank]:
        # for k in range(numeval):
        # for k in prange(numeval):

            neighbors = neighbors_all
            if ball_rad:
                # neighbors = neighbors_all[k]
                neighbors = neighbors_all[c]
                xc = trx[neighbors,:]
            fac = fac_all[neighbors]

            # for i in range(self.ntr):
            # work = X_cont[k,:] - trx[:self.ntr,:] 
            work = X_cont[k,:] - xc
            # dist = np.sqrt(D[k,:]**2 + delta)#np.sqrt(D[0][i] + delta)
            dist = np.sqrt(np.einsum('ij,ij->i',work,work)  + delta)
            # local = self.higher_terms(work, None, self.H)*fac
            local = self.higher_terms(work, None, self.H[neighbors])*fac
            # dlocal = self.higher_terms_deriv(work, None, self.H)
            dlocal = self.higher_terms_deriv(work, None, self.H[neighbors])
            
            ddist = np.einsum('i,ij->ij', 1./dist, work)
            dlocal = np.einsum('i,ij->ij', fac, dlocal)

            expfac = np.exp(-self.rho*(dist-mindist[k]))
            dexpfac = np.einsum('i,ij->ij',-self.rho*expfac, ddist)
            numer = np.dot(local,expfac)
            dnumer =  np.einsum('i,ij->j', local, dexpfac) + np.einsum('i,ij->j', expfac, dlocal)
            denom = np.sum(expfac)
            ddenom = np.sum(dexpfac, axis=0)

            y_[k] = numer/denom
            dy_[k] = (denom*dnumer - numer*ddenom)/(denom**2)

            c += 1

        y_ = comm.allreduce(y_)
        dy_ = comm.allreduce(dy_)
        ans = np.einsum('i,ij->ij',-np.sign(y_), dy_)

        #TODO: Parallelize this query as well?
        # for batches, loop over already added points to prevent clustering
        for i in range(dir):
            ind = self.ntr + i
            work = x - trx[ind]
            #dwork = np.eye(n)
            # d2 = np.dot(work, work)
            # dd2 = 2*work
            dirdist = np.linalg.norm(work) 
            # term = 1.0/(d2 + 1e-10)
            # ans += -1.0/((d2 + 1e-10)**2)*dd2
            ddirdist = work/dirdist
            quant = -self.rho*ddirdist*np.exp(-self.rho*(dirdist+ delta))
            ans += quant

            # import pdb; pdb.set_trace()
        return ans

    def higher_terms(self, dx, g, h):
        terms = np.zeros(dx.shape[0])
        
        # for j in range(dx.shape[0]):
        #     terms[j] = 0.5*innerMatrixProduct(h[j], dx[j].T)
        terms = 0.5*np.einsum('ij,ijk,ik->i', dx, h, dx)

        if self.options["return_rescaled"]:
            terms *= self.y_sca
        return terms

    def higher_terms_deriv(self, dx, g, h):
        # terms = (g*dx).sum(axis = 1)
        dterms = np.zeros_like(dx)
        # for j in range(dx.shape[0]):
        for d in range(dx.shape[1]):
            # dterms[j,d] = np.dot(h[j,d,:], dx[j,:])#0.5*innerMatrixProduct(h, dx)
            dterms = np.einsum('ik, ik ->i', h[:,d,:], dx)

        if self.options["return_rescaled"]:
            dterms *= self.y_sca
        return dterms



    def _pre_asopt(self, bounds, dir=0):
        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)

        # m, n = trx.shape

        # # factor in cell volume
        # fakebounds = copy.deepcopy(bounds)
        # fakebounds[:,0] = 0.
        # fakebounds[:,1] = 1.
        # if self.options["scale_by_volume"]:
        #     self.dV = estimate_pou_volume(trx, fakebounds)
        # else:
        #     self.dV = np.ones(trx.shape[0])
        # if(self.options["out_of_bounds"]):
        #     for i in range(self.dim):
        #         bounds[i][0] = -self.options["out_of_bounds"]
        #         bounds[i][1] = 1. + self.options["out_of_bounds"]

        return None, bounds# + 0.001*self.dminmax+randvec, bounds



    def _post_asopt(self, x, bounds, dir=0):

        #snap to edge if needed 
        # for i in range(self.dim):
        #     if(x[i] > 1.0):
        #         x[i] = 1.0
        #     if(x[i] < 0.0):
        #         x[i] = 0.0

        return x
    

    def neighbors_func(self, X_cont, rho, cap, cmin, numsample, cases):

        neighbors_all = list(range(numsample))
        ball_rad = None
        if(cap):
            ball_rad = -np.log(cap)/rho
            neighbors_all = self.tree.query_ball_point(X_cont[cases[rank],:], ball_rad)
            redo = []
            for i in range(len(neighbors_all)):
                over = len(neighbors_all[i]) - cmin
                if over < 0:
                    redo.append(i)

            if len(redo) > 0:
                dum, neighbors_redo = self.tree.query(X_cont[cases[rank],:][redo,:], cmin)

                for j in range(len(neighbors_redo)):
                    neighbors_all[redo[j]] = neighbors_redo[j]
        
        return neighbors_all, ball_rad

