import numpy as np
import copy

from matplotlib import pyplot as plt
from smt.utils.options_dictionary import OptionsDictionary
from smt.sampling_methods import LHS
from smt.surrogate_models import GEKPLS, KPLS, KRG
from surrogate.pougrad import POUCV, POUError, POUErrorVol, POUMetric, POUSurrogate, POUHessian
from infill.refinecriteria import ASCriteria
from scipy.linalg import lstsq, eig
from scipy.stats import qmc
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import Bounds
from utils.sutils import innerMatrixProduct, quadraticSolveHOnly, symMatfromVec, estimate_pou_volume,  standardization2


"""
Refine based on a first-order Taylor approximation using the gradient. Pull 
Hessian estimates from the surrogate if available, to avoid unnecessary 
computation.
"""
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
        self.rho = self.options['rscale']*pow(self.ntr, 1./self.dim)

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

            self.H = hess
            self.Mc = mcs#/np.max(mcs)
        

    # Assumption is that the quadratic terms are the error
    def _evaluate(self, x, bounds, dir=0):
        
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
        D = cdist(x, trx)
        mindist = min(D[0])

        numer = 0
        denom = 0

        for i in range(self.ntr):
            work = x[0,:] - trx[i]
            dist = np.sqrt(D[0][i]**2 + delta)#np.sqrt(D[0][i] + delta)
            local = self.higher_terms(work, None, self.H[i])*self.dV[i]*Mc[i] # NEWNEWNEW
            expfac = np.exp(-self.rho*(dist-mindist))
            numer += local*expfac
            denom += expfac

        y = numer/denom

        
        ans = -abs(y)

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
        
        # for batches, loop over already added points to prevent clustering
        for i in range(dir):
            ind = self.ntr + i
            work = x - trx[ind]
            dirdist = np.sqrt(np.dot(work, work)) 
            # ans += 1./(np.dot(work, work) + 1e-10)
            ans += np.exp(-self.rho*(dirdist+ delta))

        return ans 


    def _eval_grad(self, x, bounds, dir=0):
        
        try:
            delta = self.model.options["delta"]
        except:
            delta = 1e-10

        Mc = np.ones(self.ntr)
        if self.options["scale_by_cond"]:
            Mc = self.Mc

        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)
        # exhaustive search for closest sample point, for regularization
        D = cdist(x, trx)
        mindist = min(D[0])

        numer = 0
        denom = 0
        dnumer = np.zeros(self.dim)
        ddenom = np.zeros(self.dim)
        dwork = np.ones(self.dim)

        for i in range(self.ntr):
            work = x[0,:] - trx[i]
            dist = np.sqrt(D[0][i]**2 + delta)#np.sqrt(D[0][i] + delta)
            ddist = (work/D[0][i])*(D[0][i]/dist)
            local = self.higher_terms(work, None, self.H[i])*self.dV[i]*Mc[i]
            dlocal = self.higher_terms_deriv(work, None, self.H[i])*self.dV[i]*Mc[i]
            expfac = np.exp(-self.rho*(dist-mindist))
            dexpfac = -self.rho*expfac*ddist
            numer += local*expfac
            dnumer += local*dexpfac + dlocal*expfac
            denom += expfac
            ddenom += dexpfac

        y = numer/denom
        dy = (denom*dnumer - numer*ddenom)/(denom**2)

        ans = -np.sign(y)*dy

        # for batches, loop over already added points to prevent clustering
        for i in range(dir):
            ind = self.ntr + i
            work = x - trx[ind]
            #dwork = np.eye(n)
            # d2 = np.dot(work, work)
            # dd2 = 2*work
            dirdist = np.sqrt(np.dot(work, work)) 
            # term = 1.0/(d2 + 1e-10)
            # ans += -1.0/((d2 + 1e-10)**2)*dd2
            ddirdist = work/dirdist
            ans += -self.rho*ddirdist*np.exp(-self.rho*(dirdist+ delta))
        

        
        return ans

    def higher_terms(self, dx, g, h):
        terms = 0.5*innerMatrixProduct(h, dx.T)

        if self.options["return_rescaled"]:
            terms *= self.y_sca
        return terms

    def higher_terms_deriv(self, dx, g, h):
        # terms = (g*dx).sum(axis = 1)
        dterms = np.zeros(dx.shape[0])
        for j in range(dx.shape[0]):
            dterms[j] += np.dot(h[j,:], dx)#0.5*innerMatrixProduct(h, dx)
        
        if self.options["return_rescaled"]:
            dterms *= self.y_sca
        return dterms



    def _pre_asopt(self, bounds, dir=0):
        
        trx = qmc.scale(self.trx, bounds[:,0], bounds[:,1], reverse=True)

        m, n = trx.shape

        # factor in cell volume
        fakebounds = copy.deepcopy(bounds)
        fakebounds[:,0] = 0.
        fakebounds[:,1] = 1.
        if self.options["scale_by_volume"]:
            self.dV = estimate_pou_volume(trx, fakebounds)
        else:
            self.dV = np.ones(trx.shape[0])
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



"""
Same as HessianRefine, but instead uses the norm of H(x-x_k) to determine
the criteria function. Serves as an indicator 

"""
class HessianGradientRefine(HessianRefine):
    def __init__(self, model, grad, bounds, **kwargs):



        super().__init__(model, grad, bounds, **kwargs)
        self.name = 'POUHESSGRAD'


    def _init_options(self):

        super()._init_options()
        
        declare = self.options.declare

        declare(
            "grad_select", 
            None, 
            types=(list),
            desc="can select specific gradient dimensions to take the average of"
        )


    def higher_terms(self, dx, g, h):

        # terms = np.zeros([dx.shape[0]])
        ind_use = [l for l in range(dx.shape[0])]
        if self.options["grad_select"] is not None: 
            ind_use = self.options["grad_select"]

        terms = np.dot(h, dx)
        
        if self.options["return_rescaled"]:
            terms *= self.y_sca/self.x_sca

        avg_terms = np.linalg.norm(terms[ind_use])
        return avg_terms

    def higher_terms_deriv(self, dx, g, h):
        
        ind_use = [l for l in range(dx.shape[0])]
        if self.options["grad_select"] is not None: 
            ind_use = self.options["grad_select"]

        scaler = np.ones(dx.shape[0])
        if self.options["return_rescaled"]:
            scaler[:] = self.y_sca/self.x_sca[:]

        davg_terms = np.zeros([dx.shape[0]])
        terms = np.dot(h, dx)
        terms *= scaler
        for k in ind_use:
            davg_terms += terms[k] * h[k,:] * scaler[k]
        avg_terms = np.linalg.norm(terms[ind_use])
        davg_terms /= avg_terms
        return davg_terms






