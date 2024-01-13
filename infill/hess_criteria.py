import numpy as np
import copy

# from matplotlib import pyplot as plt
# from smt.utils.options_dictionary import OptionsDictionary
# from smt.sampling_methods import LHS
# from smt.surrogate_models import GEKPLS, KPLS, KRG
from surrogate.pougrad import POUHessian
from infill.refinecriteria import ASCriteria
# from scipy.linalg import lstsq, eig
from scipy.stats import qmc
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist, squareform
# from scipy.optimize import Bounds
from utils.sutils import divide_cases, innerMatrixProduct, quadraticSolveHOnly, symMatfromVec, estimate_pou_volume,  standardization2, gen_dist_func_nb
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# para = False
# try:
#     from numba import jit, njit, prange
#     para = True
# except:
#     pass

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
            size-1,
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
            

        else: 
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
        ind_use = [l for l in range(dx.shape[1])]
        if self.options["grad_select"] is not None: 
            ind_use = self.options["grad_select"]

        # terms = np.dot(h, dx)
        terms = np.einsum('ijk, ik->ij', h, dx)
        
        if self.options["return_rescaled"]:
            terms = np.einsum('j,ij->ij',self.y_sca/self.x_sca, terms)

        # if self.energy_mode:
        #     return terms[:,ind_use]

        avg_terms = np.linalg.norm(terms[:,ind_use], axis=1)
        return avg_terms

    def higher_terms_deriv(self, dx, g, h):
        
        ind_use = [l for l in range(dx.shape[1])]
        if self.options["grad_select"] is not None: 
            ind_use = self.options["grad_select"]

        scaler = np.ones(dx.shape[1])
        if self.options["return_rescaled"]:
            scaler[:] = self.y_sca/self.x_sca[:]

        davg_terms = np.zeros_like(dx)
        terms = np.einsum('ijk, ik->ij', h, dx)
        terms = np.einsum('j,ij->ij',scaler, terms)
        # davg_terms[:,ind_use] = np.einsum('ij,ijk,k->ij',terms[:,ind_use], h[:,:,ind_use][:,ind_use,:], scaler[ind_use])
        xH = np.einsum('j,ijk->ijk',scaler, h)
        davg_terms = np.einsum('ikj,ik->ij',xH[:,ind_use,:], terms[:,ind_use])
        avg_terms = np.linalg.norm(terms[:,ind_use], axis=1)
        davg_terms = np.einsum('ij,i->ij', davg_terms, 1./avg_terms)
        # import pdb; pdb.set_trace()
        return davg_terms




# def rcrit_func_e(k, X_cont, D, trx, fac, mindist, delta, energy_mode, higher_terms, H, rho):        
#     work = X_cont[k,:] - trx
#     dist = np.sqrt(D[k,:]**2 + delta)#np.sqrt(D[0][i] + delta)
#     local = np.einsum('ij,i->ij', higher_terms(work, None, H), fac) # NEWNEWNEW
#     expfac = np.exp(-rho*(dist-mindist[k]))
#     numer = np.einsum('ij,i->j', local, expfac)
#     denom = np.sum(expfac)

#     # y_[k] = numer/denom
#     return numer/denom


# def rcrit_func(k, X_cont, D, trx, fac, mindist, delta, energy_mode, higher_terms, H, rho):        
#     work = X_cont[k,:] - trx
#     dist = np.sqrt(D[k,:]**2 + delta)#np.sqrt(D[0][i] + delta)
#     local = higher_terms(work, None, H)*fac # NEWNEWNEW
#     expfac = np.exp(-rho*(dist-mindist[k]))
#     numer = np.dot(local, expfac)
#     denom = np.sum(expfac)

#     return numer/denom
#     # y_[k] = numer/denom


# import multiprocessing
# from functools import partial

# def pou_crit_loop(X_cont, D, trx, fac, mindist, delta, energy_mode, higher_terms, H, rho):
#     numeval = X_cont.shape[0]
#     y_ = np.zeros(numeval)
#     pool = multiprocessing.Pool(8)
#     args = (range(numeval), X_cont, D, trx, fac, mindist, delta, energy_mode, higher_terms, H, rho)
#     if energy_mode:
#         y_ = np.zeros([numeval, higher_terms(X_cont[0,:] - trx, None, H).shape[1]])
#         # for k in range(numeval):
#         funcx = partial(rcrit_func_e, X_cont=X_cont, D=D, trx=trx, fac=fac, mindist=mindist, delta=delta, energy_mode=energy_mode, higher_terms=higher_terms, H=H, rho=rho)
#     else:
#         funcx = partial(rcrit_func, X_cont=X_cont, D=D, trx=trx, fac=fac, mindist=mindist, delta=delta, energy_mode=energy_mode, higher_terms=higher_terms, H=H, rho=rho)
        
#     y_ = pool.map(funcx, range(numeval))

#     return np.array(y_)


"""
Overhead makes this too slow, not enough benefit

serial: 0.33

parallel 1: 0.44

parallel 4: 0.23
parallel 8: 0.21


"""