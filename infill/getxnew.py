from optimization.optimizers import optimize
from mpi4py import MPI
import copy
from optimization.defaults import DefaultOptOptions
import numpy as np
from smt.surrogate_models import GEKPLS
from surrogate.direct_gek import DGEK
from scipy.stats import qmc
from surrogate.pougrad import POUSurrogate
from utils.error import rmse, meane, full_error
from utils.sutils import convert_to_smt_grads

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
The adaptive sampling routine. Given a refinement criteria, find its corresponding 
optimal observation point.

Parameters
----------
rcrit : ASCriteria object
    Refinement criteria function
options : dictionary object
    Optimization settings

Returns
----------
xnew : ndarray
    New adaptive sampling point


"""

def getxnew(rcrit, bounds, batch, x_init=None, options=None):
    
    # set default options if None is provided
    if(options == None):
        options = DefaultOptOptions
    xnew = []
    bounds_used = bounds
    n = len(bounds)
    unit_bounds = np.zeros([n,2])
    unit_bounds[:,1] = 1.

    # check if refining on limited subspace, and modify lambda functions accordingly
    # do nothing with x_init if this isn't the case
    # NOTE: doesn't work for constrained criteria for now
    sub_ind = np.arange(0, n).tolist()
    fix_ind = []
    xfix = 0
    if rcrit.options['sub_index'] is not None and not isinstance(rcrit.condict, dict):
        sub_ind = rcrit.options['sub_index']
        m_e = len(sub_ind)
        # grab values for other indices from x_init
        fix_ind = [x for x in np.arange(0, n).tolist() if x not in sub_ind]
        xfix = qmc.scale(np.array([x_init]), bounds[fix_ind,0], bounds[fix_ind,1], reverse=True)#[fix_ind]

    def eval_eff(x, bounds, dir):
        x_eff = np.zeros(n)
        x_eff[sub_ind] = x
        x_eff[fix_ind] = xfix

        y = rcrit.evaluate(x_eff, bounds, dir)
        return y
    def eval_grad_eff(x, bounds, dir):
        x_eff = np.zeros(n)
        x_eff[sub_ind] = x
        x_eff[fix_ind] = xfix

        dy = rcrit.eval_grad(x_eff, bounds, dir)
        return dy[sub_ind]


    # loop over batch
    for i in range(batch):
        rx = None
        if(rcrit.opt): #for methods that don't use optimization
            x0, lbounds = rcrit.pre_asopt(bounds, dir=i)
            x0 = qmc.scale(x0, bounds_used[:,0], bounds_used[:,1], reverse=True)
            m, n = x0.shape
            if(lbounds is not None):
                bounds_used = lbounds
            args=(bounds_used, i,)
            if(isinstance(rcrit.condict, dict) ):
                rcrit.condict["args"] = [bounds_used, i]
            jac = None
            if(rcrit.supports["obj_derivatives"]):
                jac = eval_grad_eff
            if(options["local"]):

                # proper multistart
                if(options["multistart"] == 2):
                    resx = np.zeros([m,n])
                    resy = np.zeros(m)
                    succ = np.full(m, True)
                    for j in range(m):
                        
                        results = optimize(eval_eff, args=args, bounds=unit_bounds[sub_ind,:], type="local", constraints=rcrit.condict, jac=jac, x0=x0[j,sub_ind])
                        resx[j,:] = results.x
                        resy[j] = results.fun
                        succ[j] = results.success
                    valid = np.where(succ)[0]
                    try:
                        rx = resx[valid[np.argmin(resy[valid])]]
                    except:
                        rx = resx[np.argmin(resy)]
                    # print(rx)

                # start at best point
                elif(options["multistart"] == 1):
                    x0b = None
                    y0 = np.zeros(m)
                    for j in range(m):
                        y0[j] = eval_eff(x0[j], bounds_used, i)
                    ind = np.argmin(y0)
                    x0b = x0[0]
                    results = optimize(eval_eff, args=args, bounds=unit_bounds[sub_ind,:], type="local", constraints=rcrit.condict, jac=jac, x0=x0b[sub_ind])
                    rx = results.x

                # perform one optimization
                else:
                    x0b = x0[0]
                    results = optimize(eval_eff, args=args, bounds=unit_bounds[sub_ind,:], type="local", constraints=rcrit.condict, jac=jac, x0=x0b[sub_ind])
                    rx = results.x

            else:
                results = optimize(eval_eff, args=args, bounds=unit_bounds[sub_ind,:], type="global", constraints=rcrit.condict)
                rx = results.x
            
            rx = qmc.scale(np.array([rx]), bounds_used[sub_ind,0], bounds_used[sub_ind,1])
            rx = rx[0]
        else:
            rx = None

        # fixed variables are added back in post_asopt
        xnew.append(rcrit.post_asopt(rx, bounds, dir=i))

    return xnew



def adaptivesampling(func, model0, rcrit, bounds, ntr, batch=1, options=None):

    count = int(ntr/batch)
    hist = []
    errh = []
    errh2 = []
    model = copy.deepcopy(model0)
    
    intervals = np.arange(0, count+1)

    for i in range(count):
        # try:
        if 1:
            t0 = model.training_points[None][0][0]
            f0 = model.training_points[None][0][1]
            g0 = rcrit.grad
            nt, dim = t0.shape
            #x0 = np.zeros([1, dim])

            # get the new points
            xnew = np.array(getxnew(rcrit, bounds, batch, x_init=rcrit.fix_val, options=options))
            # import pdb; pdb.set_trace()
            # add the new points to the model
            t0 = np.append(t0, xnew, axis=0)
            f0 = np.append(f0, func(xnew), axis=0)
            g0 = np.append(g0, convert_to_smt_grads(func, xnew), axis=0)
            # g0 = np.append(g0, np.zeros([xnew.shape[0], xnew.shape[1]]), axis=0)
            # for j in range(dim):
            #     g0[nt:,j] = func(xnew, j)[:,0]

            model.set_training_values(t0, f0)
            convert_to_smt_grads(model, t0, g0)
            # if(isinstance(model, GEKPLS) or isinstance(model, POUSurrogate) or isinstance(model0, DGEK)):
            #     for j in range(dim):
            #         model.set_training_derivatives(t0, g0[:,j], j)
            model.train()

            # evaluate errors
            if(options["errorcheck"] is not None):
                xdata, fdata, intervals = options["errorcheck"]
                # err = rmse(model, func, xdata=xdata, fdata=fdata)
                # err2 = meane(model, func, xdata=xdata, fdata=fdata)
                if(i in intervals.tolist() and i !=0):
                    err = full_error(model, func, xdata=xdata, fdata=fdata)
                    errh.append(err[0])
                    errh2.append(err[1:])
                # import pdb; pdb.set_trace()
                    #print("yes")


            else:
                errh = None
                errh2 = None
                #hist = None

            # save training data at each interval regardless
            if i in intervals.tolist():
                hist.append(copy.deepcopy(rcrit.model.training_points[None]))        

            if(rcrit.options["print_iter"] and rank == 0):
                print("Iteration: ", i)

            # replace criteria
            rcrit.initialize(model, g0)
        # except:
        #     print(f"Run on processor {rank} failed, returning what we have")
        #     continue
        
    return model, rcrit, hist, errh, errh2
