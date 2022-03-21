from optimizers import optimize
import copy
from defaults import DefaultOptOptions
import numpy as np
from smt.surrogate_models import GEKPLS
from pougrad import POUSurrogate
from error import rmse, meane


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

def getxnew(rcrit, x0, bounds, options=None):
    
    # set default options if None is provided
    if(options == None):
        options = DefaultOptOptions
    
    xnew = []
    bounds_used = bounds

    #gresults = optimize(rcrit.evaluate, args=(), bounds=bounds, type="global")
    for i in range(rcrit.nnew):
        rx = None
        if(rcrit.opt): #for methods that don't use optimization
            x0, lbounds = rcrit.pre_asopt(bounds, dir=i)
            if(lbounds is not None):
                bounds_used = lbounds
            args=(i,)
            rcrit.condict["args"] = [i]
            jac = None
            if(rcrit.supports["obj_derivatives"]):
                jac = rcrit.eval_grad
            if(options["localswitch"]):
                results = optimize(rcrit.evaluate, args=args, bounds=bounds_used, type="local", constraints=rcrit.condict, jac=jac, x0=x0)
            else:
                results = optimize(rcrit.evaluate, args=args, bounds=bounds_used, type="global", constraints=rcrit.condict)
            rx = results.x
        else:
            rx = None

        xnew.append(rcrit.post_asopt(rx, bounds, dir=i))

    return xnew


def adaptivesampling(func, model0, rcrit, bounds, ntr, options=None):

    #TODO: Alternate Stopping Criteria
    count = int(ntr/rcrit.nnew)
    hist = []
    errh = []
    errh2 = []
    model = copy.deepcopy(model0)
    

    for i in range(count):
        t0 = model.training_points[None][0][0]
        f0 = model.training_points[None][0][1]
        g0 = rcrit.grad
        nt, dim = t0.shape
        x0 = np.zeros([1, dim])
        # if(isinstance(model, GEKPLS) or isinstance(model, POUSurrogate)):
        #     for i in range(dim):
        #         g0.append(model.training_points[None][i+1][1])

        # get the new points
        xnew = np.array(getxnew(rcrit, x0, bounds, options))

        # add the new points to the model
        t0 = np.append(t0, xnew, axis=0)
        f0 = np.append(f0, func(xnew), axis=0)
        g0 = np.append(g0, np.zeros([xnew.shape[0], xnew.shape[1]]), axis=0)
        for j in range(dim):
            g0[nt:,j] = func(xnew, j)[:,0]
        model.set_training_values(t0, f0)
        if(isinstance(model, GEKPLS) or isinstance(model, POUSurrogate)):
            for j in range(dim):
                model.set_training_derivatives(t0, g0[:,j], j)
        model.train()


        # evaluate error, rmse for now
        if(options["errorcheck"] is not None):
            xdata, fdata = options["errorcheck"]
            err = rmse(model, func, xdata=xdata, fdata=fdata)
            err2 = meane(model, func, xdata=xdata, fdata=fdata)
            errh.append(err)
            errh2.append(err2)
        else:
            errh = None
            errh2 = None

        hist.append(copy.deepcopy(rcrit))

        if(rcrit.options["print_iter"]):
            print("Iteration: ", i)

        # replace criteria
        rcrit.initialize(model, g0)

    return model, rcrit, hist, errh, errh2
