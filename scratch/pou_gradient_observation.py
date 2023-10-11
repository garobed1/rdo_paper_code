import sys, os
import importlib
import time
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
from utils.sutils import convert_to_smt_grads, print_rc_plots
from functions.problem_picker import GetProblem
from surrogate.pougrad import POUHessian
from infill.getxnew import adaptivesampling
from infill.hess_criteria import HessianRefine, HessianGradientRefine
from smt.sampling_methods import LHS, FullFactorial, Random
from scipy.stats import qmc

"""
Plot the gradient of a function, the gradient of the POU model of the function, and each individual term in the POU model gradient
"""


dim = 4
# func = GetProblem('tensexp', dim)
# func = GetProblem('lpnorm', dim)
# func = GetProblem('betatestex', dim)
func = GetProblem('shortcol', dim)
# func = GetProblem('arctan', dim)
# func = GetProblem('fuhgp3', dim)
# func = GetProblem('fuhgsh', dim)
# func = GetProblem('expsine', dim)
xlimits = np.array(func.xlimits)
xlimits_u = xlimits[0:3,:]
# sampling_f = LHS(xlimits = xlimits, criterion='maximin')
sampling_f = FullFactorial(xlimits = xlimits)#, criterion='maximin')
sampling_u = LHS(xlimits = xlimits[0:3,:], criterion='maximin')
sampling_r = Random(xlimits = xlimits)

n = 100


# n1 = 10
# x1 = np.zeros([n1,dim])
# x1u = sampling_u(n1)
# x1[:,0:3] = x1u[:,:]
# x1[:,3] = 5.0

# n2 = 10
# x2 = np.zeros([n2,dim])
# x2u = sampling_u(n2)
# x2[:,0:3] = x2u[:,:]
# x2[:,3] = 6.0


xt = sampling_f(n)
# xt = np.concatenate([x1, x2])
xt_s = qmc.scale(xt, xlimits[:,0], xlimits[:,1], reverse=True)
ft = func(xt)
gt = convert_to_smt_grads(func, xt)

neval = dim + 1
rsca = True
model = POUHessian(bounds=xlimits, rscale = 5.5, neval = neval, min_contribution = 0.0)
# model = POUHessian(bounds=xlimits, rho = 1000., neval = neval, min_contribution = 0.0)
model.options.update({"print_prediction":False})
model.set_training_values(xt, ft)
convert_to_smt_grads(model, xt, gt)
model.train()

xe = sampling_r(3)

# t0 = time.time()
# # fe = model.predict_values(xe)
# fe = model.predict_derivatives(xe, kx=0)
# t1 = time.time()

# rcfunc = HessianRefine(model, gt, xlimits, rscale = 5.5, neval = neval, scale_by_volume=False, return_rescaled=rsca, sub_index=[0])
# rcfunc.pre_asopt(xlimits)
# rcgrad = HessianGradientRefine(model, gt, xlimits, rscale = 5.5, neval = dim+3, scale_by_volume=False, return_rescaled=rsca, 
#                             #    pdf_weight=[['norm', 0.5, 0.16666666667], 1.], 
#                             #    pdf_weight=[['uniform'], 1.], 
#                                grad_select=[1], sub_index=[0])
rcgrad = HessianGradientRefine(model, gt, xlimits, rho = 80., neval = dim+3, scale_by_volume=False, return_rescaled=rsca, 
                            #    pdf_weight=[['norm', 500., 100.], ['norm', 2000., 400.], ['lognorm', 5., 0.5], 0.], 
                            #    pdf_weight=[['norm', 500., 100.], ['norm', 2000., 400.], 0., 0.], 
                               grad_select=[3], sub_index=[0,1,2])
rcgrad.pre_asopt(xlimits)
# rcfunc.set_static(np.array([6.0]))
# rcgrad.set_static(np.array([100., 6.0]))
rcgrad.set_static(np.array([6.0]))

# ff=rcgrad.evaluate(xt_s[9], xlimits)
fe=rcgrad.get_energy(xlimits)
import pdb; pdb.set_trace()
print_rc_plots(xlimits_u, "observe", rcgrad, 0)
mf, rF, d1, d2, d3 = adaptivesampling(func, model, rcgrad, xlimits, 10)

# gather the 3 gradient-contributing components
print_rc_plots(xlimits_u, "observe", rcgrad, 0)
import pdb; pdb.set_trace()


if dim == 1:
    ndir = 200
    xp = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
    F  = np.zeros([ndir]) 
    TF  = np.zeros([ndir]) 
    G  = np.zeros([ndir]) 
    G1  = np.zeros([ndir]) 
    G2  = np.zeros([ndir]) 
    G3  = np.zeros([ndir]) 
    TG  = np.zeros([ndir]) 
    RCF  = np.zeros([ndir]) 
    RCG  = np.zeros([ndir]) 

    for i in range(ndir):
        xi = np.zeros([1,1])
        xi[0,0] = xp[i]
        xi_s = qmc.scale(xi, xlimits[:,0], xlimits[:,1], reverse=True)
        F[i]  = model.predict_values(xi)
        TF[i]  = func(xi)
        G[i] = model.predict_derivatives(xi, kx=0)
        G1[i], G2[i], G3[i] = model.get_gradient_terms(xi, kx=0)
        TG[i]  = func(xi, kx=0)
        RCF[i] = rcfunc.evaluate(xi_s, xlimits)
        RCG[i] = rcgrad.evaluate(xi_s, xlimits)
    plt.plot(xp, F, label = 'POU') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, TF, label = 'true') #, levels = np.linspace(np.min(F), 0., 25)
    plt.scatter(xt, ft, label = 'data')
    plt.legend()
    plt.savefig(f"./pou_obs_function.pdf")    
    plt.clf()

    plt.plot(xp, abs(F-TF), label = 'POU Err') #, levels = np.linspace(np.min(F), 0., 25)
    # plt.plot(xp, abs(G-TG), label = 'POU (g) Err') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, abs(RCF), label = 'POU Func RC') #, levels = np.linspace(np.min(F), 0., 25)
    # plt.plot(xp, abs(RCG), label = 'POU (g) Grad RC') #, levels = np.linspace(np.min(F), 0., 25)
    plt.scatter(xt, np.zeros(xt.shape[0]), label = 'data')
    plt.legend()
    plt.savefig(f"./pou_obs_function_err.pdf")    
    plt.clf()


    plt.plot(xp, G, label = 'POU (g)') #, levels = np.linspace(np.min(F), 0., 25)
    # plt.plot(xp, G1, label = 'G1') #, levels = np.linspace(np.min(F), 0., 25)
    # plt.plot(xp, G2, label = 'G2') #, levels = np.linspace(np.min(F), 0., 25)
    # plt.plot(xp, G3, label = 'G3') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, TG, label = 'true (g)') #, levels = np.linspace(np.min(F), 0., 25)
    plt.scatter(xt, gt, label = 'data (g)')
    plt.legend()
    plt.savefig(f"./pou_obs_gradient.pdf")  
    plt.clf()


    plt.plot(xp, G, label = 'POU (g)') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, G2, label = 'G2') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, G1+G3, label = 'G1 + G3') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, TG, label = 'true (g)') #, levels = np.linspace(np.min(F), 0., 25)
    plt.scatter(xt, gt, label = 'data (g)')
    plt.legend()  
    plt.savefig(f"./pou_obs_gradient2.pdf")    
    plt.clf()

    plt.plot(xp, abs(G-TG), label = 'POU (g) Err') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, abs(RCG), label = 'POU (g) Grad RC') #, levels = np.linspace(np.min(F), 0., 25)
    # plt.plot(xp, abs(G2-TG), label = 'G2 Contr.') #, levels = np.linspace(np.min(F), 0., 25)
    # plt.plot(xp, abs(G1+G3-TG), label = 'G1 + G3 Contr.') #, levels = np.linspace(np.min(F), 0., 25)
    plt.scatter(xt, np.zeros(xt.shape[0]), label = 'data (g)')
    plt.legend()  
    plt.savefig(f"./pou_obs_gradient_err.pdf")    
    plt.clf()

    plt.plot(xp, abs(G), label = 'POU (g) Mag') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, abs(G2), label = 'G2 Contr.') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, abs(G1+G3), label = 'G1 + G3 Contr.') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, abs(TG), label = 'true Mag') #, levels = np.linspace(np.min(F), 0., 25)
    plt.scatter(xt, np.zeros(xt.shape[0]), label = 'data (g)')
    plt.legend()  
    plt.savefig(f"./pou_obs_gradient4.pdf")    
    plt.clf()

    plt.plot(xp, abs(F-TF), label = 'POU Err') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, abs(G-TG), label = 'POU (g) Err') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, abs(RCF), label = 'POU Func RC') #, levels = np.linspace(np.min(F), 0., 25)
    plt.plot(xp, abs(RCG), label = 'POU (g) Grad RC') #, levels = np.linspace(np.min(F), 0., 25)
    plt.scatter(xt, np.zeros(xt.shape[0]), label = 'data')
    plt.legend()
    plt.savefig(f"./pou_obs_all_err.pdf")    
    plt.clf()