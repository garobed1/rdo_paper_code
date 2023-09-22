import sys, os
import importlib
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
from utils.sutils import convert_to_smt_grads
from functions.problem_picker import GetProblem
from surrogate.pougrad import POUHessian
from infill.hess_criteria import HessianRefine, HessianGradientRefine
from smt.sampling_methods import LHS, FullFactorial
from scipy.stats import qmc

"""
Plot the gradient of a function, the gradient of the POU model of the function, and each individual term in the POU model gradient
"""


dim = 1
func = GetProblem('tensexp', dim = 2)
# func = GetProblem('lpnorm', dim = 1)
# func = GetProblem('arctan', dim)
# func = GetProblem('fuhgp3', dim)
# func = GetProblem('fuhgsh', dim)
# func = GetProblem('expsine', dim)
xlimits = np.array(func.xlimits)
# sampling_f = LHS(xlimits = xlimits, criterion='maximin')
sampling_f = FullFactorial(xlimits = xlimits)#, criterion='maximin')

n = 15

xt = sampling_f(n)
ft = func(xt)
gt = convert_to_smt_grads(func, xt)

neval = dim + 1
rsca = True
model = POUHessian(bounds=xlimits, rscale = 5.5, neval = neval, min_contribution = 0.0)
model.options.update({"print_prediction":False})
model.set_training_values(xt, ft)
convert_to_smt_grads(model, xt, gt)
model.train()
# rcfunc = HessianRefine(model, gt, xlimits, rscale = 5.5, neval = neval, scale_by_volume=False, return_rescaled=rsca)
# rcfunc.pre_asopt(xlimits)
rcgrad = HessianGradientRefine(model, gt, xlimits, rscale = 5.5, neval = dim+3, scale_by_volume=False, return_rescaled=rsca)
rcgrad.pre_asopt(xlimits)

# gather the 3 gradient-contributing components



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