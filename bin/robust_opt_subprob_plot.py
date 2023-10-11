#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import importlib, shutil
import openmdao.api as om
from uq_comp.stat_comp_comp import StatCompComponent
from optimization.opt_subproblem import SequentialFullSolve
from optimization.opt_trust_uncertain import UncertainTrust
from surrogate.pougrad import POUHessian
from collections import OrderedDict
import os, copy
from functions.problem_picker import GetProblem
from optimization.robust_objective import RobustSampler, CollocationSampler, AdaptiveSampler
from optimization.defaults import DefaultOptOptions
import argparse
from mpi4py import MPI
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rcParams['font.size'] = '15'

"""
Plot robust trust optimization results
"""


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--optdir', action='store', help = 'directory containing opt run outputs and settings')
parser.add_argument('-s', '--skip_most', action='store_true', help = 'skip first few plots')



args = parser.parse_args()
optdir = args.optdir
skip = args.skip_most

# import settings from given config files
root = os.getcwd()
optsplit = optdir.split(sep='/')
title = optsplit[-1]
path = optsplit[:-1]
if len(path) == 0:
    path = ''
else:
    path = path.join('/')

with open(f'/{root}/{path}/{title}/grad_lhs.pickle', 'rb') as f:
    grad_lhs = pickle.load(f)
with open(f'/{root}/{path}/{title}/grad_rhs.pickle', 'rb') as f:
    grad_rhs = pickle.load(f)
with open(f'/{root}/{path}/{title}/radii.pickle', 'rb') as f:
    radii = pickle.load(f)
with open(f'/{root}/{path}/{title}/realizations.pickle', 'rb') as f:
    realizations = pickle.load(f)
with open(f'/{root}/{path}/{title}/areds.pickle', 'rb') as f:
    areds = pickle.load(f)
with open(f'/{root}/{path}/{title}/preds.pickle', 'rb') as f:
    preds = pickle.load(f)
with open(f'/{root}/{path}/{title}/loc.pickle', 'rb') as f:
    loc = pickle.load(f)
with open(f'/{root}/{path}/{title}/models.pickle', 'rb') as f:
    models = pickle.load(f)
with open(f'/{root}/{path}/{title}/reflog.pickle', 'rb') as f:
    reflog = pickle.load(f)
with open(f'/{root}/{path}/{title}/prob_truth.pickle', 'rb') as f:
    prob_truth = pickle.load(f)
puse = path.split('/')
puse = '.'.join(puse)
optuse = '.'.join([title, 'opt_settings']) # missing path for now
suse = '.'.join([title, 'sam_settings'])
oset = importlib.import_module(optuse)
sset = importlib.import_module(suse)
# with open(f'/{root}/{path}/{title}/prob_model.pickle', 'rb') as f:
#     grad_lhs = pickle.load(f)

if not os.path.isdir(f"/{root}/{path}/{title}/plots"):
    os.mkdir(f"/{root}/{path}/{title}/plots")



############################
#### SCRIPT BEGINS HERE ####
############################
name = oset.name
print_plots = oset.print_plots

# variables
u_dim = oset.u_dim
d_dim = oset.d_dim
external_only = (oset.use_truth_to_train and oset.use_surrogate) #NEW
pdfs = oset.pdfs
t_dim = u_dim + d_dim
func = GetProblem(oset.prob, t_dim)
xlimits = func.xlimits

# uq settings
use_surrogate = oset.use_surrogate
full_surrogate = oset.full_surrogate
retain_uncertain_points = oset.retain_uncertain_points
retain_uncertain_points_T = oset.retain_uncertain_points_T
eta_use = oset.eta_use
use_truth_to_train = oset.use_truth_to_train
ref_strategy = oset.ref_strategy

# start at just one point for now
x_init = oset.x_init

# optimization
# args = (func, eta_use)
jump = oset.jump
opt_settings = oset.opt_settings
max_outer = oset.max_outer
trust_region_bound = oset.trust_region_bound
initial_trust_radius = oset.initial_trust_radius
inexact_gradient_only = oset.inexact_gradient_only
approximate_model = oset.approximate_model
approximate_truth = oset.approximate_truth
approximate_truth_max = oset.approximate_truth_max
trust_increase_terminate = oset.trust_increase_terminate


### SAMPLING STRATEGY ###
sample_type = oset.sample_type

### TRUTH SAMPLERS ###
N_t = oset.N_t
if(sample_type == 'SC'):
    jump = oset.scjump
    N_t = oset.scN_t
    approximate_truth_max = oset.sc_approximate_truth_max
    sampler_t = CollocationSampler(np.array([x_init]), N=N_t, 
                          name='truth', 
                          xlimits=xlimits, 
                          probability_functions=pdfs)
else:
    sampler_t = RobustSampler(np.array([x_init]), N=N_t, 
                          name='truth', 
                          xlimits=xlimits, 
                          probability_functions=pdfs, 
                          retain_uncertain_points=retain_uncertain_points_T)


# get variable bounds from sampler_t
xlimits_d = xlimits[sampler_t.x_d_ind]
xlimits_u = xlimits[sampler_t.x_u_ind]

##### SURROGATE MODEL PARAMETERS #####
#TODO: WRITE SURROGATE PICKER
msur = None
if use_surrogate:
    rscale = 5.5
    if hasattr(sset, 'rscale'):
        rscale = sset.rscale
    rho = 10 
    if hasattr(sset, 'rho'):
        rscale = sset.rho
    # NOTE: NON-ZERO CAUSES NANS
    # min_contributions = 1e-12
    min_contributions = 0.0

    if(full_surrogate):
        sdim = t_dim
        msur = POUHessian(bounds=xlimits)
    else:
        sdim = u_dim
        msur = POUHessian(bounds=xlimits_u)

    neval = sset.neval_fac*t_dim+sset.neval_add
    msur.options.update({"rscale":rscale})
    msur.options.update({"rho":rho})
    msur.options.update({"neval":neval})
    msur.options.update({"min_contribution":min_contributions})
    msur.options.update({"print_prediction":False})
    msur.options.update({"print_global":False})


##### ADAPTIVE SURROGATE UQ PARAMETERS #####
max_batch = sset.batch
as_options = DefaultOptOptions
as_options["local"] = sset.local
as_options["localswitch"] = sset.localswitch
as_options["errorcheck"] = None
as_options["multistart"] = sset.mstarttype
as_options["lmethod"] = sset.opt
try:
    as_options["method"] = sset.gopt
except:
    pass

### MODEL SAMPLERS ###
N_m = oset.N_m
if(sample_type == 'SC'):
    jump = oset.scjump
    N_m = oset.scN_m
    sampler_m = CollocationSampler(np.array([x_init]), N=N_m, 
                          name='model', 
                          xlimits=xlimits, 
                          probability_functions=pdfs,
                          external_only=external_only)
elif(sample_type == 'Adaptive'):
    sampler_m = AdaptiveSampler(np.array([x_init]), N=N_m, 
                          name='model', 
                          criteria=sset, #give it the whole settings object
                          max_batch=max_batch,
                          xlimits=xlimits, 
                          probability_functions=pdfs, 
                          as_options=as_options,
                          retain_uncertain_points=retain_uncertain_points,
                          external_only=external_only)
else:
    sampler_m = RobustSampler(np.array([x_init]), N=N_m, 
                          name='model', 
                          xlimits=xlimits, 
                          probability_functions=pdfs, 
                          retain_uncertain_points=retain_uncertain_points,
                          external_only=external_only)



### TRUTH OPENMDAO SETUP ###
probt = om.Problem()
probt.model.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
probt.model.dvs.add_output("x_d", val=x_init)

probt.model.add_subsystem("stat", 
                                  StatCompComponent(
                                  sampler=prob_truth,
                                  stat_type="mu_sigma", 
                                  pdfs=pdfs, 
                                  eta=eta_use, 
                                  func=func,
                                  name=name))
# doesn't need a driver
probt.driver = om.pyOptSparseDriver(optimizer= 'SNOPT') #Default: SLSQP
probt.driver.opt_settings = opt_settings
probt.driver = om.ScipyOptimizeDriver(optimizer='SLSQP') 
probt.model.connect("x_d", "stat.x_d")
probt.model.add_design_var("x_d", lower=xlimits_d[:,0], upper=xlimits_d[:,1])
# probt.driver = om.ScipyOptimizeDriver(optimizer='CG') 
probt.model.add_objective("stat.musigma")
probt.setup()
# probt.run_model()

### MODEL OPENMDAO SETUP ###
dvdict = OrderedDict()
probm = om.Problem()
probm.model.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
probm.model.dvs.add_output("x_d", val=x_init)

probm.model.add_subsystem("stat", StatCompComponent(sampler=sampler_m,
                                 stat_type="mu_sigma", 
                                 pdfs=pdfs, 
                                 eta=eta_use, 
                                 func=func,
                                 surrogate=msur,
                                 full_space=full_surrogate,
                                 name=name,
                                 print_surr_plots=print_plots))
# probm.driver = om.pyOptSparseDriver(optimizer='PSQP') #Default: SLSQP
# probm.driver.opt_settings = opt_settings
probm.driver = om.ScipyOptimizeDriver(optimizer='SLSQP') 
# probm.driver = om.ScipyOptimizeDriver(optimizer='CG') 
probm.model.connect("x_d", "stat.x_d")
probm.model.add_design_var("x_d", lower=xlimits_d[:,0], upper=xlimits_d[:,1])
probm.model.add_objective("stat.musigma")
probm.setup()

### CONVERGENCE PLOTS ###
niter = len(realizations)

if not skip:
    ### TRUST RADIUS PLOT ###
    plt.plot(realizations, radii, 'o')
    plt.xlabel(r'Number of Realizations')
    plt.ylabel(r'Trust Radius')
    plt.savefig(f"/{root}/{path}/{title}/plots/trust_radius.png", bbox_inches="tight")
    plt.clf()



    ### LHS RHS CONVERGENCE
    # get end optimization if used
    ### OVERRIDE
    if t_dim == 3:
        probt.set_val('stat.x_d', prob_truth.design_history[1])
        probt.run_driver()
        prob_truth.iter_max = probt.model.iter_count - 1
        prob_truth.design_history = prob_truth.design_history[-prob_truth.iter_max:]
    if t_dim == 4:
        probt.set_val('stat.x_d', prob_truth.design_history[-2])
        probt.run_driver()
        prob_truth.iter_max = probt.model.iter_count - 1
        prob_truth.design_history = prob_truth.design_history[-prob_truth.iter_max:]
    if t_dim == 2:
        probt.set_val('stat.x_d', prob_truth.design_history[-2])
        probt.run_driver()
        prob_truth.iter_max = probt.model.iter_count - 1
        prob_truth.design_history = prob_truth.design_history[-prob_truth.iter_max:]

    gerrm_post = np.zeros(prob_truth.iter_max)
    realizations_post = np.zeros(prob_truth.iter_max)
    for i in range(1, prob_truth.iter_max + 1):
        probt.set_val('stat.x_d', prob_truth.design_history[i])
        probt.run_model()
        gmod = probt.compute_totals(return_format='array')
        gerrm_post[i-1] = np.linalg.norm(gmod)
        realizations_post[i-1] = prob_truth.current_samples['x'].shape[0]*(i-1)

    ind = copy.deepcopy(prob_truth.iter_max)

    ### REDO FULL OPTIMIZATION 
    probt = om.Problem()
    probt.model.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
    probt.model.dvs.add_output("x_d", val=x_init)

    probt.model.add_subsystem("stat", 
                                      StatCompComponent(
                                      sampler=sampler_t,
                                      stat_type="mu_sigma", 
                                      pdfs=pdfs, 
                                      eta=eta_use, 
                                      func=func,
                                      name=name))
    # doesn't need a driver
    probt.driver = om.pyOptSparseDriver(optimizer= 'SNOPT') #Default: SLSQP
    probt.driver.opt_settings = opt_settings
    probt.driver = om.ScipyOptimizeDriver(optimizer='SLSQP') 
    probt.model.connect("x_d", "stat.x_d")
    probt.model.add_design_var("x_d", lower=xlimits_d[:,0], upper=xlimits_d[:,1])
    # probt.driver = om.ScipyOptimizeDriver(optimizer='CG') 
    probt.model.add_objective("stat.musigma")
    probt.setup()
    probt.run_driver()
    x_opt_true = copy.deepcopy(probt.get_val("stat.x_d")[0])


    # plot conv
    full_calls = probt.model.stat.func_calls
    full_calls = full_calls[:-1]
    full_objs = probt.model.stat.objs
    full_objs = full_objs[:-1]
    full_grads = probt.model.stat.grads
    full_grad_norms = []
    for i in range(len(full_grads)):
        full_grad_norms.append(np.linalg.norm(full_grads[i]))
    while len(full_objs) > len(full_grad_norms):
        full_grad_norms.insert(-2, full_grad_norms[-2])
    plt.rcParams["figure.figsize"] = (10,3.5)
    for i in range(niter-1):
        plt.plot(realizations[i] + reflog[i][:,2], reflog[i][:,1], 'r')
        plt.plot(realizations[i] + reflog[i][:,2], reflog[i][:,0], 'b')
        plt.axvline(realizations[i], color='k', linestyle='--', linewidth=1.2)
    if prob_truth.iter_max > 0:
        plt.plot(realizations[i] + reflog[i][-1,2] + realizations_post, 
                    gerrm_post, 'g', label=r'$\|\hat{S}^T_k(x_d)\|$ (SLSQP post)')

    true_fm = copy.deepcopy(probt.model.stat.objs[-1])
    plt.plot(full_calls, full_grad_norms, '-m', label=r'$\|\hat{S}^T_k(x_d)\|$ (SLSQP full)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Number of Realizations')
    plt.ylabel(r'Gradient Magnitude')
    plt.grid()
    plt.plot([],[], 'r', label=r'$\|\hat{S}^M_k(x_d)\|$')
    plt.plot([],[], 'b', label=r'$\bar{\beta}_M$')
    # plt.legend(fontsize='11')
    plt.savefig(f"/{root}/{path}/{title}/plots/lhs_rhs.png", bbox_inches="tight")
    plt.clf()
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

loc[0] = loc[0]['dvs.x_d']
loc = loc[:-1]
locval = []
# reevaluate locs
for i in range(len(loc)):
    probm.model.stat.surrogate = models[i]
    probm.set_val("stat.x_d", loc[i])
    probm.run_model()
    locval.append(copy.deepcopy(probm.get_val("stat.musigma")))
print(loc)
print(locval)
### MODEL EVOLUTION
# Original Func

# design space plots
if d_dim == 1:
    ndir = 120
    x = np.linspace(xlimits[prob_truth.x_d_ind][0,0], xlimits[prob_truth.x_d_ind][0,1], ndir)
    yt = np.zeros([ndir])
    for j in range(ndir):
        probt.set_val("stat.x_d", x[j])
        probt.run_model()
        yt[j] = probt.get_val("stat.musigma")
    minind = np.argmin(yt)
    ym = np.zeros([niter, ndir])
    plt.plot(x, yt, label='objective')
    plt.axvline(x[minind], color='k', linestyle='--', linewidth=1.2)
    plt.xlabel(r'$x_d$')
    plt.ylabel(r'$\hat{S}(x_d)$')
    plt.savefig(f"/{root}/{path}/{title}/plots/true_int.png", bbox_inches="tight")
    plt.clf()
    for i in range(niter):
        probm.model.stat.surrogate = models[i]
        for j in range(ndir):
            probm.set_val("stat.x_d", x[j])
            probm.run_model()
            ym[i,j] = probm.get_val("stat.musigma")
        # plt.autoscale(True)
        plt.plot(x, yt, label='objective')
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt.plot(x, ym[i,:], label=f'model {i}')
        plt.axvline(loc[i], color='b', linestyle='-', linewidth=1.2)
        plt.axvline(loc[i]+radii[i], color='b', linestyle='-', linewidth=1.0)
        plt.axvline(loc[i]-radii[i], color='b', linestyle='-', linewidth=1.0)
        if i < niter - 1:
            plt.axvline(loc[i+1], color='g', linestyle='-', linewidth=1.4)
        plt.axvline(x[minind], color='k', linestyle='--', linewidth=1.2)

        fac = 1
        if i > 0:
            fac = 2
        for line in loc[:i+fac]:
            plt.axvline(line[0], color='k')
        plt.scatter(loc[:i+fac], locval[:i+fac], color='r', marker='x')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(r'$x_d$')
        plt.ylabel(r'$\hat{S}(x_d)$')
        plt.legend(fontsize='13')
        plt.savefig(f"/{root}/{path}/{title}/plots/model_evo_int_{i}.png", bbox_inches="tight")
        plt.clf()

# full space plots
if t_dim == 2:
    ndir = 40
    x = np.linspace(xlimits[prob_truth.x_d_ind][0,0], xlimits[prob_truth.x_d_ind][0,1], ndir)
    y = np.linspace(xlimits[prob_truth.x_u_ind][0,0], xlimits[prob_truth.x_u_ind][0,1], ndir)
    X, Y = np.meshgrid(x, y)
    FT = np.zeros([ndir,ndir])
    for i in range(ndir):
        for j in range(ndir):
            xi = np.zeros([1,2])
            xi[0,prob_truth.x_d_ind] = x[i]
            xi[0,prob_truth.x_u_ind] = y[j]
            FT[i,j] = func(xi)
    
    cs = plt.contourf(X, Y, FT, levels = 15)
    plt.xlabel(r'$x_d$')
    plt.ylabel(r'$x_u$')
    plt.colorbar(cs, label=r'$F(x_d, x_u)$')
    plt.savefig(f"/{root}/{path}/{title}/plots/true_full.png", bbox_inches="tight")
    plt.clf()
    
    for k in range(niter):
        probm.model.stat.surrogate = models[k]
        FM = np.zeros([ndir, ndir]) 
        for i in range(ndir):
            for j in range(ndir):
                xi = np.zeros([1,2])
                xi[0,prob_truth.x_d_ind] = x[i]
                xi[0,prob_truth.x_u_ind] = y[j]
                FM[i,j] = models[k].predict_values(xi)
        plt.contourf(X, Y, FM, levels = 15)
        plt.scatter(models[k].training_points[None][0][0][:,prob_truth.x_d_ind[0]], 
                    models[k].training_points[None][0][0][:,prob_truth.x_u_ind[0]], color='r', marker='x')
        fac = 1
        if k > 0:
            fac = 2
        for line in loc[:k+fac]:
            plt.axvline(line[0], color='k')
        plt.xlabel(r'$x_d$')
        plt.ylabel(r'$x_u$')
        plt.colorbar(cs, label=r'$\hat{f}(x_d, x_u)$')
        plt.savefig(f"/{root}/{path}/{title}/plots/model_evo_full_{k}.png", bbox_inches="tight")
        plt.clf()
### ORIGINAL FUNCTION PLOT ###
# ndir = 600
# x = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
# y = np.zeros([ndir])
# for j in range(ndir):
#     probt.set_val("stat.x_d", x[j])
#     probt.run_model()
#     y[j] = probt.get_val("stat.musigma")
# minind = np.argmin(y)
# plt.plot(x, y, label='objective')
# print(f"x* = {x[minind]}")
# print(f"y* = {y[minind]}")
# plt.axvline(x[minind], color='k', linestyle='--', linewidth=1.2)
# plt.xlabel(r'$x_d$')
# plt.ylabel(r'$S(x_d)$')
# plt.savefig(f"./{name}/objrobust_true.pdf", bbox_inches="tight")

# import pdb; pdb.set_trace()


# x_opt_1 = copy.deepcopy(probm.model.get_val("x_d")[0])

# func_calls_t = probt.model.stat.func_calls
# func_calls = probm.model.stat.func_calls
# objs_t = probt.model.stat.objs
# objs = probm.model.stat.objs
# xds_t = probt.model.stat.xps
# xds = probm.model.stat.xps

# outer = sub_optimizer.outer_iter
# jumps = sub_optimizer.break_iters

# bigsum = 0
# for i in range(outer+1):
#     if i == 0:
#         ind0 = 0
#     else:
#         ind0 = sum(jumps[:i])
#     ind1 = sum(jumps[:i+1])

#     plt.plot(list(np.asarray(func_calls[ind0:ind1+1])+bigsum), objs[ind0:ind1+1], linestyle='-', marker='s', color='b', label='model')
#     plt.plot([func_calls[ind1]+bigsum,func_calls_t[i]+func_calls[ind1]], 
#              [objs[ind1], objs_t[i]], linestyle='-', marker='s', color='orange', label='validation')
#     bigsum = func_calls_t[i]

# # plt.xscale("log")
# plt.xlabel(r"Function Calls")
# plt.ylabel(r"$\mu_f(x_d)$")
# plt.legend()

# plt.savefig(f"./{name}/convrobust1_true.pdf", bbox_inches="tight")

# import pdb; pdb.set_trace()

# plt.clf()

# for i in range(outer+1):
#     if i == 0:
#         ind0 = 0
#     else:
#         ind0 = sum(jumps[:i])
#     ind1 = sum(jumps[:i+1])

#     plt.plot(xds[ind0:ind1+1], objs[ind0:ind1+1], linestyle='-', marker='s',color='b')
#     plt.plot(xds_t[i], objs_t[i], linestyle='-', marker='s', color='orange')


# plt.xlabel(r"$x_d$")
# plt.ylabel(r"$\mu_f(x_d)$")

# plt.axvline(x_init, color='k', linestyle='--', linewidth=1.2)
# plt.axvline(x_opt_1, color='r', linestyle='--', linewidth=1.2)

# ndir = 150
# x = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
# y = np.zeros([ndir])
# for j in range(ndir):
#     probt.set_val("stat.x_d", x[j])
#     probt.run_model()
#     y[j] = probt.get_val("stat.musigma")
# # Plot original function
# plt.plot(x, y, label='objective')

# plt.savefig(f"./{name}/objrobust1_true.pdf", bbox_inches="tight")



# plt.clf()


# import pdb; pdb.set_trace()



# # plot conv
# cs = plt.plot(probm.model.stat.func_calls, probm.model.stat.objs)
# plt.xlabel(r"Number of function calls")
# plt.ylabel(r"$\mu_f(x_d)$")
# #plt.legend(loc=1)
# plt.savefig(f"./{name}/convergence_model_nosurr.pdf", bbox_inches="tight")
# plt.clf()

# import pdb; pdb.set_trace()
# # plot robust func
# ndir = 150


# # Plot original function
# cs = plt.plot(x, y)
# plt.xlabel(r"$x_d$")
# plt.ylabel(r"$\mu_f(x_d)$")
# plt.axvline(x_init, color='k', linestyle='--', linewidth=1.2)
# plt.axvline(x_opt_1, color='r', linestyle='--', linewidth=1.2)
# #plt.legend(loc=1)
# plt.savefig(f"./robust_opt_subopt_plots/objrobust1_true.pdf", bbox_inches="tight")
# plt.clf()

# # plot beta dist
# x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
# y = np.zeros([ndir])
# beta_handle = beta(pdfs[0][1],pdfs[0][2])
# for j in range(ndir):
#     y[j] = beta_handle.pdf(x[j])
# cs = plt.plot(x, y)
# plt.xlabel(r"$x_d$")
# plt.ylabel(r"$\mu_f(x_d)$")
# plt.axvline(0.75, color='r', linestyle='--', linewidth=1.2)
# #plt.legend(loc=1)
# plt.savefig(f"./robust_opt_subopt_plots/betadist1_true.pdf", bbox_inches="tight")
# plt.clf()




""" 
raw optimization section
"""

# probt.run_driver()

# x_opt_true = copy.deepcopy(probt.get_val("stat.x_d")[0])

# plot conv
# cs = plt.plot(probt.model.stat.func_calls, probt.model.stat.objs)
# plt.xlabel(r"Number of function calls")
# plt.ylabel(r"$\mu_f(x_d)$")
# #plt.legend(loc=1)
# plt.savefig(f"./{name}/convergence_truth.pdf", bbox_inches="tight")
# plt.clf()

# true_fm = copy.deepcopy(probt.model.stat.objs[-1])

# probt.set_val("stat.x_d", x_init)
""" 
raw optimization section
"""