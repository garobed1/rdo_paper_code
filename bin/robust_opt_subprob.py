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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rcParams['font.size'] = '16'

"""
run a mean plus variance optimization over the 1D-1D test function, test out the sampling techniques
"""


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--optfile', action='store', default='ouu_opt_settings.py', help = 'python file containing settings for optimization parameters')
parser.add_argument('-s', '--samplingfile', action='store', default='ouu_sam_settings.py', help = 'python file containing settings for adaptive sampling parameters')

args = parser.parse_args()
osetfile = args.optfile
ssetfile = args.samplingfile


# import settings from given config files
root = os.getcwd()
# optimization imports
optsplit = osetfile.split(sep='/')
optuse = '.'.join(optsplit)
if optuse.endswith('.py'):
    optsuse = optuse.split('.')[:-1]
    optuse = '.'.join(optuse)
oset = importlib.import_module(optuse)
title = f"{oset.name}_{oset.prob}_U{oset.u_dim}D_D{oset.d_dim}D"
if(oset.path == None):
    path = "."
else:
    path = oset.path

# adaptive sampling imports, ignore path from here since it should be in oset
samsplit = ssetfile.split(sep='/')
suse = '.'.join(samsplit)
if suse.endswith('.py'):
    suse = suse.split('.')[:-1]
    suse = '.'.join(suse)
sset = importlib.import_module(suse)

if rank == 0:
    if not os.path.isdir(f"/{root}/{path}/{title}"):
        os.mkdir(f"/{root}/{path}/{title}")
    shutil.copy(f"./{osetfile}", f"/{root}/{path}/{title}/opt_settings.py")
    shutil.copy(f"./{ssetfile}", f"/{root}/{path}/{title}/sam_settings.py")



##### Adaptive Surrogate UQ Parameters #####
RC0 = None
max_batch = sset.max_batch

# adaptive sampling options
options = DefaultOptOptions
options["local"] = sset.local
options["localswitch"] = sset.localswitch
options["errorcheck"] = None
options["multistart"] = sset.mstarttype
options["lmethod"] = sset.opt
try:
    options["method"] = sset.gopt
except:
    pass













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
eta_use = oset.eta_use
use_truth_to_train = oset.use_truth_to_train
ref_strategy = oset.ref_strategy

# start at just one point for now
x_init = oset.x_init

# optimization
# args = (func, eta_use)
opt_settings = oset.opt_settings
max_outer = oset.max_outer
trust_region_bound = oset.trust_region_bound
initial_trust_radius = oset.initial_trust_radius
inexact_gradient_only = oset.exact_gradient_only

### SAMPLING STRATEGY ###
sample_type = oset.sample_type
N_t = oset.N_t
N_m = oset.N_m
if(sample_type == 'SC'):
    jump = oset.scjump
    N_t = oset.scN_t
    N_m = oset.scN_m
    sampler_t = CollocationSampler(np.array([x_init]), N=N_t, 
                          name='truth', 
                          xlimits=xlimits, 
                          probability_functions=pdfs)
    sampler_m = CollocationSampler(np.array([x_init]), N=N_m, 
                          name='model', 
                          xlimits=xlimits, 
                          probability_functions=pdfs,
                          external_only=external_only)
elif(sample_type == 'Adaptive'):
    sampler_t = RobustSampler(np.array([x_init]), N=N_t, 
                          name='truth', 
                          xlimits=xlimits, 
                          probability_functions=pdfs, 
                          retain_uncertain_points=retain_uncertain_points)
    sampler_m = AdaptiveSampler(np.array([x_init]), N=N_m, 
                          name='model', 
                          criteria=RC0,
                          max_batch=max_batch,
                          xlimits=xlimits, 
                          probability_functions=pdfs, 
                          retain_uncertain_points=retain_uncertain_points)
else:
    sampler_t = RobustSampler(np.array([x_init]), N=N_t, 
                          name='truth', 
                          xlimits=xlimits, 
                          probability_functions=pdfs, 
                          retain_uncertain_points=retain_uncertain_points)
    sampler_m = RobustSampler(np.array([x_init]), N=N_m, 
                          name='model', 
                          xlimits=xlimits, 
                          probability_functions=pdfs, 
                          retain_uncertain_points=retain_uncertain_points,
                          external_only=external_only)

# get variable bounds from sampler_t
xlimits_d = xlimits[sampler_t.x_d_ind]
xlimits_u = xlimits[sampler_t.x_u_ind]

# set up surrogates #NOTE: not doing it for truth for now
#TODO: WRITE SURROGATE PICKER AND RC0 PICKER APPS
msur = None
if use_surrogate:
    rscale = 5.5
    if hasattr(sset, 'rscale'):
        rscale = sset.rscale
    rho = 10 
    if hasattr(sset, 'rho'):
        rscale = sset.rho
    min_contributions = 1e-12

    if(full_surrogate):
        neval = 1+(t_dim+2)
        msur = POUHessian(bounds=xlimits)
    else:
        neval = 1+(u_dim+2)
        msur = POUHessian(bounds=xlimits_u)

    msur.options.update({"rscale":rscale})
    msur.options.update({"rho":rho})
    msur.options.update({"neval":neval})
    msur.options.update({"min_contribution":min_contributions})
    msur.options.update({"print_prediction":False})






### TRUTH ###
probt = om.Problem()
probt.model.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
probt.model.dvs.add_output("x_d", val=x_init)

probt.model.add_subsystem("stat", StatCompComponent(sampler=sampler_t,
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
probt.model.add_design_var("x_d", lower=xlimits_d[0,0], upper=xlimits_d[0,1])
# probt.driver = om.ScipyOptimizeDriver(optimizer='CG') 
probt.model.add_objective("stat.musigma")


# probt.setup()
# probt.run_model()

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
### MODEL ###
dvdict = OrderedDict()

# import pdb; pdb.set_trace()
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
probm.driver = om.pyOptSparseDriver(optimizer='PSQP') #Default: SLSQP
# probm.driver.opt_settings = opt_settings
# probm.driver = om.ScipyOptimizeDriver(optimizer='SLSQP') 
# probm.driver = om.ScipyOptimizeDriver(optimizer='CG') 
probm.model.connect("x_d", "stat.x_d")
probm.model.add_design_var("x_d", lower=xlimits_d[0,0], upper=xlimits_d[0,1])
# dvdict = probm.model.get_design_vars()
# probm.setup()

if trust_region_bound == 1:
    # connect all dvs 
    dv_settings = dvdict
    probm.model.add_subsystem('trust', 
                              TrustBound(dv_dict=dv_settings, 
                                            initial_trust_radius=initial_trust_radius), 
                                            promotes_inputs=list(dv_settings.keys()))
    probm.model.add_constraint('trust.c_trust', lower=0.0)
    # probm.model.trust.add_input("x_d", val=x_init)
    # probm.model.trust.set_center(zk)
    # probm.model.connect("x_d", "trust.x_d")

# if 2, handle with changing dv bounds internally DEFAULT BEHAVIOR USE THIS


probm.model.add_objective("stat.musigma")


if trust_region_bound: #anything but 0
    sub_optimizer = UncertainTrust(prob_model=probm, prob_truth=probt, 
                                    initial_trust_radius=initial_trust_radius,
                                    trust_option=trust_region_bound,
                                    flat_refinement=jump, 
                                    max_iter=max_outer,
                                    use_truth_to_train=use_truth_to_train,
                                    inexact_gradient_only=inexact_gradient_only,
                                    ref_strategy=ref_strategy)
else:
    sub_optimizer = SequentialFullSolve(prob_model=probm, prob_truth=probt, 
                                    flat_refinement=jump, 
                                    max_iter=max_outer,
                                    use_truth_to_train=use_truth_to_train,
                                    ref_strategy=ref_strategy)
sub_optimizer.setup_optimization()
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
# import pdb; pdb.set_trace()
# play around with values/model runs here
sub_optimizer.prob_truth.set_val("stat.x_d", x_init)
sub_optimizer.prob_model.set_val("stat.x_d", x_init)
# probm.setup()
sub_optimizer.prob_truth.set_val("x_d", x_init)
sub_optimizer.prob_model.set_val("x_d", x_init)
# om.n2(probm)
sub_optimizer.prob_truth.run_model()
sub_optimizer.prob_model.run_model()


sub_optimizer.solve_full()

x_opt_1 = copy.deepcopy(probm.model.get_val("x_d")[0])

func_calls_t = probt.model.stat.func_calls
func_calls = probm.model.stat.func_calls
objs_t = probt.model.stat.objs
objs = probm.model.stat.objs
xds_t = probt.model.stat.xps
xds = probm.model.stat.xps

outer = sub_optimizer.outer_iter
jumps = sub_optimizer.break_iters

bigsum = 0
for i in range(outer+1):
    if i == 0:
        ind0 = 0
    else:
        ind0 = sum(jumps[:i])
    ind1 = sum(jumps[:i+1])

    plt.plot(list(np.asarray(func_calls[ind0:ind1+1])+bigsum), objs[ind0:ind1+1], linestyle='-', marker='s', color='b', label='model')
    plt.plot([func_calls[ind1]+bigsum,func_calls_t[i]+func_calls[ind1]], 
             [objs[ind1], objs_t[i]], linestyle='-', marker='s', color='orange', label='validation')
    bigsum = func_calls_t[i]

# plt.xscale("log")
plt.xlabel(r"Function Calls")
plt.ylabel(r"$\mu_f(x_d)$")
plt.legend()

plt.savefig(f"./{name}/convrobust1_true.pdf", bbox_inches="tight")

import pdb; pdb.set_trace()

plt.clf()

for i in range(outer+1):
    if i == 0:
        ind0 = 0
    else:
        ind0 = sum(jumps[:i])
    ind1 = sum(jumps[:i+1])

    plt.plot(xds[ind0:ind1+1], objs[ind0:ind1+1], linestyle='-', marker='s',color='b')
    plt.plot(xds_t[i], objs_t[i], linestyle='-', marker='s', color='orange')


plt.xlabel(r"$x_d$")
plt.ylabel(r"$\mu_f(x_d)$")

plt.axvline(x_init, color='k', linestyle='--', linewidth=1.2)
plt.axvline(x_opt_1, color='r', linestyle='--', linewidth=1.2)

ndir = 150
x = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
y = np.zeros([ndir])
for j in range(ndir):
    probt.set_val("stat.x_d", x[j])
    probt.run_model()
    y[j] = probt.get_val("stat.musigma")
# Plot original function
plt.plot(x, y, label='objective')

plt.savefig(f"./{name}/objrobust1_true.pdf", bbox_inches="tight")



plt.clf()


import pdb; pdb.set_trace()



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




