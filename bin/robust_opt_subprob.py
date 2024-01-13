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
from functions.smt_wrapper import SMTComponent
from optimization.robust_objective import RobustSampler, CollocationSampler, AdaptiveSampler
from optimization.defaults import DefaultOptOptions
import argparse
from mpi4py import MPI
import pickle

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












############################
#### SCRIPT BEGINS HERE ####
############################
# import settings from given config files
root = os.getcwd()
# optimization imports
optsplit = osetfile.split(sep='/')
optuse = '.'.join(optsplit)
if optuse.endswith('.py'):
    optuse = optuse.split('.')[:-1]
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


# copy over settings, and grab previous iteration index if it exists
k = 0
if rank == 0:
    if not os.path.isdir(f"/{root}/{path}/{title}"):
        os.mkdir(f"/{root}/{path}/{title}")
    shutil.copy(f"{osetfile}", f"/{root}/{path}/{title}/opt_settings.py")
    shutil.copy(f"{ssetfile}", f"/{root}/{path}/{title}/sam_settings.py")

    # also, grab the last iteration
    if os.path.isfile(f'{path}/{title}/previous_iter.pickle'):
        with open(f'{path}/{title}/previous_iter.pickle', 'rb') as f:
            k = pickle.load(f)
k = comm.bcast(k)

name = oset.name
print_plots = oset.print_plots

# variables
u_dim = oset.u_dim
d_dim = oset.d_dim
external_only = (oset.use_truth_to_train and oset.use_surrogate) #NEW
pdfs = oset.pdfs
t_dim = u_dim + d_dim

# get robust function
func = GetProblem(oset.prob, t_dim, use_design=True)
xlimits = func.xlimits
try:
    p_con = oset.p_con
    p_eq = oset.p_eq
    p_ub = oset.p_ub
    p_lb = oset.p_lb
except:
    p_con = False
    p_eq = None
    p_ub = None
    p_lb = None

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
xi = oset.xi
try:
    tol_ignore_sdist = oset.tol_ignore_sdist
except:
    tol_ignore_sdist = False

### SAMPLING STRATEGY ###
sample_type_t = oset.sample_type_t
sample_type_m = oset.sample_type

### TRUTH SAMPLERS ###
N_t = oset.N_t
if(sample_type_t == 'SC'):
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
min_contribution = 1e-14
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
        msur = POUHessian(bounds=xlimits, rscale = 5.5, neval = t_dim+3, min_contribution = min_contribution)
    else:
        sdim = u_dim
        msur = POUHessian(bounds=xlimits_u, rscale = 5.5, neval = u_dim+3, min_contribution = min_contribution)

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
if(sample_type_m == 'SC'):
    jump = oset.scjump
    N_m = oset.scN_m
    sampler_m = CollocationSampler(np.array([x_init]), N=N_m, 
                          name='model', 
                          xlimits=xlimits, 
                          probability_functions=pdfs,
                          external_only=external_only)
elif(sample_type_m == 'Adaptive'):
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
                                  sampler=sampler_t,
                                  stat_type="mu_sigma", 
                                  pdfs=pdfs, 
                                  eta=eta_use, 
                                  func=func,
                                  name=name))
# doesn't need a driver
try:
    import pyoptsparse
    probt.driver = om.pyOptSparseDriver(optimizer='IPOPT') 
except:
    probt.driver = om.ScipyOptimizeDriver(optimizer='SLSQP') 
    probt.driver.opt_settings = opt_settings
probt.model.connect("x_d", "stat.x_d")
probt.model.add_design_var("x_d", lower=xlimits_d[:,0], upper=xlimits_d[:,1])
# probt.driver = om.ScipyOptimizeDriver(optimizer='CG') 


### NOTE NOTE NOTE ###
### TEMPORARY EXECCOMP FOR CONSTRAINED PROBLEM ###
if oset.prob == "toylinear":
    excomp = om.ExecComp('y = 10-x')
elif oset.prob == "uellipse_loc": #assume rosenbrock
    excomp = SMTComponent("rosenbrock", dim_base=2)
elif oset.prob == "uellipse_rad": #assume rosenbrock
    excomp = SMTComponent("rosenbrock", dim_base=2)
elif oset.prob == "betatestex":
    excomp = om.ExecComp('y = x*0.2')
else:
    excomp = om.ExecComp('y = x*0.2')

if p_con:
    probt.model.add_constraint("stat.musigma", lower=p_lb, upper=p_ub, equals=p_eq)
    probt.model.add_subsystem('comp_obj', excomp)
    probt.model.connect('x_d', 'comp_obj.x')
    probt.model.add_objective("comp_obj.y")
else:
    probt.model.add_objective("stat.musigma")
probt.setup()
probt.run_model()




### ORIGINAL FUNCTION PLOT ###
# ndir = 200
# x = np.linspace(xlimits_d[0,0], xlimits_d[0,1], ndir)
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
# plt.savefig(f"/{root}/{path}/{title}/objrobust_true.pdf", bbox_inches="tight")
# plt.clf()
# import pdb; pdb.set_trace()

""" 
raw optimization section
"""

# probt.run_driver()

# x_opt_true = copy.deepcopy(probt.get_val("stat.x_d")[0])

# # plot conv
# cs = plt.plot(probt.model.stat.func_calls, probt.model.stat.objs)
# plt.xlabel(r"Number of function calls")
# plt.ylabel(r"$\mu_f(x_d)$")
# #plt.legend(loc=1)
# plt.savefig(f"/{root}/{path}/{title}/convergence_truth.png", bbox_inches="tight")
# plt.clf()

# true_fm = copy.deepcopy(probt.model.stat.objs[-1])

# probt.set_val("stat.x_d", x_init)
# import pdb; pdb.set_trace()
""" 
raw optimization section
"""


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
try:
    import pyoptsparse
    probm.driver = om.pyOptSparseDriver(optimizer='IPOPT') 
except:
    probm.driver = om.ScipyOptimizeDriver(optimizer='SLSQP') 
    probm.driver.opt_settings = opt_settings
probm.model.connect("x_d", "stat.x_d")
probm.model.add_design_var("x_d", lower=xlimits_d[:,0], upper=xlimits_d[:,1])
if p_con:
    probm.model.add_constraint("stat.musigma", lower=p_lb, upper=p_ub, equals=p_eq)
    probm.model.add_subsystem('comp_obj', excomp)
    probm.model.connect('x_d', 'comp_obj.x')
    probm.model.add_objective("comp_obj.y")
else:
    probm.model.add_objective("stat.musigma")

#TODO:: ADD CONSTRAINTS NOW

if trust_region_bound: #anything but 0
    sub_optimizer = UncertainTrust(prob_model=probm, prob_truth=probt, 
                                    title=title,
                                    path='/' + root + '/' + path,
                                    xi=xi,
                                    initial_trust_radius=initial_trust_radius,
                                    trust_option=trust_region_bound,
                                    flat_refinement=jump, 
                                    max_iter=max_outer,
                                    use_truth_to_train=use_truth_to_train,
                                    inexact_gradient_only=inexact_gradient_only,
                                    ref_strategy=ref_strategy,
                                    model_grad_err_est=approximate_model,
                                    truth_func_err_est=approximate_truth,
                                    truth_func_err_est_max=approximate_truth_max,
                                    trust_increase_terminate=trust_increase_terminate,
                                    tol_ignore_sdist=tol_ignore_sdist)
else:
    sub_optimizer = SequentialFullSolve(prob_model=probm, prob_truth=probt, 
                                    title=title,
                                    path='/' + root + '/' + path,
                                    flat_refinement=jump, 
                                    max_iter=max_outer,
                                    use_truth_to_train=use_truth_to_train,
                                    ref_strategy=ref_strategy,
                                    approximate_truth=approximate_truth,
                                    approximate_truth_max=approximate_truth_max)
sub_optimizer.setup_optimization()


### ORIGINAL FUNCTION PLOT ###
# ndir = 150
# x = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
# y = np.zeros([ndir])
# for j in range(ndir):
#     probt.set_val("stat.x_d", x[j])
#     probt.run_model()
#     y[j] = probt.get_val("stat.musigma")
# plt.plot(x, y, label='objective')

# plt.savefig(f"./{name}/objrobust1_true.pdf", bbox_inches="tight")

### INITIALIZE DESIGN ###
sub_optimizer.prob_truth.set_val("stat.x_d", x_init)
sub_optimizer.prob_model.set_val("stat.x_d", x_init)
# probm.setup()
sub_optimizer.prob_truth.set_val("x_d", x_init)
sub_optimizer.prob_model.set_val("x_d", x_init)
# om.n2(probm)
sub_optimizer.prob_truth.run_model()
sub_optimizer.prob_model.run_model()


### INITIALIZE RECORDERS ###

rec_t = om.SqliteRecorder(f'/{root}/{path}/{title}_{k}' + '_truth.sql')
rec_m = om.SqliteRecorder(f'/{root}/{path}/{title}_{k}' + '_model.sql')
sub_optimizer.prob_truth.add_recorder(rec_t)
sub_optimizer.prob_model.add_recorder(rec_m)
sub_optimizer.prob_truth.driver.recording_options['record_inputs'] = True
sub_optimizer.prob_truth.driver.recording_options['record_outputs'] = True
sub_optimizer.prob_truth.driver.recording_options['record_residuals'] = True
sub_optimizer.prob_truth.driver.recording_options['record_derivatives'] = True
sub_optimizer.prob_model.driver.recording_options['record_inputs'] = True
sub_optimizer.prob_model.driver.recording_options['record_outputs'] = True
sub_optimizer.prob_model.driver.recording_options['record_residuals'] = True
sub_optimizer.prob_model.driver.recording_options['record_derivatives'] = True


### PERFORM OPTIMIZATION
succ, fail = sub_optimizer.solve_full()
### PERFORM OPTIMIZATION

# if fail > 0:
if 1:
    sub_optimizer.prob_truth.set_val("x_d", sub_optimizer.result_cur)
    sub_optimizer.prob_truth.run_driver()


### PICKLE RIGHT AWAY
if rank == 0:
    with open(f'/{root}/{path}/{title}/grad_lhs.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.grad_lhs, f)
    with open(f'/{root}/{path}/{title}/grad_rhs.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.grad_rhs, f)
    with open(f'/{root}/{path}/{title}/radii.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.radii, f)
    with open(f'/{root}/{path}/{title}/realizations.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.realizations, f)
    with open(f'/{root}/{path}/{title}/areds.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.areds, f)
    with open(f'/{root}/{path}/{title}/preds.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.preds, f)
    with open(f'/{root}/{path}/{title}/loc.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.loc, f)
    with open(f'/{root}/{path}/{title}/models.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.models, f)
    with open(f'/{root}/{path}/{title}/reflog.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.reflog, f)
    with open(f'/{root}/{path}/{title}/prob_truth.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.prob_truth.model.stat.sampler, f)
    with open(f'/{root}/{path}/{title}/prob_model_history.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.prob_model.model.stat.sampler.history, f)


# ### CONVERGENCE PLOTS ###
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




