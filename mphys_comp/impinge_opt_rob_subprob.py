import imp
import pickle
import numpy as np
import argparse
from mpi4py import MPI
import os,sys, copy
import matplotlib.pyplot as plt
import openmdao.api as om
import importlib, shutil

from functions.shock_problem import ImpingingShock
from beam.mphys_eb import Top as Top_EB
from optimization.robust_objective import RobustSampler, CollocationSampler, AdaptiveSampler
from uq_comp.stat_comp_comp import StatCompComponent
from optimization.opt_subproblem import SequentialFullSolve
from optimization.opt_trust_uncertain import UncertainTrust
from surrogate.pougrad import POUHessian
from collections import OrderedDict
from optimization.defaults import DefaultOptOptions

# these imports will be from the respective codes' repos rather than mphys
#from mphys.mphys_meld import MeldBuilder
#from mphys.mphys_rlt import RltBuilder

from baseclasses import AeroProblem

# from tacs import elements, constitutive, functions

# contains all options, aero, opt, struct, uq, warp
import mphys_comp.impinge_setup as default_impinge_setup

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# rank += 1
# use as scratch space for playing around


### PARAMS ###
# nsweep = 12
# s_list = 4*[25., 25., 25., 25., 22., 22., 22., 22., 27., 27., 27., 27.]
# M0_list = 4*[1.4, 1.8, 2.2, 2.6, 1.4, 1.8, 2.2, 2.6, 1.4, 1.8, 2.2, 2.6]
# # M0_list = 4*[1.4, 2.2, 3.0, 1.4, 2.2, 3.0, 1.4, 2.2, 3.0]
# smax_list = nsweep*[5e5] + nsweep*[1e6] + nsweep*[5e5] +  nsweep*[1e6] 
# E_list = nsweep*[69000000000] +nsweep*[69000000000] + nsweep*[54500000000] + nsweep*[54500000000]

ndv = 4 # number of thickness variables
s = 25. # shock angle
M0 = 1.8 # upstream mach number
smax = 5e5 # max stress constraint
E = 69000000000
ndv = 4
# eta_use = 1.0

# home = '/gpfs/u/home/ODLC/ODLCbdnn/'
# barn = 'barn'
# name = 'test_case_reload'
home = '/home/garobed/'
barn = ''
mesh = f'{home}{barn}/garo-rpi-graduate-work/meshes/imp_mphys_73_73_25.cgns'
# mesh = f'{home}{barn}/garo-rpi-graduate-work/meshes/imp_long_145_145_25.cgns'
# mesh = f'{home}{barn}/garo-rpi-graduate-work/meshes/imp_long_217_217_25.cgns'

# N_t = 2
inputs = ["dv_struct_TRUE", "shock_angle", "M0"]
x_init = np.ones(ndv)*0.006
pdfs = ndv*[0.]
pdfs = pdfs + [['uniform'], ['uniform']]
xlimits = np.zeros([ndv+2, 2])
xlimits[:ndv,0] = 0.0004
xlimits[:ndv,1] = 0.007
xlimits[ndv,0] = 23.
xlimits[ndv,1] = 27.
xlimits[ndv+1,0] = 2.5
xlimits[ndv+1,1] = 2.9

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--optfile', action='store', default='imp_opt_settings.py', help = 'python file containing settings for optimization parameters')
parser.add_argument('-s', '--samplingfile', action='store', default='imp_sam_settings.py', help = 'python file containing settings for adaptive sampling parameters')

args = parser.parse_args()
osetfile = args.optfile
ssetfile = args.samplingfile





# removed settings, now up top
# start at just one point for now
# x_init = oset.x_init
# pdfs = oset.pdfs



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
title = title + f'_ndv{ndv}_smax{smax}_E{E}'
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
    shutil.copy(f"{osetfile}", f"/{root}/{path}/{title}/opt_settings.py")
    shutil.copy(f"{ssetfile}", f"/{root}/{path}/{title}/sam_settings.py")


name = oset.name
print_plots = oset.print_plots

# variables
u_dim = oset.u_dim
d_dim = oset.d_dim
external_only = (oset.use_truth_to_train and oset.use_surrogate) #NEW
t_dim = u_dim + d_dim

# name = 'first_robust_attempt'

# initialize shock function
func = ImpingingShock(input_bounds=xlimits, ndv=ndv, E=E, smax=smax, inputs=inputs, mesh=mesh)

# uq settings
use_surrogate = oset.use_surrogate
full_surrogate = oset.full_surrogate
retain_uncertain_points = oset.retain_uncertain_points
retain_uncertain_points_T = oset.retain_uncertain_points_T
eta_use = oset.eta_use
use_truth_to_train = oset.use_truth_to_train
ref_strategy = oset.ref_strategy



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

### SAMPLING STRATEGY ###
sample_type_t = oset.sample_type_t
sample_type_m = oset.sample_type


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


# prob = om.Problem(comm=MPI.COMM_SELF)
# try:
#     import pyoptsparse
#     prob.driver = om.pyOptSparseDriver(optimizer='IPOPT') 
# except:
#     prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP') 
# prob.model = Top(problem_settings=problem_settings, subsonic=subsonic,
#                                                      use_shock_comp=use_shock, 
#                                                      use_inflow_comp=use_inflow, 
#                                                      full_free=full_far)




probt = om.Problem()
# probt = om.Problem(comm=MPI.COMM_SELF)
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
# probt.driver = om.pyOptSparseDriver(optimizer= 'SNOPT') #Default: SLSQP
try:
    import pyoptsparse
    probt.driver = om.pyOptSparseDriver(optimizer='IPOPT') 
except:
    probt.driver = om.ScipyOptimizeDriver(optimizer='SLSQP') 
# probt.driver.opt_settings = {'ACC': 1e-6}

# probt.driver = om.ScipyOptimizeDriver(optimizer='SLSQP') 
probt.model.connect("x_d", "stat.x_d")
probt.model.add_design_var("x_d", lower=xlimits_d[:,0], upper=xlimits_d[:,1])
# probt.driver = om.ScipyOptimizeDriver(optimizer='CG') 

probt.model.add_constraint("stat.musigma", upper=0.)
probt.model.add_subsystem('mass_only', Top_EB(problem_settings=func.problem_settings))
# probt.model.add_subsystem('comp_obj', EBMass(struct_objects=func.problem_settings))
probt.model.connect('x_d', 'mass_only.dv_interp.DVS')#dv_struct_TRUE')#
probt.model.add_objective("mass_only.test.mass")

probt.setup()



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
probm.model.add_constraint("stat.musigma", upper=0.)
probm.model.add_subsystem('mass_only', Top_EB(problem_settings=func.problem_settings))
# probt.model.add_subsystem('comp_obj', EBMass(struct_objects=func.problem_settings))
probm.model.connect('x_d', 'mass_only.dv_interp.DVS')#dv_struct_TRUE')#
probm.model.add_objective("mass_only.test.mass")




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
                                    trust_increase_terminate=trust_increase_terminate)
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


### TODO: GET RID OF THIS
### INITIALIZE DESIGN ###
sub_optimizer.prob_truth.set_val("stat.x_d", x_init)
sub_optimizer.prob_model.set_val("stat.x_d", x_init)
# probm.setup()
sub_optimizer.prob_truth.set_val("x_d", x_init)
sub_optimizer.prob_model.set_val("x_d", x_init)
# om.n2(probm)
# sub_optimizer.prob_truth.run_model()
sub_optimizer.prob_model.run_model()

# get_last_case = False
# i = 0
# while os.path.isfile(title):
#     i += 1
#     get_last_case = True
#     # check if subsequent file exists
#     title_old = copy.deepcopy(title)
#     title = f'{name}_ndv{ndv}_smax{smax}_E{E}_{i}.sql'

# if get_last_case:
#     cr = om.CaseReader(title_old)
#     last_case = cr.get_case(-1)
#     probt.load_case(last_case)
    # import pdb; pdb.set_trace()

# recorder 
### INITIALIZE RECORDERS ###

rec_t = om.SqliteRecorder(f'/{root}/{path}/{title}' + '_truth.sql')
rec_m = om.SqliteRecorder(f'/{root}/{path}/{title}' + '_model.sql')
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
""" 
raw optimization section
"""

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
    with open(f'/{root}/{path}/{title}/prob_model.pickle', 'wb') as f:
        pickle.dump(sub_optimizer.prob_model.model.stat.sampler, f)
