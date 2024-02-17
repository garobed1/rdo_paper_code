import numpy as np
import argparse
from mpi4py import MPI
import os,sys, copy
import matplotlib.pyplot as plt
import openmdao.api as om

from functions.shock_problem import ImpingingShock
from beam.mphys_eb import Top as Top_EB
from optimization.robust_objective import RobustSampler, CollocationSampler, AdaptiveSampler
from uq_comp.stat_comp_comp import StatCompComponent

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

ndv = 5 # number of thickness variables
s = 25. # shock angle
M0 = 1.8 # upstream mach number
smax = 5e5 # max stress constraint
E = 69000000000
eta_use = 1.0

name = 'ouu_sbli_A_sc'
home = '/gpfs/u/home/ODLC/ODLCbdnn/'
barn = 'barn'
# name = 'test_case_reload'
# home = '/home/garobed/'
# barn = ''


# mesh = f'{home}{barn}/garo-rpi-graduate-work/meshes/imp_mphys_73_73_25.cgns'
# mesh = f'{home}{barn}/garo-rpi-graduate-work/meshes/imp_long_145_145_25.cgns'
mesh = f'{home}{barn}/garo-rpi-graduate-work/meshes/imp_long_217_217_25.cgns'

N_t = 6
inputs = ["dv_struct_TRUE", "shock_angle", "M0"]
x_init = np.ones(ndv)*0.0035
pdfs = ndv*[0.]
# pdfs = pdfs + [['uniform'], ['uniform']]
pdfs = pdfs + [['beta', 5., 2.], ['uniform']]
xlimits = np.zeros([ndv+2, 2])
xlimits[:ndv,0] = 0.0009
xlimits[:ndv,1] = 0.007
xlimits[ndv,0] = 23.
xlimits[ndv,1] = 27.
xlimits[ndv+1,0] = 2.5
xlimits[ndv+1,1] = 2.9

sampler_t = CollocationSampler(np.array([x_init]), N=N_t, 
            name='truth', 
            xlimits=xlimits, 
            probability_functions=pdfs)

xlimits_d = xlimits[sampler_t.x_d_ind]
xlimits_u = xlimits[sampler_t.x_u_ind]


func = ImpingingShock(input_bounds=xlimits, ndv=ndv, E=E, smax=smax, inputs=inputs, mesh=mesh)

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




# set design variables
# prob.model.add_design_var("dv_struct_TRUE", lower = 0.0004, upper = 0.007)
# prob.model.add_design_var("shock_angle")

# set objective
# prob.model.add_objective("test.aero_post.d_def")
# prob.model.add_objective("test.struct_post.stresscon")

# prob.model.add_objective("test.mass")
# prob.model.add_constraint("test.stresscon", upper = 0.)
# # # set constraints
# # prob.model.add_objective("test.aero_post.d_def")
# # prob.model.add_objective("test.struct_post.mass")
# prob.setup(mode='rev')
# prob.setup(mode='rev')

# set fixed parameters
# prob.set_val("shock_angle", s)
# prob.set_val("M0", M0)

# prob.model.set_val("shock_angle", s)
# prob.model.add_design_var("M1")
# prob.model.add_design_var("beta")
# prob.model.add_design_var("P1")
# prob.model.add_design_var("T1")
# prob.model.add_design_var("r1")
# prob.model.add_design_var("P0")
# prob.model.add_design_var("M0")
# prob.model.add_design_var("T0")
# prob.model.add_design_var("vx0")
# prob.model.add_design_var("r0")
# prob.model.add_design_var("P0")


title = f'{name}_ndv{ndv}_smax{smax}_E{E}_eta{eta_use}.sql'
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
recorder = om.SqliteRecorder(title)
# # prob.model.add_recorder(recorder)
probt.driver.add_recorder(recorder)
probt.driver.recording_options['record_inputs'] = True
probt.driver.recording_options['record_outputs'] = True
probt.driver.recording_options['record_residuals'] = True
probt.driver.recording_options['record_derivatives'] = True
""" 
raw optimization section
"""

probt.run_driver()

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