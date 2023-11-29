import imp
import numpy as np
import argparse
from mpi4py import MPI
import sys

import openmdao.api as om

from mphys import Multipoint
from mphys_comp.shock_angle_comp import ShockAngleComp
from mphys_comp.inflow_comp import InflowComp
from mphys_comp.impinge_analysis import Top
from mphys.scenario_aerostructural import ScenarioAeroStructural

# these imports will be from the respective codes' repos rather than mphys
from mphys.mphys_adflow import ADflowBuilder
from beam.mphys_eb import EBBuilder
from beam.mphys_onetoone import OTOBuilder
from beam.om_beamdvs import beamDVComp

#from mphys.mphys_meld import MeldBuilder
#from mphys.mphys_rlt import RltBuilder

from baseclasses import AeroProblem

# from tacs import elements, constitutive, functions

# contains all options, aero, opt, struct, uq, warp
import mphys_comp.impinge_setup as default_impinge_setup

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

# use as scratch space for playing around


################################################################################
# OpenMDAO setup
################################################################################
use_shock = True
use_inflow = True
full_far = False
subsonic = False

### PARAMS ###
nsweep = 9
s_list = 4*[25., 25., 25., 22., 22., 22., 27., 27., 27.]
M0_list = 4*[1.4, 2.2, 3.0, 1.4, 2.2, 3.0, 1.4, 2.2, 3.0]
smax_list = nsweep*[1e5] + nsweep*[1e6] + nsweep*[1e5] +  nsweep*[1e6] 
E_list = nsweep*[69000000000] +nsweep*[69000000000] + nsweep*[6900000000] + nsweep*[6900000000]

ndv = 4 # number of thickness variables
s = s_list[rank] # shock angle
M0 = M0_list[rank] # upstream mach number
smax = smax_list[rank] # max stress constraint
E = E_list[rank]

# aero solver
problem_settings = default_impinge_setup
# problem_settings.aeroOptions['equationType'] = 'laminar NS'
# problem_settings.aeroOptions['equationType'] = 'Euler'
problem_settings.aeroOptions['NKSwitchTol'] = 1e-6 #1e-6
# problem_settings.aeroOptions['NKSwitchTol'] = 1e-3 #1e-6
problem_settings.aeroOptions['nCycles'] = 5000000
problem_settings.aeroOptions['L2Convergence'] = 1e-12
problem_settings.aeroOptions['printIterations'] = False
problem_settings.aeroOptions['printTiming'] = True

if full_far:
    aeroGridFile = f'../meshes/imp_TEST_73_73_25.cgns'
elif subsonic:
    aeroGridFile = f'../meshes/imp_subs_73_73_25.cgns'
else:
    # aeroGridFile = f'../meshes/imp_mphys_73_73_25.cgns'
    # nelem = 30
    # L = .254
    aeroGridFile = f'../meshes/imp_long_217_217_25.cgns'
    nelem = 117
    L = .75
problem_settings.aeroOptions['gridFile'] = aeroGridFile

# aero settings
# problem_settings.mach = 2.7
# problem_settings.beta = 0.8
# problem_settings.mach = 0.8

# struct solver
problem_settings.nelem = nelem
problem_settings.structOptions['E'] = E
problem_settings.structOptions['L'] = L
problem_settings.structOptions['smax'] = smax
problem_settings.structOptions['Nelem'] = nelem
problem_settings.structOptions['force'] = np.ones(nelem+1)*1.0
problem_settings.structOptions["th"] = np.ones(nelem+1)*0.0005
problem_settings.structOptions["ndv_true"] = ndv
problem_settings.structOptions["th_true"] = np.ones(ndv)*0.0005

prob = om.Problem(comm=MPI.COMM_SELF)
prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP') 
prob.model = Top(problem_settings=problem_settings, subsonic=subsonic,
                                                     use_shock_comp=use_shock, 
                                                     use_inflow_comp=use_inflow, 
                                                     full_free=full_far)









# set design variables
prob.model.add_design_var("dv_struct_TRUE", lower = 0.0001, upper = 0.007)
# prob.model.add_design_var("shock_angle")

# set objective
# prob.model.add_objective("test.aero_post.d_def")
# prob.model.add_objective("test.struct_post.stresscon")

prob.model.add_objective("test.mass")
prob.model.add_constraint("test.stresscon", upper = smax)
# # set constraints
# prob.model.add_objective("test.aero_post.d_def")
# prob.model.add_objective("test.struct_post.mass")
prob.setup(mode='rev')
# prob.setup(mode='rev')

# set fixed parameters
prob.model.set_val("shock_angle", s)
prob.model.set_val("M0", M0)

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


# recorder 
recorder = om.SqliteRecorder(f'opt_test_ndv{ndv}_s{s}_M0{M0}_smax{smax}_E{E}.sql')
prob.driver.add_recorder(recorder)

# s = 25. # shock angle
# M0 = 2.2 # upstream mach number
# smax = 1e5 # max stress constraint
# E = 69000000000
#prob.set_val("mach", 2.)
#prob.set_val("dv_struct", impinge_setup.structOptions["th"])
#prob.set_val("beta", 7.)
#x = np.linspace(2.5, 3.5, 10)
#y = np.zeros(10)
#for i in range(10):
#prob.set_val("M0", x[i])
# prob.set_val("shock_angle", 25.)

#prob.model.approx_totals()
# prob.run_model()
prob.run_driver()

#NOTE: NEED TO SAVE DATA AND CONVERGENCE

import pdb; pdb.set_trace()
# import copy
# y0 = copy.deepcopy(prob.get_val("test.aero_post.cd_def"))
# #totals1 = prob.compute_totals(wrt='rsak')
# #prob.model.approx_totals()
# totals2 = prob.compute_totals(of='test.aero_post.cd_def', wrt=['shock.mach1','shock_angle','rsak'])
# h = 1e-8
# prob.set_val("rsak", 0.41 + h)
# prob.run_model()
# y1k = copy.deepcopy(prob.get_val("test.aero_post.cd_def"))
# prob.set_val("rsak", 0.41)
# prob.set_val("shock.mach1", default_impinge_setup.mach+h)
# prob.run_model()
# y1s = copy.deepcopy(prob.get_val("test.aero_post.cd_def"))

