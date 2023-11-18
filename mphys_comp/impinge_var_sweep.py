import sys, os
from utils.sutils import divide_cases
from functions.shock_problem import ImpingingShock
from functions.example_problems import FakeShock
from smt.sampling_methods import LHS, Random
from mpi4py import MPI
import pickle
import openmdao.api as om
import numpy as np

import mphys_comp.impinge_setup as default_impinge_setup
from mphys_comp.impinge_analysis import Top

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

args = sys.argv[1:]

Ncase = 12

inputs_f = {"mach": 1.5,
            "shock_angle": 25.}
# inputs_s = ["dv_struct"]
# inputs = 
problem_settings = default_impinge_setup
dim = problem_settings.ndv_true
xlimits = np.zeros([dim,2])
xlimits[:,0] = 0.0001
xlimits[:,1] = 0.005
sampling = Random(xlimits=xlimits)

# x = sampling(Ncase)
title = f'shock_sweep_mach{inputs_f["mach"]}_sangle{inputs_f["shock_angle"]}'
if not os.path.exists('./{title}/x.npy'):
    if rank == 0:
        if not os.path.isdir(title):
            os.mkdir(title)
        x = sampling(Ncase)
        with open(f'./{title}/x.npy', 'wb') as f:
            pickle.dump(x, f)
comm.barrier()
with open(f'./{title}/x.npy', 'rb') as f:
    x = pickle.load(f)
# x = comm.bcast(xfull, root=0)
cases = divide_cases(Ncase, size)

problem_settings = default_impinge_setup
problem_settings.aeroOptions['L2Convergence'] = 1e-15
problem_settings.aeroOptions['printIterations'] = False
problem_settings.aeroOptions['printTiming'] = False
problem_settings.optOptions['shock_angle'] = inputs_f["shock_angle"]
problem_settings.mach = inputs_f["mach"]
# problem_settings.aeroOptions['gridFile'] = f'../meshes/imp_mphys_73_73_25.cgns'
problem_settings.aeroOptions['gridFile'] = f'../meshes/imp_mphys_145_145_25.cgns'
nelem = 60
# nelem = 30
problem_settings.nelem = nelem
problem_settings.structOptions['Nelem'] = nelem
problem_settings.structOptions['force'] = np.ones(nelem+1)*1.0
problem_settings.structOptions["th"] = np.ones(nelem+1)*0.0005
prob = om.Problem(comm=MPI.COMM_SELF)
prob.model = Top(problem_settings=problem_settings, subsonic=False,
                                                     use_shock_comp=True, 
                                                     use_inflow_comp=True, 
                                                     full_free=False)
prob.setup(mode='rev')
# func = ImpingingShock(ndim=dim, input_bounds=xlimits, inputs=inputs, problem_settings=problem_settings)
# func = FakeShock(ndim=dim)
# np.ones(dim)*0.0005
for i in cases[rank]:
    print(x[i])
    prob.set_val("dv_struct_TRUE", x[i])
    prob.run_model()
    y = np.zeros(5)
    if not self.prob.driver.fail:
        y[0] = copy.deepcopy(prob.get_val("test.struct_post.func_struct"))[0] #mass
        y[1] = copy.deepcopy(prob.get_val("test.struct_post.func_struct"))[1] #max stress
        y[2] = copy.deepcopy(prob.get_val("test.aero_post.d_def"))
        y[3] = copy.deepcopy(prob.get_val("test.struct_post.dv_def"))
        y[4] = copy.deepcopy(prob.get_val("test.struct_post.dp_def"))
    else:
        y[0] = np.nan
        y[1] = np.nan
        y[2] = np.nan
        y[3] = np.nan
        y[4] = np.nan

    with open(f'./{title}/x_{i}.npy', 'wb') as f:
        pickle.dump(x[i], f)
    with open(f'./{title}/y_{i}.npy', 'wb') as f:
        pickle.dump(y, f)
