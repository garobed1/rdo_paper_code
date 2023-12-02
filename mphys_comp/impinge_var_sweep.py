import sys, os
from utils.sutils import divide_cases
from functions.shock_problem import ImpingingShock
from functions.example_problems import FakeShock
from smt.sampling_methods import LHS, Random
from mpi4py import MPI
import pickle
import openmdao.api as om
import numpy as np
import copy

import mphys_comp.impinge_setup as default_impinge_setup
from mphys_comp.impinge_analysis import Top

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

args = sys.argv[1:]

Ncase = 500
ndv = 4
smax = 1e6 # max stress constraint
E = 690e6
problem_settings = default_impinge_setup
problem_settings.ndv_true = ndv
inputs_s = "dv_struct_TRUE"
inputs_f = {"M0": 1.5,
            "shock_angle": 25.}
# inputs_s = "M0"
# inputs_f = {"dv_struct_TRUE": 0.0005,
#             "shock_angle": 25.}
# inputs_s = "shock_angle"
# inputs_f = {"M0": 1.5,
#             "dv_struct_TRUE": 0.0005,}

home = '/home/garobed/'
# home = '/home/garobed/'
if inputs_s == "dv_struct_TRUE":
    dim = problem_settings.ndv_true
    # inputs = 
    xlimits = np.zeros([dim,2])
    xlimits[:,0] = 0.0001
    xlimits[:,1] = 0.008
if inputs_s == "M0":
    dim = 1
    # inputs = 
    xlimits = np.zeros([dim,2])
    xlimits[:,0] = 1.2
    xlimits[:,1] = 3.2
if inputs_s == "shock_angle":
    dim = 1
    # inputs = 
    xlimits = np.zeros([dim,2])
    xlimits[:,0] = 21.
    xlimits[:,1] = 28.

problem_settings = default_impinge_setup
problem_settings.aeroOptions['NKSwitchTol'] = 1e-4 #1e-6
problem_settings.aeroOptions['L2Convergence'] = 1e-12
problem_settings.aeroOptions['printIterations'] = False
problem_settings.aeroOptions['printTiming'] = False
# problem_settings.aeroOptions['gridFile'] = f'{home}/garo-rpi-graduate-work/meshes/imp_mphys_73_73_25.cgns'
# nelem = 30

problem_settings.aeroOptions['gridFile'] = f'{home}/garo-rpi-graduate-work/meshes/imp_mphys_145_145_25.cgns'
nelem = 60
L = 0.254

# problem_settings.aeroOptions['gridFile'] = f'{home}/garo-rpi-graduate-work/meshes/imp_long_145_145_25.cgns'
# nelem = 78
# L = 0.75

# problem_settings.aeroOptions['gridFile'] = f'{home}/garo-rpi-graduate-work/meshes/imp_long_217_217_25.cgns'
# nelem = 117
# L = 0.75

problem_settings.nelem = nelem
problem_settings.structOptions['Nelem'] = nelem
problem_settings.structOptions['force'] = np.ones(nelem+1)*1.0
problem_settings.structOptions["th"] = np.ones(nelem+1)*0.0005
problem_settings.structOptions["ndv_true"] = ndv
problem_settings.structOptions["th_true"] = np.ones(ndv)*0.0005
problem_settings.structOptions['E'] = E
problem_settings.structOptions['L'] = L
problem_settings.structOptions['smax'] = smax

if "dv_struct_TRUE" in inputs_f:
    problem_settings.structOptions["th"] = np.ones(nelem+1)*inputs_f["dv_struct_TRUE"]
if "shock_angle" in inputs_f:
    problem_settings.optOptions['shock_angle'] = inputs_f["shock_angle"]
if "M0" in inputs_f:
    problem_settings.M0 = inputs_f["M0"]

# home = '/gpfs/u/home/ODLC/ODLCbdnn/'
home = '/home/garobed/'

sampling = Random(xlimits=xlimits)

# x = sampling(Ncase)
title = f'shock_sweep_4_{list(inputs_f.keys())[0]}{inputs_f[list(inputs_f.keys())[0]]}_{list(inputs_f.keys())[1]}{inputs_f[list(inputs_f.keys())[1]]}'
# if not os.path.exists('./{title}/x.npy'):
#     if rank == 0:
#         if not os.path.isdir(title):
#             os.mkdir(title)
if not os.path.exists(f'./{title}/x.npy'):
    if rank == 0:
        if not os.path.isdir(f'./{title}'):
            os.makedirs(f'./{title}')
        x = sampling(Ncase)
        with open(f'./{title}/x.npy', 'wb') as f:
            pickle.dump(x, f)
comm.barrier()
with open(f'./{title}/x.npy', 'rb') as f:
    x = pickle.load(f)
# x = comm.bcast(xfull, root=0)
cases = divide_cases(Ncase, size)



                 
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
    prob.set_val(inputs_s, x[i])
    prob.run_model()
    # totals = prob.compute_totals(of=['test.struct_post.mass', "test.struct_post.stress", "test.aero_post.d_def"], wrt=['dv_struct_TRUE','shock_angle','M0'])
    y = np.zeros(5)
    sig = None
    if not prob.driver.fail:
        y[0] = copy.deepcopy(prob.get_val("test.struct_post.mass")) #mass
        y[1] = copy.deepcopy(max(abs(prob.get_val("test.struct_post.stress")))) #max stress
        sig =  copy.deepcopy(prob.get_val("test.struct_post.stress"))
        y[2] = copy.deepcopy(prob.get_val("test.aero_post.d_def"))
        y[3] = copy.deepcopy(prob.get_val("test.aero_post.dv_def"))
        y[4] = copy.deepcopy(prob.get_val("test.aero_post.dp_def"))
    else:
        y[0] = np.nan
        y[1] = np.nan
        y[2] = np.nan
        y[3] = np.nan
        y[4] = np.nan
        sig = np.nan

    with open(f'./{title}/x_{i}.npy', 'wb') as f:
        pickle.dump(x[i], f)
    with open(f'./{title}/y_{i}.npy', 'wb') as f:
        pickle.dump(y, f)
    # with open(f'./{title}/g_{i}.pickle', 'wb') as f:
    #     pickle.dump(totals, f)
    with open(f'./{title}/s_{i}.npy', 'wb') as f:
        pickle.dump(sig, f)
