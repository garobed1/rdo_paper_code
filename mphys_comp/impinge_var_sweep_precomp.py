import sys, os
from utils.sutils import divide_cases
from functions.shock_problem import ImpingingShock
from functions.example_problems import FakeShock
from smt.sampling_methods import LHS, Random, FullFactorial
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

# args = sys.argv[1:]


# Ncase = 500
ndv = 5
smax = 5e5 # max stress constraint
E = 69000000000
title = f'shock_precomp_data_ndv{ndv}_smax{smax}_1'

problem_settings = default_impinge_setup
problem_settings.ndv_true = ndv
# inputs_s = "dv_struct_TRUE"
# inputs_f = {"M0": 1.5,
#             "shock_angle": 25.}
inputs = ["dv_struct_TRUE", "shock_angle", "M0"]
th = np.array([0.0009, 0.0010, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005, 0.006, 0.007])
N_fullfac = 8**2 # 64 evals per direction

N_total = len(th)*N_fullfac

xlimits = np.zeros([ndv+2, 2])
xlimits[:ndv,0] = 0.0009
xlimits[:ndv,1] = 0.007
xlimits[ndv,0] = 23.
xlimits[ndv,1] = 27.
xlimits[ndv+1,0] = 2.5
xlimits[ndv+1,1] = 2.9
xlimits_u = xlimits[-2:,:]

# home = '/gpfs/u/home/ODLC/ODLCbdnn/'
home = '/home/garobed/'

# home = '/home/garobed/'
# if inputs_s == "dv_struct_TRUE":
#     dim = problem_settings.ndv_true
#     # inputs = 
#     xlimits = np.zeros([dim,2])
#     xlimits[:,0] = 0.0001
#     xlimits[:,1] = 0.008
# if inputs_s == "M0":
#     dim = 1
#     # inputs = 
#     xlimits = np.zeros([dim,2])
#     xlimits[:,0] = 1.2
#     xlimits[:,1] = 3.2
# if inputs_s == "shock_angle":
#     dim = 1
#     # inputs = 
#     xlimits = np.zeros([dim,2])
#     xlimits[:,0] = 21.
#     xlimits[:,1] = 28.

# problem_settings = default_impinge_setup
# mesh = f'{home}/garo-rpi-graduate-work/meshes/imp_mphys_73_73_25.cgns'
# problem_settings.aeroOptions['gridFile'] = f'{home}/garo-rpi-graduate-work/meshes/imp_mphys_145_145_25.cgns'
# problem_settings.aeroOptions['gridFile'] = f'{home}/garo-rpi-graduate-work/meshes/imp_long_145_145_25.cgns'
mesh = f'{home}/garo-rpi-graduate-work/meshes/imp_long_217_217_25.cgns'


sampling_u = FullFactorial(xlimits=xlimits_u)


func = ImpingingShock(input_bounds=xlimits, ndv=ndv, E=E, smax=smax, inputs=inputs, mesh=mesh)


# x = sampling(Ncase)
# if not os.path.exists('./{title}/x.npy'):
#     if rank == 0:
#         if not os.path.isdir(title):
#             os.mkdir(title)
x = np.zeros([N_total , ndv+2])
if not os.path.exists(f'./{title}/x.npy'):
    if rank == 0:
        if not os.path.isdir(f'./{title}'):
            os.makedirs(f'./{title}')

        x_u = sampling_u(N_fullfac)
        for i in range(len(th)):
            x[N_fullfac*i:N_fullfac*(i+1), -2:] = x_u
            for j in range(ndv):
                x[N_fullfac*i:N_fullfac*(i+1), j] = th[i]

        with open(f'./{title}/x.npy', 'wb') as f:
            pickle.dump(x, f)

comm.barrier()
with open(f'./{title}/x.npy', 'rb') as f:
    x = pickle.load(f)
# x = comm.bcast(xfull, root=0)
cases = divide_cases(N_total, size)


for i in cases[rank]:

    print(x[i,:])
    f = func(np.atleast_2d(x[i,:]))
    totals = func.prob.compute_totals(of=['test.mass', "test.stresscon"], wrt=['dv_struct_TRUE','test.dv_struct', 'shock_angle','M0'])
    y = np.zeros(3)#, "test.d_def"
    g_uq = np.zeros([2, 2])
    g_dv = np.zeros([2, ndv])
    g_th = np.zeros([2, func.problem_settings.structOptions["th"].shape[0]])
    sig = None
    if not func.prob.driver.fail:
        y[0] = f #mass
        y[1] = copy.deepcopy(func.prob.get_val("test.struct_post.mass")) #mass
        # y[2] = copy.deepcopy(func.prob.get_val("test.aero_post.d_def"))
        y[2] = copy.deepcopy(max(abs(func.prob.get_val("test.struct_post.stress")))) #max stress
        g_uq[0,:] = np.array([copy.deepcopy(totals["test.stresscon",'shock_angle'])[0][0], copy.deepcopy(totals["test.stresscon",'M0'])[0][0]])
        g_uq[1,:] = np.array([copy.deepcopy(totals["test.mass",'shock_angle'])[0][0], copy.deepcopy(totals["test.mass",'M0'])[0][0]])
        # g_uq[2] = np.array([copy.deepcopy(totals["test.struct_post.d_def"]['shock_angle']), copy.deepcopy(totals["test.struct_post.d_def"]['M0'])])
        g_dv[0,:] = copy.deepcopy(totals["test.stresscon",'dv_struct_TRUE'])
        g_dv[1,:] = copy.deepcopy(totals["test.mass",'dv_struct_TRUE'])
        # g_dv[2] = copy.deepcopy(totals["test.struct_post.d_def"]['dv_struct_TRUE'])
        g_th[0,:] = copy.deepcopy(totals["test.stresscon",'test.dv_struct'])
        g_th[1,:] = copy.deepcopy(totals["test.mass",'test.dv_struct'])
        # g_th[2] = copy.deepcopy(totals["test.struct_post.d_def"]['dv_struct'])
        sig =  copy.deepcopy(func.prob.get_val("test.struct_post.stress"))
    else:
        y[0] = np.nan
        y[1] = np.nan
        y[2] = np.nan
        # y[3] = np.nan
        g_uq = np.nan
        g_dv = np.nan
        g_th = np.nan
        sig = np.nan

    with open(f'./{title}/x_{i}.npy', 'wb') as f:
        pickle.dump(x[i], f)
    with open(f'./{title}/y_{i}.npy', 'wb') as f:
        pickle.dump(y, f)
    with open(f'./{title}/g_uq_{i}.npy', 'wb') as f:
        pickle.dump(g_uq, f)
    with open(f'./{title}/g_dv_{i}.npy', 'wb') as f:
        pickle.dump(g_dv, f)
    with open(f'./{title}/g_th_{i}.npy', 'wb') as f:
        pickle.dump(g_th, f)
    with open(f'./{title}/s_{i}.npy', 'wb') as f:
        pickle.dump(sig, f)

    breakpoint()
    