import numpy as np
from surrogate.pougrad import POUHessian
import matplotlib.pyplot as plt
import argparse
from mpi4py import MPI
from utils.sutils import convert_to_smt_grads
import pickle
import os, sys
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Grab POU models from optimizations and plot slices according to input
"""

machline = 2.7
shockline = 25.
out = 0 # 0 for KS
optiter = [0, 4, 6, 8, 9]#, -1] # -1 latest opt iter
min_contribution = 1e-14
rscale = 5.5
rscale = 100.5
rho = 10 
neval_add = 3 #applies to both criteria and POU model
# neval_add = 30 #applies to both criteria and POU model
neval_fac = 1
delta = 1e-10
add_data = True

plt.rcParams['font.size'] = '16'
plt.rcParams['savefig.dpi'] = 600

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--optdir', action='store', help = 'directory containing opt run outputs and settings')
parser.add_argument('-d', '--datadir', action='store', help = 'directory containing constant thickness data')

args = parser.parse_args()
datadir = args.datadir
optdir = args.optdir




root = os.getcwd()
# put data into arrays
i = 0
x_list = []
y_list = []
g_uq_list0 = []
g_uq_list1 = []
g_dv_list0 = []
g_dv_list1 = []
g_th_list0 = []
g_th_list1 = []
while 1:
    try:
        with open(f'/{root}/{datadir}/x_{i}.npy', 'rb') as f:
            x_cur = pickle.load(f)
        with open(f'/{root}/{datadir}/y_{i}.npy', 'rb') as f:
            y_cur = pickle.load(f)
        with open(f'/{root}/{datadir}/g_uq_{i}.npy', 'rb') as f:
            g_uq_cur = pickle.load(f)
        with open(f'/{root}/{datadir}/g_dv_{i}.npy', 'rb') as f:
            g_dv_cur = pickle.load(f)
        with open(f'/{root}/{datadir}/g_th_{i}.npy', 'rb') as f:
            g_th_cur = pickle.load(f)
        x_list.append(x_cur)
        y_list.append(y_cur)
        g_uq_list0.append(g_uq_cur[0,:])
        g_uq_list1.append(g_uq_cur[1,:])
        g_dv_list0.append(g_dv_cur[0,:])
        g_dv_list1.append(g_dv_cur[1,:])
        g_th_list0.append(g_th_cur[0,:])
        g_th_list1.append(g_th_cur[1,:])
        i += 1
    except:
        break

x = np.array(x_list)
y = np.array(y_list)

ndvd = x.shape[1] - 2

# read data
c = 0
old = x[0,0]
new = x[0,0]
while abs(old - new) < 1e-9:
    c += 1
    new = x[c,0]

g_uq0 = np.array(g_uq_list0)
g_uq1 = np.array(g_uq_list1)
g_dv0 = np.array(g_dv_list0)
g_dv1 = np.array(g_dv_list1)
g_th0 = np.array(g_th_list0)
g_th1 = np.array(g_th_list1)

# attach uq to dv
g_full0 = np.append(g_dv0, g_uq0, axis = 1)
g_full1 = np.append(g_dv1, g_uq1, axis = 1) # not really needed for this

# find 
indslicepre = np.argmin(abs(x[:,ndvd] - shockline) + abs(x[:,ndvd+1] - machline))
indslice = []
d = 0
while d*c < x.shape[0]:
    indslice.append(d*c+indslicepre)
    d += 1


x_dat = x[indslice,0]
y_dat = y[indslice,out]

# get actual values as well
shocklinetrue = x[indslice[0], ndvd]
machlinetrue  = x[indslice[0], ndvd+1]

# or do mean



##### SURROGATE MODEL PARAMETERS #####
#TODO: WRITE SURROGATE PICKER
optsplit = optdir.split(sep='/')
title = optsplit[-1]
path = optsplit[:-1]
if len(path) == 0:
    path = ''
else:
    path = path.join('/')

areds = []
preds = []
feasability = []
optimality = []
grad_lhs = []
grad_rhs = []
loc = []
models = []
prob_model_points = []
prob_truth_points = []
radii = []
realizations = []
reflog = []
duals = []

k = 0
while 1:
    if not os.path.isfile(f'/{root}/{path}/{title}/loc_{k}.pickle'):
        break

    with open(f'/{root}/{path}/{title}/grad_lhs_{k}.pickle', 'rb') as f:
        grad_lhs.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/grad_rhs_{k}.pickle', 'rb') as f:
        grad_rhs.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/radii_{k}.pickle', 'rb') as f:
        radii.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/realizations_{k}.pickle', 'rb') as f:
        realizations.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/optimality_{k}.pickle', 'rb') as f:
        optimality.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/feasability_{k}.pickle', 'rb') as f:
        feasability.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/areds_{k}.pickle', 'rb') as f:
        areds.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/preds_{k}.pickle', 'rb') as f:
        preds.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/loc_{k}.pickle', 'rb') as f:
        loc.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/duals_{k}.pickle', 'rb') as f:
        duals.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/reflog_{k}.pickle', 'rb') as f:
        reflog.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/prob_truth_points_{k}.pickle', 'rb') as f:
        prob_truth_points.append(pickle.load(f))
    with open(f'/{root}/{path}/{title}/prob_model_points_{k}.pickle', 'rb') as f:
        prob_model_points.append(pickle.load(f))
    try:
        with open(f'/{root}/{path}/{title}/models_{k}.pickle', 'rb') as f:
            models.append(pickle.load(f))
    except:
        models.append(None)
    # import pdb; pdb.set_trace()
    k += 1

ndv = prob_model_points[0]['x'].shape[1] - 2
xlimits = np.zeros([ndv+2, 2])
xlimits[:ndv,0] = 0.0009
xlimits[:ndv,1] = 0.007
xlimits[ndv,0] = 23.
xlimits[ndv,1] = 27.
xlimits[ndv+1,0] = 2.5
xlimits[ndv+1,1] = 2.9
xlimits_u = xlimits[-2:,:]
msur = None
t_dim = ndv + 2
msur = POUHessian(bounds=xlimits, delta=delta, rscale = rscale, neval = neval_fac*t_dim+neval_add, min_contribution = min_contribution)

ndir = 100
msur.options.update({"print_prediction":False})
msur.options.update({"print_global":False})
x_sur = np.linspace(xlimits[0,0], xlimits[0,1], ndir)
x_sur_use = np.zeros([ndir, ndv+2])
for i in range(ndv):
    x_sur_use[:,i] = x_sur

### change only few DVS
x_sur_use[:,:] = x_sur[1]
# x_sur_use[:,1] = np.flip(x_sur)
# x_sur_use[:,0] = x_sur
x_sur_use[:,4] = x_sur
    
# use true values
x_sur_use[:,ndv] = shocklinetrue
x_sur_use[:,ndv+1] = machlinetrue

# use queried values
# x_sur_use[:,ndv] = shockline
# x_sur_use[:,ndv+1] = machline


# plot data
if rank == 0:
    plt.plot(x_dat, y_dat, '-x', label='true')

# breakpoint()
for k in range(len(optiter)):
    xt = prob_model_points[optiter[k]]['x']
    ft = prob_model_points[optiter[k]]['f']
    gt = prob_model_points[optiter[k]]['g']

    # try adding data to these
    if add_data:
        xt = np.append(xt, x, axis = 0)
        ft = np.append(ft, np.array([y[:,out]]).T, axis = 0)
        gt = np.append(gt, g_full0, axis = 0)

    msur.set_training_values(xt, ft)
    # convert_to_smt_grads(models[-1], prob_model_points[-1]['x'], prob_model_points[-1]['g'])
    convert_to_smt_grads(msur, xt, gt)
    # models[-1].train()
    msur.train()
    
    y_sur = msur.predict_values(x_sur_use)



    # y_sur = msur.predict_values(x_sur_use)


    if rank == 0:
        plt.plot(x_sur, y_sur, '-', label=f'optmod{optiter[k]}')

# do just the data surrogate
msur.set_training_values(x,  np.array([y[:,out]]).T)
# convert_to_smt_grads(models[-1], prob_model_points[-1]['x'], prob_model_points[-1]['g'])
convert_to_smt_grads(msur, x, g_full0)
# models[-1].train()
msur.train()
y_sur = msur.predict_values(x_sur_use)

if rank == 0:
    plt.plot(x_sur, y_sur, '--k', label=f'datamod')

    plt.xlabel('uniform th')
    plt.ylabel('KS')
    plt.legend()
    plt.savefig(f'slice_s{shocklinetrue}_m{machlinetrue}.png', bbox_inches="tight")

    breakpoint()