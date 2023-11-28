import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import argparse
import os
import pickle
# plt.rcParams['font.size'] = '15'
# plt.rcParams['savefig.dpi'] = 600

"""
scratch space for looking at shock problem data to determine appropriate problem
"""


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datadir', action='store', help = 'directory containing opt run outputs and settings')



args = parser.parse_args()
optdir = args.datadir

inputs = ["shock_angle", "M0", "dv_struct_TRUE"]
input_sweep = "shock_angle"


root = os.getcwd()
# put data into arrays
# i = 0
# x_list = []
# y_list = []
# while 1:
#     try:
#         with open(f'/{root}/{optdir}/x_{i}.npy', 'rb') as f:
#             x_cur = pickle.load(f)
#         with open(f'/{root}/{optdir}/y_{i}.npy', 'rb') as f:
#             y_cur = pickle.load(f)
#         x_list.append(x_cur)
#         y_list.append(y_cur)
#         i += 1
#     except:
#         break

# x = np.array(x_list).T
# y = np.array(y_list).T

with open(f'/{root}/{optdir}/x_full.npy', 'rb') as f:
    x = pickle.load(f)
with open(f'/{root}/{optdir}/y_full.npy', 'rb') as f:
    y = pickle.load(f)

nind = x.shape[0]
ynames = ['mass', 'maxstress', 'd_def', 'dv_def', 'dp_def']
for i in range(y.shape[0]):
    plt.scatter(x, y[i,:])
    plt.xlabel(input_sweep)
    plt.ylabel(ynames[i])
    plt.savefig(f'/{root}/{optdir}/sweep_{ynames[i]}.png', bbox_inches='tight')
    plt.clf()


# with open(f'/{root}/{optdir}/x_full.npy', 'wb') as f:
#     pickle.dump(x, f)
# with open(f'/{root}/{optdir}/y_full.npy', 'wb') as f:
#     pickle.dump(y, f)

# linear fit to start?
    
# def fnq(x_d, C):
#     ans = 0.
#     for i in range(C.size - 1):
#         ans += C[i]*x_d[i]

#     ans += C[C.size - 1]

#     return ans

# def wrap(x_d, *args):
#     coeff = np.array(args)[0]
#     return fnq(x_d, coeff)


# yuse = 2 # dependent var to fit
# popt, pcov = curve_fit(lambda X, *params: wrap(X, params), x, y[yuse,:], p0=np.ones(nind))

# print(pcov)
# print(popt)
# sol = minimize()

# import pdb; pdb.set_trace()