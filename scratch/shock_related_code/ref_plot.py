import sys, os
import copy
import pickle
import math
from matplotlib.transforms import Bbox
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt

from functions.problem_picker import GetProblem
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from surrogate.pougrad import POUSurrogate, POUHessian
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

# Give directory with desired results as argument
title = '10000_shock_results'

plt.rcParams['font.size'] = '18'
plt.rc('legend',fontsize=14)

# Adaptive Data
with open(f'{title}/xref.pickle', 'rb') as f:
    xref = pickle.load(f)
with open(f'{title}/fref.pickle', 'rb') as f:
    fref = pickle.load(f)
xref = np.array(xref, dtype=np.float64)
fref = np.array(fref, dtype=np.float64)

# Plot Error contour
#contour
ndir = 150
# x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
# y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
# X, Y = np.meshgrid(x, y)
# Za = np.zeros([ndir, ndir])
# Zk = np.zeros([ndir, ndir])
# F  = np.zeros([ndir, ndir])
# FK  = np.zeros([ndir, ndir])
# TF = np.zeros([ndir, ndir])
# for i in range(ndir):
#     for j in range(ndir):
#         xi = np.zeros([1,2])
#         xi[0,0] = x[i]
#         xi[0,1] = y[j]
#         F[j,i]  = pmod.predict_values(xi)
#         FK[j,i] = mk.predict_values(xi)
#         TF[j,i] = trueFunc(xi)
#         Za[j,i] = abs(F[j,i] - TF[j,i])
#         Zk[j,i] = abs(FK[j,i] - TF[j,i])
# # Plot original function
cs = plt.tricontourf(xref[:,0], xref[:,1], fref, levels = 40)
cbar = plt.colorbar(cs, aspect=20, )
cbar.formatter.set_powerlimits((0, 0))
cbar.set_label(r"$c_d$")
plt.xlabel(r"$\theta_s$")
plt.ylabel(r"$\kappa$")
plt.xticks(ticks=[23,24,25,26,27])
#plt.legend(loc=1)
plt.savefig(f"{title}_true.pdf", bbox_inches="tight")
plt.clf()