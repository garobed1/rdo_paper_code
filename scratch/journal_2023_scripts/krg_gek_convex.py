import sys, os
import copy
import pickle
import math
from matplotlib.transforms import Bbox
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from utils.sutils import divide_cases
from utils.error import rmse, meane

# from smt.problems import TensorProduct, 
from functions.example_problems_2 import SimpleConvex
from smt.surrogate_models import KPLS, GEKPLS, KRG
from surrogate.gek_1d import GEK1D
#from smt.surrogate_models.rbf import RBF
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS, FullFactorial, Random
from optimization.defaults import DefaultOptOptions

# D = False
# adapt = True
# loaddata = True
plt.rcParams['font.size'] = '14'
# prob = "arctan"
corr = "squar_exp"
dim = 1

# Problem Settings
trueFunc = SimpleConvex(order=6)

xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')
# sampling = FullFactorial(xlimits=xlimits)
# sampling = Random(xlimits=xlimits)

Nerr = 5000*2
xtest = sampling(Nerr)
ftest = trueFunc(xtest)
testdata = [xtest, ftest]
limits = [-6, 2]

nt0 = 20

# xk = sampling(nt0)
xk = np.atleast_2d(np.linspace(0.05, 0.95, nt0)).T
# xk = np.array([[0.1],[0.3],[0.5],[0.7],[0.9]])
fk = trueFunc(xk)
gk = np.zeros([nt0,dim])
for j in range(dim):
    gk[:,j:j+1] = trueFunc(xk, j)

modelbase = GEK1D(xlimits=xlimits)
modelbase.options.update({"corr":corr})
modelbase.options.update({"poly":"linear"})
modelbase.options.update({"theta_bounds":[10**limits[0], 10**limits[1]]})
modelbase.options.update({"delta_x":1e-4})
# modelbase.options.update({"zero_out_y":True})
# modelbase.options.update({"n_start":5})
modelbase.set_training_values(xk, fk)
for j in range(dim):
    modelbase.set_training_derivatives(xk, gk[:,j:j+1], j)

modelbase.train()
modelbase.options.update({"print_global":False})

modelbase2 = KRG()
# modelbase2 = KPLS()
# modelbase2.options.update({"n_comp":dim})
modelbase2.options.update({"corr":corr})
modelbase2.options.update({"poly":"linear"})
modelbase2.options.update({"theta_bounds":[10**limits[0], 10**limits[1]]})
# modelbase.options.update({"n_start":5})
modelbase2.set_training_values(xk, fk)
modelbase2.train()
modelbase2.options.update({"print_global":False})

thopt = modelbase.optimal_theta
thopt2 = modelbase2.optimal_theta

err1 = rmse(modelbase, trueFunc, N=Nerr, xdata = xtest, fdata=ftest)
err2 = rmse(modelbase2, trueFunc, N=Nerr, xdata = xtest, fdata=ftest)

ndir = 75
xlimits = trueFunc.xlimits
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
ZDGEK = np.zeros([ndir])
ZKRG = np.zeros([ndir])
FDGEK = np.zeros([ndir])
FKRG = np.zeros([ndir])
TF = np.zeros([ndir])
for i in range(ndir):
    xi = np.zeros([1,1])
    xi[0] = x[i]
    TF[i] = trueFunc(xi)
    FDGEK[i] = modelbase.predict_values(xi)
    FKRG[i] = modelbase2.predict_values(xi)
    ZDGEK[i] = abs(FDGEK[i] - TF[i])
    ZKRG[i] = abs(FKRG[i] - TF[i])

trx = modelbase.training_points[None][0][0]
trf = modelbase.training_points[None][0][1]

# Plot Non-Adaptive Error
plt.plot(x, FDGEK, linewidth=1.2, label=f'GEK')
plt.plot(x, FKRG, linewidth=1.2, label=f'Kriging')
plt.plot(x, TF, 'k-', linewidth=1.2, label=f'Original')

# plt.yscale("log")
plt.xlabel(r"$x$")
plt.ylabel(r"$x^6$")
plt.grid()
#plt.legend(loc=1)
plt.plot(trx[0:nt0,0], trf[0:nt0,0], "bo", ms=4 , label=f'Sample Locations')#min(np.min(ZKRG),np.min(ZDGEK))*np.ones_like(
plt.legend()


plt.arrow(0.16, 0.16, -0.15, -0.15, width=0.01, length_includes_head=True)
plt.arrow(0.84, 0.84, 0.15, 0.15, width=0.01, length_includes_head=True)
plt.savefig(f"./convex_thing.pdf", bbox_inches="tight")

#left
plt.axis([0.0, 0.01, -0.0001, 0.0001])
plt.savefig(f"./convex_thing_left.pdf", bbox_inches="tight")

#right
plt.axis([0.9998, 1.0, 0.999, 1.0])
plt.ticklabel_format(axis='x', style='sci')
plt.savefig(f"./convex_thing_right.pdf", bbox_inches="tight")

plt.clf()

import pdb; pdb.set_trace()
# plt.savefig(f"./gek_issues.{save_format}", bbox_inches="tight")





