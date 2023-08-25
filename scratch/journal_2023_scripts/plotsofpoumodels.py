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

from smt.problems import TensorProduct, Rosenbrock
from smt.surrogate_models import KPLS, GEKPLS, KRG
from surrogate.pougrad import POUHessian
from functions.problem_picker import GetProblem
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
# dim = 2
prob = "fuhgp3"
dim = 1

# Problem Settings
trueFunc = GetProblem(prob, dim)

xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')
# sampling = FullFactorial(xlimits=xlimits)
# sampling = Random(xlimits=xlimits)

Nerr = 5000*dim
xtest = sampling(Nerr)
ftest = trueFunc(xtest)
testdata = [xtest, ftest]

rscale = 5.5
neval = 1+dim+2
nt0 = dim*20

xk = sampling(nt0)
fk = trueFunc(xk)
gk = np.zeros([nt0,dim])
for j in range(dim):
    gk[:,j:j+1] = trueFunc(xk, j)

modelbase = POUHessian(bounds=xlimits, rscale=rscale)
modelbase.options.update({"neval":neval})
# modelbase.options.update({"zero_out_y":True})
# modelbase.options.update({"n_start":5})
modelbase.set_training_values(xk, fk)
for j in range(dim):
    modelbase.set_training_derivatives(xk, gk[:,j:j+1], j)

modelbase.train()
modelbase.options.update({"print_global":False})



if(dim == 1):
    plt.clf()

    ndir = 75
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
    trx = modelbase.training_points[None][0][0]
    trf = modelbase.training_points[None][0][1]
    Z = np.zeros([ndir])
    Zh = np.zeros([ndir])
    F = np.zeros([ndir])
    Fh = np.zeros([ndir])
    TF = np.zeros([ndir])

    for i in range(ndir):
        xi = np.zeros([1,1])
        xi[0] = x[i]
        TF[i] = trueFunc(xi)
        Fh[i] = modelbase.predict_values(xi)
        Zh[i] = abs(Fh[i] - TF[i])


    # Plot Non-Adaptive Error
    plt.plot(x, TF, "-k", label=f'True')
    plt.plot(x, Fh, "-b", label=f'POU')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(\mathbf{x})$")
    #plt.legend(loc=1)
    plt.plot(trx[:,0], trf[:,0], "bo", label='LHS Points')
    plt.legend(fontsize='13')
    plt.savefig(f"plotsofpoumodels/1d_models_{prob}.pdf", bbox_inches="tight")
    plt.clf()

    # Plot Non-Adaptive Error
    # plt.plot(x, Zgek, "-m", label=f'IGEK')
    plt.plot(x, Zh, "-m")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$|\hat{f}(\mathbf{x}) - f(\mathbf{x})|$")
    plt.plot(trx[:,0], np.zeros_like(trf[:,0]), "bo", label='LHS Points')
    plt.legend(fontsize='13')
    plt.savefig(f"plotsofpoumodels/1d_errcon_{prob}.pdf", bbox_inches="tight")

    plt.clf()


if(dim == 2):


    # Plot Error contour
    #contour
    ndir = 150
    xlimits = trueFunc.xlimits
    x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
    y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)

    X, Y = np.meshgrid(x, y)
    Za = np.zeros([ndir, ndir])
    Zk = np.zeros([ndir, ndir])
    F  = np.zeros([ndir, ndir])
    FK  = np.zeros([ndir, ndir])
    TF = np.zeros([ndir, ndir])

    for i in range(ndir):
        for j in range(ndir):
            xi = np.zeros([1,2])
            xi[0,0] = x[i]
            xi[0,1] = y[j]
            FK[j,i] = modelbase.predict_values(xi)
            TF[j,i] = trueFunc(xi)
            Zk[j,i] = abs(FK[j,i] - TF[j,i])

    # Plot Model
    tk = modelbase.training_points[None][0][0]
    cs = plt.contourf(X, Y, FK, levels = np.linspace(-2.0, 2.0, 30))
    cbar = plt.colorbar(cs, )#levels=np.linspace(-2.0, 2.0, 30)
    cbar.set_label(r"$\hat{f}(\mathbf{x})$")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.plot(tk[:,0], tk[:,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='LHS Points')
    plt.legend(fontsize=13)
    plt.savefig(f"plotsofpoumodels/2d_models_{prob}.pdf", bbox_inches="tight")
    plt.clf()

    # Plot Non-Adaptive Error
    tk = modelbase.training_points[None][0][0]
    cs = plt.contourf(X, Y, Zk, levels = 30)
    cbar = plt.colorbar(cs, )
    cbar.set_label(r"$|\hat{f}(\mathbf{x}) - f(\mathbf{x})|$")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.plot(tk[:,0], tk[:,1], "o", fillstyle='full', markerfacecolor='c', markeredgecolor='c', label='LHS Points')
    plt.legend(fontsize=13)
    plt.savefig(f"plotsofpoumodels/2d_errcon_{prob}.pdf", bbox_inches="tight")
    plt.clf()

# plt.savefig(f"./gek_issues.{save_format}", bbox_inches="tight")





