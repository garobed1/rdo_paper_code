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
from functions.example_problems import ToyLinearScale, QuadHadamard
from functions.example_problems_2 import ScalingExpSine
#from smt.surrogate_models.rbf import RBF
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS, FullFactorial, Random
from optimization.defaults import DefaultOptOptions


func = [ToyLinearScale(ndim=10, use_design=False),
    QuadHadamard(ndim=8),
    ToyLinearScale(ndim=10, use_design=True),
    ScalingExpSine(ndim=10)]

dim = [10, 8, 10, 10]

nt0 = 20

model = []
sampling = []
Nerr = []
xtest = []
ftest = []
xk = []
fk = []
gk = []
NRMSE = []
for i in range(4):
    # import pdb; pdb.set_trace()
    model.append(POUHessian(bounds=func[i].xlimits, rscale=5.5, neval = 1+dim[i]+2))
    sampling.append(LHS(xlimits=func[i].xlimits, criterion="maximin"))
    Nerr.append(2000*dim[i])
    xtest.append(sampling[i](Nerr[i]))
    ftest.append(func[i](xtest[i]))
    xk.append(sampling[i](nt0))
    fk.append(func[i](xk[i]))
    gk.append(np.zeros([nt0,dim[i]]))
    for j in range(dim[i]):
        gk[i][:,j:j+1] = func[i](xk[i], j)

    model[i].set_training_values(xk[i], fk[i])
    for j in range(dim[i]):
        model[i].set_training_derivatives(xk[i], gk[i][:,j:j+1], j)
    model[i].train()

    NRMSE.append(rmse(model[i], func[i], N=Nerr[i], xdata=xtest[i], fdata=ftest[i]))


import pdb; pdb.set_trace()