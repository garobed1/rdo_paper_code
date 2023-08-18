import sys, os
import shutil
import copy
import pickle
import importlib
from mpi4py import MPI

import numpy as np
#import matplotlib.pyplot as plt
from infill.refinecriteria import looCV, HessianFit, TEAD
from infill.aniso_criteria import AnisotropicRefine
from infill.taylor_criteria import TaylorRefine, TaylorExploreRefine
from infill.hess_criteria import HessianRefine, POUSSA
from infill.loocv_criteria import POUSFCVT, SFCVT
from infill.aniso_transform import AnisotropicTransform
from infill.getxnew import getxnew, adaptivesampling
from optimization.defaults import DefaultOptOptions
from utils.sutils import divide_cases, convert_to_smt_grads
from utils.error import rmse, meane
from functions.problem_picker import GetProblem

from smt.surrogate_models import KPLS, GEKPLS, KRG
from surrogate.gek_1d import GEK1D
from surrogate.direct_gek import DGEK
#from smt.surrogate_models.rbf import RBF
from surrogate.pougrad import POUSurrogate, POUHessian
from smt.sampling_methods import LHS
from scipy.stats import qmc
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
Perform adaptive sampling and estimate error

Instead of fanning out cases, fan out batch requested sampling instead

We want this to work for optimization
"""

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true') 

parser.add_argument('-d', '--configfile', action='store', default='results_settings.py', help = 'python file containing settings for running results')
parser.add_argument('-r', '--restartdir', action='store', default=None, help = 'directory containing existing results, restart from there')
parser.add_argument('-n', '--numadd', action='store', default=0, help = 'if non zero, override number to add from settings')

args = parser.parse_args()
verbose = args.verbose
settings = args.configfile
fulltitle = args.restartdir
ntro = args.numadd

fresh = True
# If restart folder is given, start from there
if fulltitle is not None:
    fresh = False
    tsplit = fulltitle.split('/')
    if len(tsplit) == 1:
        path = "."
    else:
        path = '/'.join(tsplit[:-1])
    title = tsplit[-1]

    set = importlib.import_module(f"{path}.{title}.settings")
    if(path == None):
        path = "."

    #need to load the current model
    with open(f'{path}/{title}/modelf.pickle', 'rb') as f:
        model0lists = pickle.load(f)
    if not isinstance(model0lists, list):
        model0 = model0lists
    else:
        model0 =  model0lists[0][0]

else:
    # All variables not initialized come from this import
    ssplit = settings.split(sep='/')
    suse = '.'.join(ssplit)
    if suse.endswith('.py'):
        suse = suse.split('.')[:-1]
        suse = '.'.join(suse)
    set = importlib.import_module(suse)
    # from results_settings import * 
    # Generate results folder and list of inputs
    title = f"{set.header}_{set.prob}_{set.dim}D"
    if(set.path == None):
        path = "."
    else:
        path = set.path
    if rank == 0:
        if not os.path.isdir(f"{path}/{title}"):
            os.mkdir(f"{path}/{title}")
        shutil.copy(f"./{settings}", f"{path}/{title}/settings.py")

if ntro:
    ntr = ntro
else:
    ntr = set.ntr

if fresh:
    nt0 = set.nt0
else:
    nt0 = model0.training_points[None][0][0].shape[0]


# Problem Settings
ud = set.perturb
trueFunc = GetProblem(set.prob, set.dim, use_design=ud)
xlimits = trueFunc.xlimits
sampling = LHS(xlimits=xlimits, criterion='m')
if(set.rtype == 'anisotransform'):
    sequencer = qmc.Halton(d=set.dim)


# Error
xtest = None 
ftest = None
testdata = None

try:
    chkpt = set.chkpt
except:
    chkpt = 1*set.batch
intervals = np.arange(0, ntr, chkpt)

if(set.prob != 'shock'):
    if rank == 0:
        xtest = sampling(set.Nerr)
        ftest = trueFunc(xtest)

        testdata = [xtest, ftest, intervals]

    xtest = comm.bcast(xtest, root=0)
    ftest = comm.bcast(ftest, root=0)
    testdata = comm.bcast(testdata, root=0)


# Adaptive Sampling Conditions
options = DefaultOptOptions
options["local"] = set.local
options["localswitch"] = set.localswitch
options["errorcheck"] = testdata
options["multistart"] = set.mstarttype
options["lmethod"] = set.opt
try:
    options["method"] = set.gopt
except:
    pass

# Print Conditions
if rank == 0:
    print("\n")
    print("\n")
    print("Surrogate Type       : ", set.stype)
    print("Refinement Type      : ", set.rtype)
    print("Refinement Multistart: ", set.multistart)
    print("Correlation Function : ", set.corr)
    print("Regression Function  : ", set.poly)
    print("GEK Extra Points     : ", set.extra)
    print("Problem              : ", set.prob)
    print("Problem Dimension    : ", set.dim)
    print("Initial Sample Size  : ", nt0)
    print("Refined Points Size  : ", ntr)
    print("Total Points         : ", nt0+ntr)
    print("Points Per Iteration : ", set.batch)
    print("RMSE Size            : ", set.Nerr)
    print("\n")
    print("Computing Initial Designs for Adaptive Sampling ...")

# Adaptive Sampling Initial Design
if fresh:
    xtrain0 = sampling(nt0)
    ftrain0 = trueFunc(xtrain0)
    gtrain0 = convert_to_smt_grads(trueFunc, xtrain0)
else:
    xtrain0 = model0.training_points[None][0][0]
    ftrain0 = model0.training_points[None][0][1]
    gtrain0 = convert_to_smt_grads(trueFunc, xtrain0)


idx = np.round(np.linspace(0, len(intervals)-1, set.LHS_batch+1)).astype(int)
intervalsk = intervals[idx]
samplehistK = intervalsk + nt0*np.ones(len(intervalsk), dtype=int)
samplehistK[-1] += 1

if rank == 0:
    print("Computing Non-Adaptive Designs ...")

if not set.skip_LHS:
    # Final Design(s)
    xtrainK = []
    ftrainK = []
    gtrainK = []
    for m in range(len(samplehistK)):
        xtrainK.append(sampling(samplehistK[m]))
        ftrainK.append(trueFunc(xtrainK[m]))
        gtrainK.append(convert_to_smt_grads(trueFunc, xtrainK[m]))

if rank == 0:
    print("Training Initial Surrogate ...")

# Initial Design Surrogate
if fresh:
    if(set.stype == "gekpls"):
        if(set.dim > 1):
            modelbase = GEKPLS(xlimits=xlimits)
            # modelbase.options.update({"hyper_opt":'TNC'})
            modelbase.options.update({"n_comp":set.dim})
            modelbase.options.update({"extra_points":set.extra})
            modelbase.options.update({"delta_x":set.delta_x})
            if(set.dim > 2):
                modelbase.options.update({"zero_out_y":True})
        else: # to get gek runs to work in 1D
            modelbase = GEK1D(xlimits=xlimits)
            #modelgek.options.update({"hyper_opt":"TNC"})
        modelbase.options.update({"theta0":set.t0})
        modelbase.options.update({"theta_bounds":set.tb})
        modelbase.options.update({"corr":set.corr})
        modelbase.options.update({"poly":set.poly})
        modelbase.options.update({"n_start":5})

    elif(set.stype == "dgek"):
        modelbase = DGEK(xlimits=xlimits)
        # modelbase.options.update({"hyper_opt":'TNC'})
        modelbase.options.update({"theta0":set.t0})
        modelbase.options.update({"theta_bounds":set.tb})
        modelbase.options.update({"corr":set.corr})
        modelbase.options.update({"poly":set.poly})
        modelbase.options.update({"n_start":5})
    elif(set.stype == "pou"):
        modelbase = POUSurrogate()
        modelbase.options.update({"rho":set.rho})
    elif(set.stype == "pouhess"):
        modelbase = POUHessian(bounds=xlimits, rscale=set.rscale)
        modelbase.options.update({"rho":set.rho})
        modelbase.options.update({"neval":set.neval})
    elif(set.stype == "kpls"):
        modelbase = KPLS()
        # modelbase.options.update({"hyper_opt":'TNC'})
        modelbase.options.update({"n_comp":set.dim})
        modelbase.options.update({"corr":set.corr})
        modelbase.options.update({"poly":set.poly})
        modelbase.options.update({"n_start":5})
    else:
        modelbase = KRG()
        # modelbase.options.update({"hyper_opt":'TNC'})
        modelbase.options.update({"corr":set.corr})
        modelbase.options.update({"poly":set.poly})
        modelbase.options.update({"n_start":5})
    modelbase.options.update({"print_global":False})


else:
    modelbase = copy.deepcopy(model0)

model0 = copy.deepcopy(modelbase)
model0.set_training_values(xtrain0, ftrain0)
convert_to_smt_grads(model0, xtrain0, gtrain0)
model0.train()

# Initial Model Error
if(set.prob != 'shock'):
    if rank == 0:
        print("Computing Initial Surrogate Error ...")
    err0rms = rmse(model0, trueFunc, N=set.Nerr, xdata=xtest, fdata=ftest)
    err0mean = meane(model0, trueFunc, N=set.Nerr, xdata=xtest, fdata=ftest)

if rank == 0:
    print("Computing Final Non-Adaptive Surrogate Error ...")

if(not set.skip_LHS):
    # Non-Adaptive Model Error
    modelK = copy.deepcopy(modelbase)
    errkrms = []
    errkmean = []
    xtrainK[0] = xtrain0
    ftrainK[0] = ftrain0
    gtrainK[0] = gtrain0

    for m in range(len(samplehistK)):
        try:
            modelK.set_training_values(xtrainK[m], ftrainK[m])
            convert_to_smt_grads(modelK, xtrainK[m], gtrainK[m])
            modelK.train()
            
            errkrms.append(rmse(modelK, trueFunc, N=set.Nerr, xdata=xtest, fdata=ftest))
            errkmean.append(meane(modelK, trueFunc, N=set.Nerr, xdata=xtest, fdata=ftest))
        except:
            print("LHS run started failing!")
            continue
    
    # LHS Data
    with open(f'{path}/{title}/xk.pickle', 'wb') as f:
        pickle.dump(xtrainK, f)

    with open(f'{path}/{title}/fk.pickle', 'wb') as f:
        pickle.dump(ftrainK, f)

    with open(f'{path}/{title}/gk.pickle', 'wb') as f:
        pickle.dump(gtrainK, f)

    with open(f'{path}/{title}/errkrms.pickle', 'wb') as f:
        pickle.dump(errkrms, f)

    with open(f'{path}/{title}/errkmean.pickle', 'wb') as f:
        pickle.dump(errkmean, f)


if rank == 0:
    print("Initial Refinement Criteria ...")

# Initial Refinement Criteria
if(set.rtype == "aniso"):
    RC0 = AnisotropicRefine(model0, gtrain0, xlimits, rscale=set.rscale, nscale=set.nscale, improve=set.pperb, neval=set.neval, hessian=set.hess, interp=set.interp, bpen=set.bpen, objective=set.obj, multistart=set.multistart)
elif(set.rtype == "anisotransform"):
    RC0 = AnisotropicTransform(model0, sequencer, gtrain0, improve=set.pperb, nmatch=set.nmatch, neval=set.neval, hessian=set.hess, interp=set.interp)
elif(set.rtype == "tead"):
    RC0 = TEAD(model0, gtrain0, xlimits, gradexact=True)
elif(set.rtype == "taylor"):
    RC0 = TaylorRefine(model0, gtrain0, xlimits, volume_weight=set.perturb, rscale=set.rscale, improve=set.pperb, multistart=set.multistart)
elif(set.rtype == "taylorexp"):
    RC0 = TaylorExploreRefine(model0, gtrain0, xlimits, rscale=set.rscale, improve=set.pperb, objective=set.obj, multistart=set.multistart)
elif(set.rtype == "hess"):
    RC0 = HessianRefine(model0, gtrain0, xlimits, neval=set.neval, rscale=set.rscale, improve=set.pperb, multistart=set.multistart, print_rc_plots=set.rc_print)
elif(set.rtype == "poussa"):
    RC0 = POUSSA(model0, gtrain0, xlimits, improve=set.pperb, multistart=set.multistart, print_rc_plots=set.rc_print)
elif(set.rtype == "pousfcvt"):
    RC0 = POUSFCVT(model0, gtrain0, xlimits, improve=set.pperb, multistart=set.multistart, print_rc_plots=set.rc_print)
elif(set.rtype == "sfcvt"):
    RC0 = SFCVT(model0, gtrain0, xlimits,  print_rc_plots=set.rc_print) # improve=pperb, multistart=multistart, not implemented
else:
    raise ValueError("Given criteria not valid.")

if rank == 0:
    print("Performing Adaptive Sampling ...")

# Perform Adaptive Sampling
mf, rF, hf, ef, ef2 = adaptivesampling(trueFunc, model0, RC0, xlimits, ntr, batch=set.batch, options=options)
modelf = mf
RCF = rF
hist = hf
errhrms = ef
errhmean = ef2

if rank == 0:
    print("\n")
    print("Experiment Complete")

    if(fresh):
        affix = ""
    else:
        affix = f"_{nt0}"

    # Adaptive Data
    with open(f'{path}/{title}/modelf.pickle', 'wb') as f:
        pickle.dump(modelf, f)

    with open(f'{path}/{title}/err0rms{affix}.pickle', 'wb') as f:
        pickle.dump(err0rms, f)

    with open(f'{path}/{title}/err0mean{affix}.pickle', 'wb') as f:
        pickle.dump(err0mean, f)

    with open(f'{path}/{title}/hist{affix}.pickle', 'wb') as f:
        pickle.dump(hist, f)

    with open(f'{path}/{title}/errhrms{affix}.pickle', 'wb') as f:
        pickle.dump(errhrms, f)

    with open(f'{path}/{title}/errhmean{affix}.pickle', 'wb') as f:
        pickle.dump(errhmean, f)

    with open(f'{path}/{title}/intervals{affix}.pickle', 'wb') as f:
        pickle.dump(intervals, f)
