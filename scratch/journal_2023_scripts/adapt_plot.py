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
title = sys.argv[1]
alt_model = ['POU','KRG','GEK']#sys.argv[2]

plt.rcParams['font.size'] = '18'
plt.rc('legend',fontsize=14)

# Adaptive Data
with open(f'{title}/LHSerrors.pickle', 'rb') as f:
    LHSerr = pickle.load(f)
with open(f'{title}/samplehistK.pickle', 'rb') as f:
    samplehistK = pickle.load(f)

with open(f'{title}/Adapterrors.pickle', 'rb') as f:
    Adapterr = pickle.load(f)
with open(f'{title}/samplehist.pickle', 'rb') as f:
    samplehist = pickle.load(f)

LHSerr = np.array(LHSerr, dtype=np.float64)
Adapterr = np.array(Adapterr, dtype=np.float64)

itersk = len(samplehistK)
iters = len(samplehist)

nruns = len(Adapterr)

lhs0 = np.zeros(itersk)
lhs1 = np.zeros(itersk)
lhs2 = np.zeros(itersk)

ada0 = np.zeros(iters)
ada1 = np.zeros(iters)
ada2 = np.zeros(iters)

for i in range(itersk):
    for j in range(nruns):
        lhs0[i] += LHSerr[j,0,i]/nruns
        lhs1[i] += LHSerr[j,1,i]/nruns
        lhs2[i] += LHSerr[j,2,i]/nruns

for i in range(iters):
    for j in range(nruns):
        if(j != 17 and j != 45 and j != 38):
            ada0[i] += Adapterr[j,0,i]/(nruns-1)
        ada1[i] += Adapterr[j,1,i]/nruns
        ada2[i] += Adapterr[j,2,i]/nruns

# import pdb; pdb.set_trace()
ax = plt.gca()
plt.loglog(samplehist[:-1], ada0[:-1], "b-", label=f'Adapt (POU)')
#plt.fill_between(samplehist, ehrm - ehrs, ehrm + ehrs, color='b', alpha=0.2)
plt.loglog(samplehistK, lhs0, 'b--', label='LHS (POU)')
#plt.fill_between(samplehistk, ekrm - ekrs, ekrm + ekrs, color='b', alpha=0.1)
plt.loglog(samplehist, ada1, 'g-', label=f'Adapt ({alt_model[1]})')
#plt.fill_between(samplehistk, earm1 - ears1, earm1 + ears1, color='g', alpha=0.2)
plt.loglog(samplehistK, lhs1, 'g--',  label=f'LHS ({alt_model[1]})')
#plt.fill_between(samplehistk, ehrm1 - ehrs1, ehrm1 + ehrs1, color='g', alpha=0.1)
plt.loglog(samplehist, ada2, 'r-', label=f'Adapt ({alt_model[2]})')
#plt.fill_between(samplehistk, earm2 - ears2, earm2 + ears2, color='r', alpha=0.2)
plt.loglog(samplehistK, lhs2, 'r--', label=f'LHS ({alt_model[2]})')
#plt.fill_between(samplehistk, ehrm2 - ehrs2, ehrm2 + ehrs2, color='r', alpha=0.1)
plt.xlabel("Number of samples")
plt.ylabel("NRMSE")
plt.gca().set_ylim(top=10 ** math.ceil(math.log10(max([ada0[0], lhs0[0], ada1[0],lhs1[0],ada2[0],lhs2[0]]))))
plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(min([ada0[-1], lhs0[-1], ada1[-1],lhs1[-1],ada2[-1],lhs2[-1]]))))
plt.xticks(ticks=np.arange(min(samplehist), max(samplehist)+10, 60), labels=np.arange(min(samplehist), max(samplehist)+10, 60) )
plt.grid()
# ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
# ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
# ax.ticklabel_format(style='plain', axis='x')
plt.legend(loc=3)
plt.savefig(f"{title}/{title}_err_nrmse_ensemble_alt.pdf", bbox_inches="tight")
plt.clf()



# Plot Error contour
#contour
# ndir = 150
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
# cs = plt.tricontourf(xref[:,0], xref[:,1], fref, levels = 40)
# plt.colorbar(cs, aspect=20)
# plt.xlabel(r"$\theta_s$")
# plt.ylabel(r"$\kappa$")
# #plt.legend(loc=1)
# plt.savefig(f"{title}_true.pdf", bbox_inches="tight")
# plt.clf()