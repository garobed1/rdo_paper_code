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
title2 = sys.argv[2]
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
with open(f'{title2}/Adapterrors.pickle', 'rb') as f:
    Adapterr2 = pickle.load(f)
with open(f'{title}/samplehist.pickle', 'rb') as f:
    samplehist = pickle.load(f)

LHSerr = np.array(LHSerr, dtype=np.float64)
Adapterr = np.array(Adapterr, dtype=np.float64)
Adapterr2 = np.array(Adapterr2, dtype=np.float64)

itersk = len(samplehistK)
iters = len(samplehist)

nruns = len(Adapterr)

lhs0 = np.zeros(itersk)
lhs1 = np.zeros(itersk)
lhs2 = np.zeros(itersk)

ada0 = np.zeros(iters)
ada02 = np.zeros(iters)
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
            ada02[i] += Adapterr2[j,0,i]/(nruns)
        ada0[i] += Adapterr[j,0,i]/(nruns)
        ada1[i] += Adapterr[j,1,i]/nruns
        ada2[i] += Adapterr[j,2,i]/nruns

plt.figure(figsize=(6.5,5.2))
plt.subplots_adjust(bottom=0.13, left = 0.17, top=0.98, right=0.98)
ax = plt.gca()
# import pdb; pdb.set_trace()
sh = np.array(samplehist)
shk = np.array(samplehistK)
ax.semilogy(2*sh[:-1], ada0[:-1], "b-", label=f'POU Hess')
ax.semilogy(2*shk, lhs0, 'b--', label='POU LHS')
ax.semilogy(2*sh[:-1],ada02[:-1], "b-.", label=f'POU SFCV')
ax.semilogy(2*sh, ada2, 'r-', label=f'{alt_model[2]} Hess')
ax.semilogy(2*shk, lhs2, 'r--', label=f'{alt_model[2]} LHS ')
ax.semilogy(samplehist, ada1, 'g-', label=f'{alt_model[1]} Hess')
ax.semilogy(samplehistK, lhs1, 'g--',  label=f'{alt_model[1]} LHS')

#plt.fill_between(samplehistk, ehrm2 - ehrs2, ehrm2 + ehrs2, color='r', alpha=0.1)
plt.xlabel("Sampling Effort")
plt.ylabel("NRMSE")
ax.set_xlim(left=int(-0.25*sh[-1]*2))

plt.gca().set_ylim(top=10 ** math.ceil(math.log10(max([ada0[0], lhs0[0], ada1[0],lhs1[0],ada2[0],lhs2[0]]))))
plt.gca().set_ylim(bottom=10 ** math.floor(math.log10(min([ada0[-1], lhs0[-1], ada1[-1],lhs1[-1],ada2[-1],lhs2[-1]]))))
# plt.xticks(ticks=np.arange(min(samplehist), max(samplehist)+10, 60), labels=np.arange(min(samplehist), max(samplehist)+10, 60) )
plt.grid()
# ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
# ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
# ax.ticklabel_format(style='plain', axis='x')
plt.legend(loc="lower left", fontsize=11, borderaxespad=0, edgecolor="inherit")

nmax = int(iters)
nmaxk = int(itersk)
ax.axvline(nmax+20, color='k', linestyle='--', linewidth=1.2)
margins = ax.margins()
plt.savefig(f"{title}/{title}_err_nrmse_ensemble_shock_2D.pdf", bbox_inches="tight")
plt.clf()


boxes = []
blabels = []

boxes.append(Adapterr[:,0,int(nmax/2)].flatten())
blabels.append(f'POU\n Hess')
boxes.append(LHSerr[:,0,int(nmaxk/2)].flatten())
blabels.append(f'POU\n LHS')
boxes.append(Adapterr2[:,0,int(nmax/2)].flatten())
blabels.append(f'POU\n SFCV')
boxes.append(Adapterr[:,2,int(nmax/2)].flatten())
blabels.append(f'GEK\n Hess')
boxes.append(LHSerr[:,2,int(nmaxk/2)].flatten())
blabels.append(f'GEK\n LHS')
boxes.append(Adapterr[:,1,-1].flatten())
blabels.append(f'KRG\n Hess')
boxes.append(LHSerr[:,1,-1].flatten())
blabels.append(f'KRG\n LHS')

colordict = {
    "POU":"b",
    "POUHessian":"b",
    "GEKPLS":"r",
    "GEK1D":"r",
    "GEK":"r",
    "Kriging":"g",
    "PCEStrict":"c",
    "KRG":"g"
}

# import pdb; pdb.set_trace()
plt.figure(figsize=(6.5,5.2))
plt.subplots_adjust(bottom=0.13, left = 0.17, top=0.98, right=0.98)
plt.yscale('log')
plt.boxplot(boxes, labels=blabels)
boxplot = plt.boxplot(boxes, labels=blabels, patch_artist=True)
for i in range(len(boxplot['medians'])):
    # import pdb; pdb.set_trace()
    name = blabels[i].split('\n')[0]
    boxplot['medians'][i].set_color(colordict[name])
    boxplot['medians'][i].set_linewidth(2.0)
    boxplot['boxes'][i].set(fill=False)
# ax.set_xlabel("Sampling Effort")
plt.xticks(fontsize=12)
plt.ylabel("NRMSE")
# plt.yticks(plt.yticks()[0], 10.0**plt.yticks()[0])
plt.margins(*margins)
# ax.legend(loc=3)
plt.savefig(f"{title}/boxplot_ensemble_shock_2D.pdf", bbox_inches="tight")
# if dim > 2:
#     import pdb; pdb.set_trace()
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