#!/usr/bin/env python3

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
from surrogate.pce_strict import PCEStrictSurrogate
import matplotlib as mpl
import matplotlib.ticker as mticker
from smt.sampling_methods import LHS

# Give directories with desired results as argument
titles = sys.argv[1:]

"adding comm line option to double x axis for gradient models"
fac = 2.0 # function+gradient costs this much
# if len(sys.argv) > 2:
#     fac = sys.argv[2]
# if len(sys.argv) > 3:
#     title2 = sys.argv[3]

if not os.path.isdir(titles[0].split("/")[0]):
    os.mkdir(titles[0])

prob = titles[0].split("_")[-2]
plt.rcParams['font.size'] = '16'


modelf = []
err0rms = []
hist = []
errhrms = []
xtrainK = []
ftrainK = []
gtrainK = []
errkrms = []

# Concatenate lists
xk = []
fk = []
gk = []
ekr = []
mf = []
e0r = []
hi = []
ehr = []
nprocs = []
mname = []
adapt = []


for i in range(len(titles)):
    # Adaptive Data
    with open(f'{titles[i]}/modelf.pickle', 'rb') as f:
        modelf.append(pickle.load(f))
    with open(f'{titles[i]}/errhrms.pickle', 'rb') as f:
        errhrms.append(pickle.load(f))


    # if we're dealing with strict SC
    if isinstance(modelf[i], PCEStrictSurrogate):
        with open(f'{titles[i]}/csamplers.pickle', 'rb') as f:
            hist.append(pickle.load(f))

        nprocs.append(1)
        mname.append(modelf[i].name)
        adapt.append("no_adapt")
        xk.append([]) 
        fk.append([]) 
        gk.append([]) 
        mf.append([]) 
        hi.append([]) 

        ehr.append([])
        ehr[i].append(errhrms[i])
        ekr.append(None)
        for j in range(nprocs[i]):
            xk[i].append(None)
            fk[i].append(None)
            gk[i].append(None)
            mf[i].append(modelf[i])
            hi[i].append([hist[i][k] for k in range(len(hist[i]))])


    else:
        with open(f'{titles[i]}/err0rms.pickle', 'rb') as f:
            err0rms.append(pickle.load(f))
        with open(f'{titles[i]}/hist.pickle', 'rb') as f:
            hist.append(pickle.load(f))
    
        # import pdb; pdb.set_trace()
        # LHS Data
        try:
            with open(f'{titles[i]}/xk.pickle', 'rb') as f:
                xtrainK.append(pickle.load(f))
            with open(f'{titles[i]}/fk.pickle', 'rb') as f:
                ftrainK.append(pickle.load(f))
            with open(f'{titles[i]}/gk.pickle', 'rb') as f:
                gtrainK.append(pickle.load(f))
            with open(f'{titles[i]}/errkrms.pickle', 'rb') as f:
                errkrms.append(pickle.load(f))
        except:
            xtrainK.append(None)
            ftrainK.append(None)
            gtrainK.append(None)
            errkrms.append(None)

        nprocs.append(len(modelf[i]))
        mname.append(modelf[i][0][0].name)
        adapt.append(titles[i].split("_")[2]) #should hit either hess or sfcv
        xk.append([]) 
        fk.append([]) 
        gk.append([]) 
        mf.append([]) 
        hi.append([]) 

        e0rc = np.atleast_2d(np.array(err0rms[i]).squeeze()).T
        ehrc = np.array(errhrms[i]).squeeze()
        ehr.append(np.append(e0rc, ehrc, axis = 1))
        ekr.append(np.array(errkrms[i]).squeeze())
        for j in range(nprocs[i]):
            xk[i].append(xtrainK[i][j][:])
            fk[i].append(ftrainK[i][j][:])
            gk[i].append(gtrainK[i][j][:])
            # ekr[i].append( errkrms[i][j][:])
            mf[i].append(modelf[i][j][:])
            # e0r[i].append(err0rms[i][j])
            hi[i].append(hist[i][j][:])
            # ehr[i].append(errhrms[i][j][:])
    

nmodels = len(mf)
nruns = len(mf[0])
dim = mf[0][0][0].training_points[None][0][0].shape[1]


# Problem Settings
trueFunc = GetProblem(prob, dim)


# for i in range(nmodels):
#     for j in range(nruns):
#         ehr[i][j] = [e0r[i][j]] + ehr[i][j] #[errf] #errh
# ekr = [ekr]
# ekm = [ekm]

# Plot Error History
# if(dim > 3):
#     with open(f'{title}/intervals.pickle', 'rb') as f:
#         intervals = pickle.load(f)
#     #iters = intervals.shape[0] + 1
# else:
# intervals = np.arange(iters)

# itersk = len(ekr[0])
# if(title2):
#     iterst = len(ehrt[0])
#     if(iterst < iters):
#         iters = iterst

# samplehist = np.zeros(iters, dtype=int)
# samplehistk = np.zeros(itersk, dtype=int)

# samplehist[0] = hi[0][0][0][0].shape[0] #training_points 
# for i in range(1, iters-1):
#     samplehist[i] = samplehist[i-1] + (intervals[1] - intervals[0])
# samplehist[iters-1] = mf[0].training_points[None][0][0].shape[0]
# for i in range(itersk):
#     samplehistk[i] = len(xk[i])

sh = []
shk = []
iters = []
for i in range(nmodels):
    iters.append(len(ehr[i][0]))
    if not isinstance(modelf[i], PCEStrictSurrogate):
        iters[i] = min(iters[i], len(hi[i][0][0]))
    else:
        iters[i] = min(iters[i], len(hi[i][0]))
    sh.append(np.zeros(iters[i], dtype=int))
    # import pdb; pdb.set_trace()
    for j in range(iters[i]):
        if not isinstance(modelf[i], PCEStrictSurrogate):
            sh[i][j] = hi[i][0][0][j][0][0].shape[0]
        else:
            # import pdb; pdb.set_trace()
            sh[i][j] = hi[i][0][j].current_samples['x'].shape[0]
    try:
        itersk = len(ekr[i][0])
        shk.append(np.zeros(itersk, dtype=int))
        for j in range(itersk):
            shk[i][j] = xk[i][0][j].shape[0]
    except:
        shk.append(None)

# Average out runs
ehrm = []
ekrm = []
for i in range(nmodels):
    if not isinstance(modelf[i], PCEStrictSurrogate):
        ehrm.append(np.mean(ehr[i][:,:iters[i]], axis=0))
        try:
            ekrm.append(np.mean(ekr[i], axis=0))
        except:
            ekrm.append(None)
    else:
        # import pdb; pdb.set_trace()
        ehrm.append(ehr[i][0])
        

#NRMSE
# need key for coloring, model settings
colordict = {
    "POU":"b",
    "POUHessian":"b",
    "GEK":"r",
    "GEKPLS":"r",
    "GEK1D":"r",
    "KRG":"g",
    "Kriging":"g",
    "PCEStrict":"c",
}

namedict = {
    "POUHessian":"POU",
    "GEKPLS":"GEK",
    "GEK1D":"GEK",
    "Kriging":"KRG",
    "PCEStrict":"SC",
}

adaptdict = {
    "hess":"Hess",
    "sfcv":"SFCVT",
    "no_adapt":" "
}

adaptdict2 = {
    "hess":"-",
    "sfcv":"-.",
    "no_adapt":"-"
}

facdict = {
    "POUHessian":fac,
    "GEKPLS":fac,
    "GEK1D":fac,
    "Kriging":1,
    "PCEStrict":1,
}





plt.figure(figsize=(6.5,5.2))
plt.subplots_adjust(bottom=0.13, left = 0.17, top=0.98, right=0.98)
ax = plt.gca()
# axb = 
lhslist = []
for i in range(nmodels):
    ax.semilogy(sh[i]*facdict[mname[i]], 
                ehrm[i], 
                f"{colordict[mname[i]]}{adaptdict2[adapt[i]]}", 
                label=f'{namedict[mname[i]]} {adaptdict[adapt[i]]}')
    try:
        if(f'{namedict[mname[i]]} LHS' not in lhslist):
            ax.semilogy(shk[i]*facdict[mname[i]], 
                        ekrm[i], 
                        f'{colordict[mname[i]]}--', 
                        label=f'{namedict[mname[i]]} LHS')
            lhslist.append(f'{namedict[mname[i]]} LHS')
    except:
        print("No LHS for this one")
ax.set_xlabel("Sampling Effort")
ax.set_ylabel("NRMSE")
ax.set_xlim(left=int(-0.25*sh[0][-1]*facdict[mname[0]]))
# ax.set_ylim(top=10 ** math.ceil(math.log10(ehrm[0][0])))
# ax.set_ylim(bottom=10 ** math.floor(math.log10(ehrm[-1][-1])))
# plt.xticks(ticks=np.arange(min(samplehist), max(samplehist), 40), labels=np.arange(min(samplehist), max(samplehist), 40) )
ax.grid()
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.legend(loc="best", fontsize=11, borderaxespad=0, edgecolor="inherit")


# determine appropriate boxplot data before finishing plot
# need largest common index
# ignore lhs for now
imax = np.zeros(nmodels, dtype=int)
imaxk = np.zeros(nmodels, dtype=int)
if nmodels > 1:
    nmax = int(sh[0][-1]*facdict[mname[0]])
    for i in range(nmodels):
        if not isinstance(modelf[i], PCEStrictSurrogate):
            nmax = min(nmax, sh[i][-1]*facdict[mname[i]])
    for i in range(nmodels):
        if not isinstance(modelf[i], PCEStrictSurrogate):

            # import pdb; pdb.set_trace()
            try:
                imax[i] = np.where(sh[i]*int(facdict[mname[i]]) == nmax)[0][0]
            except: #round to nearest even number if there's no match
                try:
                    imax[i] = np.where(sh[i]*int(facdict[mname[i]]) == math.ceil(nmax/ 2.) * 2)[0][0]
                except:
                    # import pdb; pdb.set_trace()
                    imax[i] = np.where(sh[i]*int(facdict[mname[i]]) == math.floor(nmax/ 2.) * 2)[0][0]       
        try:
            if(f'{namedict[mname[i]]} LHS' in lhslist):
                imaxk[i] = np.argmin(abs(nmax-shk[i]*int(facdict[mname[i]])))
                nmaxk = shk[i][imaxk[i]]
        except:
            print("No LHS for this one (boxplot 1)")
else:
    nmax = sh[0][-1]
    imax[0] = sh[0].shape[0]-1
    try:
        nmaxk = shk[0][-1]
        imaxk[0] = shk[0].shape[0]-1
    except:
        print("No LHS for this one (boxplot 2)")




ax.axvline(nmax+1, color='k', linestyle='--', linewidth=1.2)
margins = ax.margins()
plt.savefig(f"{titles[0]}/err_nrmse_ensemble_{prob}_{dim}D.pdf")#, bbox_inches="tight")
bbox = plt.figure().get_tightbbox(plt.figure().canvas.get_renderer())
# size = plt.figure().get_size_inches()*plt.figure().dpi
figsize = plt.figure().get_size_inches()#.transformed(plt.figure().dpi_scale_trans.inverted())
plt.clf()





boxes = []
blabels = []
lhslist2 = []
# import pdb; pdb.set_trace()
for i in range(nmodels):
    if not isinstance(modelf[i], PCEStrictSurrogate):
        # boxes.append(np.log10(ehr[i][:,imax[i]]))
        boxes.append(ehr[i][:,imax[i]])
        blabels.append(f'{namedict[mname[i]]}\n {adaptdict[adapt[i]]}')
    try:
        if(f'{namedict[mname[i]]} LHS' in lhslist and f'{namedict[mname[i]]}\n LHS' not in lhslist2):
            boxes.append(ekr[i][:,imaxk[i]])
            blabels.append(f'{namedict[mname[i]]}\n LHS')
            lhslist2.append(f'{namedict[mname[i]]}\n LHS')
    except:
        print("No LHS for this one (boxplot 3)")

# import pdb; pdb.set_trace()
plt.figure(figsize=(6.5,5.2))
plt.subplots_adjust(bottom=0.13, left = 0.17, top=0.98, right=0.98)


# plt.figure(figsize=figsize)
plt.yscale('log')
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
# plt.figure().set_size_inches(size)

# plt.subplots_adjust(bottom=0.15)
plt.savefig(f"{titles[0]}/boxplot_ensemble_{prob}_{dim}D_{nmax+1}.pdf")#, bbox_inches='tight')
# if dim > 2:
#     import pdb; pdb.set_trace()
plt.clf()



# # # Plot points

# if(dim == 1):
#     plt.clf()
#     nt0 = samplehist[0]
#     # Plot Training Points
#     plt.plot(trx[0:nt0,0], np.zeros(trx[0:nt0,0].shape[0]), "bo", label='Initial Samples')
#     plt.plot(trx[nt0:,0], np.zeros(trx[nt0:,0].shape[0]), "ro", label='Adaptive Samples')
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$f$")
#     #plt.legend(loc=1)
#     plt.savefig(f"{title}/1d_adaptive_pts.pdf", bbox_inches="tight")#"tight")
#     plt.clf()

#     ndir = 75
#     xlimits = trueFunc.xlimits
#     x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)

#     Z = np.zeros([ndir])
#     Zh = np.zeros([ndir])
#     F = np.zeros([ndir])
#     Fh = np.zeros([ndir])
#     TF = np.zeros([ndir])

#     for i in range(ndir):
#         xi = np.zeros([1,1])
#         xi[0] = x[i]
#         TF[i] = trueFunc(xi)
#         F[i] = pmod.predict_values(xi)
#         Fh[i] = mk.predict_values(xi)
#         Z[i] = abs(F[i] - TF[i])
#         Zh[i] = abs(Fh[i] - TF[i])

#     # Plot the target function
#     plt.plot(x, TF, "-k", label=f'True')
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$f$")
#     #plt.legend(loc=1)
#     plt.savefig(f"{title}/1dtrue.pdf", bbox_inches="tight")
#     plt.clf()

#     # Plot Non-Adaptive Error
#     plt.plot(x, TF, "-k", label=f'True')
#     # plt.plot(x, Fgek, "-m", label=f'IGEK')
#     plt.plot(x, F, "-b", label=f'AS')
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$f$")
#     #plt.legend(loc=1)
#     plt.plot(trx[0:nt0,0], trf[0:nt0,0], "bo", label='Initial Samples')
#     plt.plot(trx[nt0:,0], trf[nt0:,0], "ro", label='Adaptive Samples')
#     plt.savefig(f"{title}/1dplot.pdf", bbox_inches="tight")
#     plt.clf()

#     # Plot Non-Adaptive Error
#     # plt.plot(x, Zgek, "-m", label=f'IGEK')
#     plt.plot(x, Z, "-k", label=f'Adaptive (POU)')
#     plt.plot(x, Zh, "--k", label=f'LHS (POU)')
#     plt.xlabel(r"$x$")
#     plt.ylabel(r"$|\hat{f}(\mathbf{x}) - f(\mathbf{x})|$")
#     plt.plot(trx[0:nt0,0], np.zeros_like(trf[0:nt0,0]), "bo", label='Initial Samples')
#     plt.plot(trx[nt0:,0], np.zeros_like(trf[nt0:,0]), "ro", label='Added Samples')
#     plt.plot(trxk, max(np.max(Z), np.max(Zh))*np.ones_like(trxk), "ko", label='LHS Samples')
#     plt.legend(fontsize='13')
#     plt.savefig(f"{title}/1derr.pdf", bbox_inches="tight")

#     plt.clf()

# just the first one
plotthis = False
if(dim == 2 and plotthis):

    plt.clf()
    nt0 = sh[0][0]
    # Plot Training Points
    trx = modelf[0][0][0].training_points[None][0][0]
    plt.plot(trx[0:nt0,0], trx[0:nt0,1], "bo", label='Initial Samples')
    plt.plot(trx[nt0:,0], trx[nt0:,1], "ro", label='Adaptive Samples')
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.savefig(f"{titles[0]}/2d_adaptive_pts.pdf", bbox_inches="tight")#"tight")
    plt.clf()


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
            F[j,i]  = modelf[0][0][0].predict_values(xi)
            TF[j,i] = trueFunc(xi)
            Za[j,i] = abs(F[j,i] - TF[j,i])

    # Plot original function
    # cs = plt.contourf(X, Y, TF, levels = 40)
    # plt.colorbar(cs, aspect=20)
    # plt.xlabel(r"$x_1$")
    # plt.ylabel(r"$x_2$")
    # #plt.legend(loc=1)
    # plt.savefig(f"{titles[0]}/2d_true.pdf", bbox_inches="tight")

    # plt.clf()

    cs = plt.contourf(X, Y, F, levels = 40)
    plt.colorbar(cs, aspect=20, label = r"$|\hat{f}(\mathbf{x}) - f(\mathbf{x})|$")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(trx[0:nt0,0], trx[0:nt0,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.plot(trx[nt0:,0], trx[nt0:,1], "o", fillstyle='full', markerfacecolor='r', markeredgecolor='r', label='Adaptive Samples')
    plt.savefig(f"{titles[0]}/2d_model_a.pdf", bbox_inches="tight")

    cs = plt.contourf(X, Y, Za, levels = 40)
    plt.colorbar(cs, aspect=20, label = r"$|\hat{f}(\mathbf{x}) - f(\mathbf{x})|$")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.legend(loc=1)
    plt.plot(trx[0:nt0,0], trx[0:nt0,1], "o", fillstyle='full', markerfacecolor='b', markeredgecolor='b', label='Initial Samples')
    plt.plot(trx[nt0:,0], trx[nt0:,1], "o", fillstyle='full', markerfacecolor='r', markeredgecolor='r', label='Adaptive Samples')
    plt.savefig(f"{titles[0]}/2d_errcon_a.pdf", bbox_inches="tight")

    plt.clf()










# Nerr = 5000
# sampling = LHS(xlimits=trueFunc.xlimits, criterion='m')
# xtest = sampling(Nerr)
# ftest = trueFunc(xtest)
# meantrue = sum(ftest)/Nerr
# stdtrue = np.sqrt((sum(ftest*ftest)/Nerr) - (sum(ftest)/Nerr)**2)

# meanlhstrue = sum(fk[0])/fk[0].shape[0]
# stdlhstrue = np.sqrt((sum(fk[0]*fk[0])/fk[0].shape[0]) - (sum(fk[0])/fk[0].shape[0])**2)

# faiges = mf[0].predict_values(xtest)
# meanaiges = sum(faiges)/Nerr
# stdaiges = np.sqrt((sum(faiges*faiges)/Nerr) - (sum(faiges)/Nerr)**2)

# ftead = mft[0].predict_values(xtest)
# meantead = sum(ftead)/Nerr
# stdtead = np.sqrt((sum(ftead*ftead)/Nerr) - (sum(ftead)/Nerr)**2)


# mf[0].set_training_values(xk[0], fk[0])
# if(isinstance(mf[0], GEKPLS) or isinstance(mf[0], POUSurrogate)):
#     for i in range(dim):
#         mf[0].set_training_derivatives(xk[0], gk[0][:,i:i+1], i)
# mf[0].train()
# flhs = mf[0].predict_values(xtest)

# meanlhs = sum(flhs)/Nerr
# stdlhs  = np.sqrt((sum(flhs*flhs)/Nerr) - (sum(flhs)/Nerr)**2)


# print("True Mean: ", meantrue)
# print("True LHS Mean: ", meanlhstrue)
# print("LHS Mean: ", meanlhs)
# print("AIGES Mean: ", meanaiges)
# print("TEAD Mean: ", meantead)
# print("True std: ", stdtrue)
# print("True LHS std: ", stdlhstrue)
# print("LHS std: ", stdlhs)
# print("AIGES std: ", stdaiges)
# print("TEAD std: ", stdtead)