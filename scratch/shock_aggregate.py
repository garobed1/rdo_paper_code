import sys, os
import copy
import pickle
from matplotlib.transforms import Bbox
sys.path.insert(1,"../surrogate")

import numpy as np
import matplotlib.pyplot as plt
from infill.refinecriteria import looCV, HessianFit
from infill.aniso_criteria import AnisotropicRefine
from infill.getxnew import getxnew, adaptivesampling
from optimization.defaults import DefaultOptOptions
from utils.sutils import divide_cases
from utils.error import rmse, meane

from smt.problems import Branin, Sphere, LpNorm, Rosenbrock, WaterFlow, WeldedBeam, RobotArm, CantileverBeam
from smt.surrogate_models import KPLS, GEKPLS, KRG
#from smt.surrogate_models.rbf import RBF
from surrogate.pougrad import POUSurrogate
import matplotlib as mpl
from smt.sampling_methods import LHS

# Give directory with desired results as argument
usetead = False

title = "10000_shock_results"

if not os.path.isdir(title):
    os.mkdir(title)


plt.rcParams['font.size'] = '12'
#indlist = [[0, 96], [96, 192], [192, 288], [288, 384], [384,388], [388,580], [868,1156], [1156,1444],[2884, 3172], [3172, 5000]]

tot1 = 5500
jump1 = 250
nfiles1 = int(tot1/jump1)

tot2 = 6000
jump2 = 100
nfiles2 = int((tot2-tot1)/jump2)

tot3 = 9000
jump3 = 250
nfiles3 = int((tot3-tot2)/jump3)

tot4 = 9500
jump4 = 100
nfiles4 = int((tot4-tot3)/jump4)

tot5 = 10000
jump5 = 250
nfiles5 = int((tot5-tot4)/jump5)

indlist = [[i*jump1, (i+1)*jump1] for i in range(nfiles1)]
indlist.append([i*jump2, (i+1)*jump2] for i in range(nfiles2))
indlist.append([i*jump3, (i+1)*jump3] for i in range(nfiles3))
indlist.append([i*jump4, (i+1)*jump4] for i in range(nfiles4))
indlist.append([i*jump5, (i+1)*jump5] for i in range(nfiles5))


### X
with open(f'./{title}/x.npy', 'rb') as f:
    xreffull = pickle.load(f)

### F, G
fref = None
gref = None
xref = None
total = 0
for key in indlist:
    total += key[1] - key[0]
    xref = np.append(xref, xreffull[key[0]:key[1]])
    with open(f'./{title}/y{key[0]}to{key[1]}.npy', 'rb') as f:
        fref = np.append(fref, pickle.load(f))
    with open(f'./{title}/g{key[0]}to{key[1]}.npy', 'rb') as f:
        gref = np.append(gref, pickle.load(f))
# import pdb; pdb.set_trace()
xref = xref[1:]
fref = fref[1:]
gref = gref[1:]
xref = np.reshape(xref, [total, 2])
gref = np.reshape(gref, [total, 2])

with open(f'./{title}/xref.pickle', 'wb') as f:
    pickle.dump(xref, f)
with open(f'./{title}/fref.pickle', 'wb') as f:
    pickle.dump(fref, f)
with open(f'./{title}/gref.pickle', 'wb') as f:
    pickle.dump(gref, f)

