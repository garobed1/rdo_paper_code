
runs_per_proc = 1

# Surrogate Settings
stype = "pouhess"   #surrogate type

### FOR POU HESS
rtype =  "hessgrad"
opt = 'L-BFGS-B' #'SLSQP'#
local = True
gopt = 'ga' #'brute'
localswitch = True
# POU Params
rscale = 5.5
# rho = 5000. # NOTE: Only applies to criteria
rho = 50. # NOTE: Only applies to criteria
# rho2 = 5000. # NOTE: USE FOR ENERGY ONLY?
# NOTE: We'll use a separate rho = 5000 for energy

rsca = True  # hess function value scale
vsca = False # hess cell volume scale

### FOR POU SFCVT
# rtype =  "pousfcvt"
# opt = 'SLSQP' #for SFCVT constraint
# local = True
# localswitch = True

### FOR REGULAR SFCVT
# rtype =  "sfcvt"
# opt = 'SLSQP' #for SFCVT constraint, 
# local = False
# localswitch = False #fully global optimizer

# Kriging params
corr  = "squar_exp"  #kriging correlation
poly  = "quadratic"    #kriging regression
delta_x = 1e-4 #1e-7
# extra = dim           #gek extra points, set in script
t0 = [1e-0]
tb = [1e-6, 2e+1]


# rho = 10

# Adaptive Sampling Settings
nt0  = 10       #initial design size
ntr = 40      #number of points to add
ntot = nt0 + ntr    #total number of points


batch = 5#dim*2        #batch size for refinement, as a percentage of ntr
# batch = 5#dim*2        #batch size for refinement, as a percentage of ntr
min_contribution=1e-14 # for pou hess

mstarttype = 2            # 0: No multistart
                          # 1: Start at the best out of a number of samples
                          # 2: Perform multiple optimizations
if(mstarttype == 1):   
    multistart = 50#*dim multiply by dim in script
if(mstarttype == 2):
    multistart = 5#*dim multiply by dim in script


# Other Refinement Settings
neval_add = 3 #applies to both criteria and POU model
neval_fac = 1 #applies to both criteria and POU model
hess  = "neighborhood"
interp = "honly"
criteria = "distance"
perturb = True
bpen = False
obj = "inv"
nscale = 10.0 #1.0 for 2D

# nmatch = dim

# Print RC plots
rc_print = False
