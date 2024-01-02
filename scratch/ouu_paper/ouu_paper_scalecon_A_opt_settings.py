# naming
name = 'ouu_paper_scalecon_A'
path = '.'


##### problem #####
"""
This function can serve as an objective or a constraint
Only support this one robust quantity for the time being
"""
u_dim = 5
d_dim = 1
prob = 'toylinear'
p_con = True
p_eq = None
p_ub = 0.
p_lb = None

# refinement threshold strategies
"""
0: Flat refinement
1: Refine by first-order proximity to 0
2: Refine by inexact gradient condition (Conn, Kouri papers)
"""
ref_strategy = 2

##### surrogate #####
use_truth_to_train = True #NEW, ONLY IF USING SURR
use_surrogate = True
full_surrogate = True
retain_uncertain_points = False
retain_uncertain_points_T = True

##### design noise idea ##### (not particularly relevant)
use_design_noise = False #NEW, ONLY IF USING SURR
design_noise = 0.0
design_noise_act = 0.0


##### trust region #####
trust_region_bound = 2    #NEW 1: use nonlinear sphere component
                            #  2: use dv bound constraints instead of nonlinear sphere
initial_trust_radius = 0.1 #"""We're making this relative to the design bound scale"""
xi = 0.1
# eta1
# eta2
# gamma1
# gamma2

##### optimization #####
x_init = 12.
# x_init = 5.5
inexact_gradient_only = False
approximate_model = True
approximate_truth = False
approximate_truth_max = 5000*u_dim
trust_increase_terminate = False
# gtol 
# stol
# xi

##### UQ Parameters #####
eta_use = 1.0
if not approximate_truth:
    N_t = approximate_truth_max
else:
    N_t = 100*u_dim
# N_t = 500*u_dim
N_m = 12
jump = 10
# model sampling
sample_type = 'Adaptive'
# truth sampling
sample_type_t = 'Normal'

##### Collocation UQ Parameters #####
sc_approximate_truth_max = 48
if not approximate_truth:
    scN_t = sc_approximate_truth_max
else:
    scN_t = 8
scN_m = 2
scjump = 1 # stochastic collocation jump

##### UQ Input PDFS #####
pdfs = [x_init] 

import numpy as np
sigtotal = 0.01
fac = 0
for i in range(u_dim):
    fac += 1./(i+1.)
A = sigtotal/fac

pdfs2 = []
for i in range(u_dim):
    sigs = np.sqrt(A/(i+1.))
    pdfs2 = pdfs2 + [['norm', 0., sigs]]# replace 2nd arg with the current design var

pdfs = pdfs2 + pdfs

##### Optimization options #####
max_outer = 20
opt_settings = {}
opt_settings['ACC'] = 1e-6



##### plotting #####
print_plots = True


