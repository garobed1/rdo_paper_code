# naming
name = 'ouu_paper_rosencirc_A'
path = '.'


##### problem #####
"""
This function can serve as an objective or a constraint
Only support this one robust quantity for the time being
"""
u_dim = 1
d_dim = 2
prob = 'uellipse_rad'
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
x_init = [-1., 1.]
inexact_gradient_only = False
approximate_model = True
approximate_truth = False
approximate_truth_max = 5000*u_dim
trust_increase_terminate = False
tol_ignore_sdist = True
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
N_m = 10
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
import numpy as np
pdfs = [0., 0., ['norm', np.sqrt(1), 0.2]] #sigma = 0.2 # replace 2nd arg with the current design var
# pdfs = ['uniform', 0.] # replace 2nd arg with the current design var

# in case we use a beta dist, modify function limits with this
beta_bounds=[]
beta_bounds.append([0., 1.0])

##### Optimization options #####
max_outer = 20
opt_settings = {}
opt_settings['ACC'] = 1e-6



##### plotting #####
print_plots = True




# last used settings before transitioning to this
"""
##### surrogate #####
use_truth_to_train = True #NEW, ONLY IF USING SURR
use_surrogate = False
full_surrogate = True
retain_uncertain_points = True

##### design noise idea ##### (not particularly relevant)
use_design_noise = False #NEW, ONLY IF USING SURR
design_noise = 0.0
design_noise_act = 0.0


##### trust region #####
trust_region_bound = 2    #NEW 1: use nonlinear sphere component
                            #  2: use dv bound constraints instead of nonlinear sphere
initial_trust_radius = 1.0
# eta1
# eta2
# gamma1
# gamma2

##### optimization #####
x_init = 5.
# gtol 
# stol
# xi

##### UQ Parameters #####
u_dim = 1
eta_use = 1.0
N_t = 5000*u_dim
# N_t = 500*u_dim
N_m = 2
jump = 100
sample_type = 'Adaptive'

##### Collocation UQ Parameters #####
scN_t = 48
scN_m = 2
scjump = 1 # stochastic collocation jump


##### Adaptive Surrogate UQ Parameters #####
RC0 = None
max_batch = sset.max_batch

##### plotting #####
print_plots = True

# set up beta test problem parameters
dim = 2
prob = 'betatestex'
# pdfs = ['uniform', 0.] # replace 2nd arg with the current design var

"""