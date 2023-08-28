import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import openmdao.api as om
from uq_comp.stat_comp_comp import StatCompComponent

"""
run a mean plus variance optimization over the 1D-1D test function, pure MC
for now
"""

# from optimization.optimizers import optimize
from pyoptsparse import Optimization, SLSQP
from functions.problem_picker import GetProblem
from utils.error import stat_comp
from optimization.robust_objective import RobustSampler
from optimization.defaults import DefaultOptOptions

plt.rcParams['font.size'] = '22'



# set up robust objective UQ comp parameters
u_dim = 1
eta_use = 1.0
N = 5000*u_dim


# set up beta test problem parameters
dim = 2
prob = 'betatestex'
# pdfs = ['uniform', 0.] # replace 2nd arg with the current design var

func = GetProblem(prob, dim)
xlimits = func.xlimits
# start at just one point for now
x_init = 5.


# run optimizations
x_init = 5.
# args = (func, eta_use)
xlimits_d = np.zeros([1,2])
xlimits_d[:,1] = 10.

pdfs = [['beta', 3., 1.], 0.] # replace 2nd arg with the current design var
sampler = RobustSampler(np.array([x_init]), N, xlimits=xlimits, probability_functions=pdfs, retain_uncertain_points=True)

# openmdao
prob = om.Problem()
prob.model.add_subsystem("stat", StatCompComponent(sampler=sampler,
                                 stat_type="mu_sigma", 
                                 pdfs=pdfs, 
                                 eta=eta_use, 
                                 func=func))
# prob.model = StatCompComponent(sampler=sampler,
#                                  stat_type="mu_sigma", 
#                                  pdfs=pdfs, 
#                                  eta=eta_use, 
#                                  func=func)
prob.driver = om.pyOptSparseDriver() #Default: SLSQP
prob.driver.hist_file = "./robust_opt_pyopt_plots/robust_opt_pyopt_histfile1"

prob.model.add_design_var("stat.x_d", lower=xlimits_d[0,0], upper=xlimits_d[0,1])
prob.model.add_objective("stat.musigma")

prob.setup()
prob.set_val("stat.x_d", x_init)
prob.run_driver()
x_opt_1 = prob.get_val("stat.x_d")[0]

# plot robust func
ndir = 150
x = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
y = np.zeros([ndir])
for j in range(ndir):
    prob.set_val("stat.x_d", x[j])
    prob.run_model()
    y[j] = prob.get_val("stat.musigma")

# Plot original function
cs = plt.plot(x, y)
plt.xlabel(r"$x_d$")
plt.ylabel(r"$\mu_f(x_d)$")
plt.axvline(x_init, color='k', linestyle='--', linewidth=1.2)
plt.axvline(x_opt_1, color='r', linestyle='--', linewidth=1.2)
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_pyopt_plots/objrobust1_true.pdf", bbox_inches="tight")
plt.clf()

# plot beta dist
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.zeros([ndir])
beta_handle = beta(pdfs[0][1],pdfs[0][2])
for j in range(ndir):
    y[j] = beta_handle.pdf(x[j])
cs = plt.plot(x, y)
plt.xlabel(r"$x_d$")
plt.ylabel(r"$\mu_f(x_d)$")
plt.axvline(0.75, color='r', linestyle='--', linewidth=1.2)
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_pyopt_plots/betadist1_true.pdf", bbox_inches="tight")
plt.clf()





# # second opt, need to re-setup
# pdfs = [['beta', 1., 3.], 0.]
# sampler = RobustSampler(np.array([x_init]), N, xlimits=xlimits, probability_functions=pdfs)
# prob.model.stat.options["pdfs"] = pdfs
# prob.model.stat.options["sampler"] = sampler
# prob.driver.hist_file = "./robust_opt_pyopt_plots/robust_opt_pyopt_histfile2"
# prob.setup()
# prob.set_val("stat.x_d", x_init)
# prob.run_driver()
# x_opt_2 = prob.get_val("stat.x_d")[0]

# fun plots
ndir = 150
x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
X, Y = np.meshgrid(x, y)
TF = np.zeros([ndir, ndir])
for i in range(ndir):
    for j in range(ndir):
        xi = np.zeros([1,2])
        xi[0,0] = x[i]
        xi[0,1] = y[j]
        TF[j,i] = func(xi)
# Plot original function
cs = plt.contourf(X, Y, TF, levels = 40)
plt.colorbar(cs, aspect=20, label = r"$f(x_u, x_d)$")
plt.xlabel(r"$x_u$")
plt.ylabel(r"$x_d$")
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_pyopt_plots/betarobust_true.pdf", bbox_inches="tight")
plt.clf()

# plot robust func
x = np.linspace(xlimits[1][0], xlimits[1][1], ndir)
y = np.zeros([ndir])
for j in range(ndir):
    prob.set_val("stat.x_d", x[j])
    prob.run_model()
    y[j] = prob.get_val("stat.musigma")
# Plot original function
cs = plt.plot(x, y)
plt.xlabel(r"$x_d$")
plt.ylabel(r"$\mu_f(x_d)$")
plt.axvline(x_init, color='k', linestyle='--', linewidth=1.2)
# plt.axvline(x_opt_2, color='r', linestyle='--', linewidth=1.2)
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_pyopt_plots/objrobust2_true.pdf", bbox_inches="tight")
plt.clf()

x = np.linspace(xlimits[0][0], xlimits[0][1], ndir)
y = np.zeros([ndir])
beta_handle = beta(pdfs[0][1],pdfs[0][2])
for j in range(ndir):
    y[j] = beta_handle.pdf(x[j])
cs = plt.plot(x, y)

plt.xlabel(r"$x_d$")
plt.ylabel(r"$\mu_f(x_d)$")
plt.axvline(0.25, color='r', linestyle='--', linewidth=1.2)
#plt.legend(loc=1)
plt.savefig(f"./robust_opt_pyopt_plots/betadist2_true.pdf", bbox_inches="tight")
plt.clf()

import pdb; pdb.set_trace()