import matplotlib.pyplot as plt
import openmdao.api as om
import numpy as np
import argparse
import os

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--optcase', action='store', help = 'case file name')

args = parser.parse_args()
optcase = args.optcase

# Instantiate your CaseReader
cr = []
# Plot the path the design variables took to convergence
# Note that there are two lines in the right plot because "Z"
# contains two variables that are being optimized
dv_values = []
obj_values = []
con_values = []


cr.append(om.CaseReader(optcase))

# also look for subsequent files
i = 1
get_more = True
while get_more:
    title = '.'.join(optcase.split('.')[:-1]) + f'_{i}.sql'
    if os.path.isfile(title):
        cr.append(om.CaseReader(title))
    else:
        get_more = False
    
    i += 1
# import pdb; pdb.set_trace()
for reader in cr:
    # Get driver cases (do not recurse to system/solver cases)
    driver_cases = reader.get_cases('driver', recurse=False)    
    for case in driver_cases:
        
        # dv_values.append(case.get_design_vars()['dv_struct_TRUE'][1])
        dv_values.append(case.get_design_vars()['x_d'][1])
        # obj_values.append(case.get_objectives()['test.mass'])
        obj_values.append(case.get_objectives()['mass_only.test.mass'])
        # con_values.append(case.get_constraints()['test.stresscon'])
        con_values.append(case.get_constraints()['stat.musigma'])

# import pdb; pdb.set_trace()
# import pdb; pdb.set_trace()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
fig.set_size_inches(6, 12)
# import pdb; pdb.set_trace()
ax1.plot(np.arange(len(dv_values)), np.array(obj_values))
ax1.set(ylabel='Mass Objective', title=f'Optimization History, {optcase[10:-4]}')
ax1.grid()

ax2.plot(np.arange(len(dv_values)), np.array(con_values))
ax2.set(ylabel='Max Stress KS Constraint')
ax2.grid()

ax3.plot(np.arange(len(dv_values)), np.array(dv_values))
ax3.set(xlabel='Iterations', ylabel='Thickness DV 2')
ax3.grid()

plt.savefig(f'{optcase[:-4]}.png', bbox_inches='tight')
plt.clf()

# final design

L = 1. # normalized length i guess
ns = 150
x = np.linspace(0., L, ns)
from beam.om_beamdvs import beamDVComp
import copy
fig, (ax1) = plt.subplots(1, 1)
fig.set_size_inches(12, 6)

# dv_init = cr[0].get_cases('driver', recurse=False)[0].get_design_vars()['dv_struct_TRUE']
dv_init = cr[0].get_cases('driver', recurse=False)[0].get_design_vars()['x_d']
# dv_final = case.get_design_vars()['dv_struct_TRUE']
dv_final = case.get_design_vars()['x_d']

# import pdb; pdb.set_trace()
ndv = dv_final.size
dvprob = om.Problem()
dvprob.model.add_subsystem('dvcomp', beamDVComp(ndv=ndv), promotes_outputs=['th'])
dvprob.model.add_subsystem('sink', om.ExecComp('y=th', shape=ns), promotes_inputs=['th'])
dvprob.setup()
dvprob.set_val('dvcomp.DVS', dv_final)
dvprob.run_model()
th_final = copy.deepcopy(dvprob.get_val('sink.y'))
dvprob.set_val('dvcomp.DVS', dv_init)
dvprob.run_model()
th_init = copy.deepcopy(dvprob.get_val('sink.y'))

plt.plot(x, np.zeros(th_init.shape[0]), 'k-')
plt.plot(x, -th_init, 'k-')
# plt.plot(x, th_final/2., 'r-')
plt.plot(x, -th_final, 'r-')
plt.grid()
plt.xlabel(rf'$s$')
plt.ylabel(rf'$y$')
plt.savefig(f'{optcase[:-4]}_final.png', bbox_inches='tight')
plt.clf()
# plt.show()