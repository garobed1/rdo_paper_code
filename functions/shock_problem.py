"""
Implements the impinging shock openmdao problem as an smt Problem
"""
from cmath import cos, sin
import numpy as np
from mpi4py import MPI
import sys, time

import openmdao.api as om
from smt.problems.problem import Problem
from mphys_comp.impinge_analysis import Top
from utils.sutils import divide_cases, print_mpi
from utils.om_utils import map_om_to_smt, get_om_dict, get_om_design_size, om_dict_to_flat_array

import mphys_comp.impinge_setup as default_impinge_setup

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

class ImpingingShock(Problem):
    def _initialize(self):
        self.options.declare("ndim", 2, values=[2], types=int, desc='Does nothing for this problem')
        self.options.declare("ndv", 4, types=int, desc='Number of DVs to use')
        self.options.declare("E", 69000000000, types=(int,float), desc='Beam Stiffness (Pa)')
        self.options.declare("smax", 5e5, types=(int,float), desc='Maximum Stress Constraint for stresscon')
        self.options.declare("name", "ImpingingShock", types=str)
        
        self.options.declare("problem_settings", default=default_impinge_setup)
        self.options.declare("mesh", types=str, desc="Absolute filepath of the mesh to use")
        self.options.declare("inputs", ["dv_struct_TRUE", "shock_angle", "M0", "rsak"], types=list)
        self.options.declare("input_bounds", np.zeros([2,2]), types=np.ndarray)
        # self.options.declare("output", ["test.aero_post.cd_def"], types=list) #surrogate only returns the first element but we'll store the others
        self.options.declare("output", ["test.stresscon"], types=list) #surrogate only returns the first element but we'll store the others

        self.options.declare("comm", MPI.COMM_WORLD, types=MPI.Comm)



    def _setup(self):
        
        self.comm = self.options["comm"]
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # assert(self.options["ndim"] == len(self.options["inputs"]))

        # list of inputs we would need to finite difference
        # self.fdlist = ['M0', 'P0', 'T0', 'shock_angle'] #NOTE: Shouldn't need this anymore
        self.fdlist = [] # try this? and lower tolerance on solver

        ndv = self.options['ndv']
        E = self.options['E']
        smax = self.options['smax']

        # list of inputs we can get exact derivatives for
        #self.adlist = self.options["inputs"]
        self.adind = []
        self.adlist = []
        self.fdind = []
        j = 0
        for i in range(len(self.options["inputs"])):
            key = self.options["inputs"][i]
            if key == 'dv_struct_TRUE':
                for j in range(ndv):
                    self.adind.append(j)
                self.adlist.append(i)
            elif(key in self.fdlist):
                self.fdind.append(i+j)
            else:
                self.adind.append(i+j)
                self.adlist.append(i)

        # ensure we can get SA derivatives
        actual_settings = self.options["problem_settings"]

        saconsts = ['rsak','rsacb1','rsacb2','rsacb3','rsacv1','rsacw2','rsacw3','rsact1','rsact2','rsact3','rsact4','rsacrot']
        salist = []
        for key in self.options["inputs"]:
            if(key in saconsts):
                salist = salist + [key]
        actual_settings.aeroOptions["SAGrads"] = salist

        ntot = ndv + 2 #TODO: FIX
        self.options['ndim'] = ntot
        self.xlimits = np.zeros([ntot, 2])
        self.xlimits[:, 0] = self.options["input_bounds"][:,0]
        self.xlimits[:, 1] = self.options["input_bounds"][:,1]

        # further setting
        actual_settings.aeroOptions['NKSwitchTol'] = 1e-6 #1e-6
        # problem_settings.aeroOptions['NKSwitchTol'] = 1e-3 #1e-6
        # actual_settings.aeroOptions['nCycles'] = 5000000
        actual_settings.aeroOptions['nCycles'] = 500000
        actual_settings.aeroOptions['L2Convergence'] = 1e-12
        actual_settings.aeroOptions['printIterations'] = False
        actual_settings.aeroOptions['printTiming'] = False

        # mesh-based settings
        mx = int(self.options["mesh"].split('_')[-2])
        mstr = self.options["mesh"].split('_')[-4]
        if mx == 73 and mstr == 'mphys':
            nelem = 30
            L = .254
        if mx == 145 and mstr == 'long':
            nelem = 78
            L = .75
        if mx == 217 and mstr == 'long':
            nelem = 117
            L = .75
        aeroGridFile = self.options["mesh"]
        actual_settings.aeroOptions['gridFile'] = aeroGridFile

        # struct solver
        actual_settings.nelem = nelem
        actual_settings.structOptions['E'] = E
        actual_settings.structOptions['L'] = L
        actual_settings.structOptions['smax'] = smax
        actual_settings.structOptions['Nelem'] = nelem
        actual_settings.structOptions['force'] = np.ones(nelem+1)*1.0
        actual_settings.structOptions["th"] = np.ones(nelem+1)*0.006 #start high
        actual_settings.structOptions["ndv_true"] = self.options['ndv']
        actual_settings.structOptions["th_true"] = np.ones(ndv)*0.006
        self.problem_settings = actual_settings
        self.prob = om.Problem(comm=MPI.COMM_SELF)
        self.prob.model = Top(problem_settings=actual_settings, subsonic=False,
                                                     use_shock_comp=True, 
                                                     use_inflow_comp=True, 
                                                     full_free=False)

        # NOTE: shouldn't be necessary
        # # set up model inputs
        # dim = self.options["ndim"]
        # for i in range(dim):
        #     self.prob.model.add_design_var(self.options["inputs"][i], \
        #                             lower=self.xlimits[i,0], upper=self.xlimits[i,1])

        # for i in range(len(self.options["output"])):
        #     self.prob.model.add_objective(self.options["output"][i])

        self.prob.setup(mode='rev')

        # keep the current input state to avoid recomputing the gradient on subsequent calls
        self.xcur = np.zeros([1])
        self.fcur = None
        self.gcur = None

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape
        ninputs = len(self.options['inputs'])
        ndv = self.options['ndv']
        y = np.zeros((ne, 1), complex)
        h = 1e-8
        # dv_settings = self.prob.model.get_design_vars()
        #import pdb; pdb.set_trace()
        if(not np.array_equal(x, self.xcur)): # Don't recompute if we already have the answer for the given inputs
            self.xcur = x
            self.fcur = np.zeros((ne, 1), complex)
            self.gcur = np.zeros((ne, nx), complex)
            cases = divide_cases(ne, self.size)
            #for i in range(ne):
            for i in cases[self.rank]:
                # TODO: FIX THIS, EACH INPUT VAR SIZE
                # for j in range(nx):
                for j in range(ninputs):
                    if j == 0:
                        self.prob.set_val(self.options["inputs"][j], x[i, 0:ndv])
                    else:
                        self.prob.set_val(self.options["inputs"][j], x[i, ndv+(j-1)])
                # import pdb; pdb.set_trace()
                # for input in self.options['inputs']:
                #     vsize = self.prob.get_val(input).size
                #     self.prob.set_val(input, x[i, in_list])
                # import pdb; pdb.set_trace()
                self.prob.run_model()
                self.comm.barrier()
                if not self.prob.driver.fail:
                    self.fcur[i] = self.prob.get_val(self.options["output"][0])

                    #analytic
                    work = [self.options["inputs"][k] for k in self.adlist]
                    adgrads = self.prob.compute_totals(of=self.options["output"][0], wrt=work, return_format="array")
                    # import pdb; pdb.set_trace()
                    self.gcur[i][self.adind] = adgrads#[:,0]
                else:
                    self.fcur[i] = np.nan
                    self.gcur[i][self.adind] = np.nan

                
                
                #finite diff
                for key in self.fdind:
                    self.prob.set_val(self.options["inputs"][key], x[i, key] + h)
                    self.prob.run_model()
                    self.gcur[i][key] = (self.prob.get_val(self.options["output"][0]) - self.fcur[i])/h
                    self.prob.set_val(self.options["inputs"][key], x[i, key])


            self.fcur = self.comm.allreduce(self.fcur)
            self.gcur = self.comm.allreduce(self.gcur)

        for i in range(ne):
            if kx is None:
                y[i,0] = self.fcur[i]
            else:
                y[i,0] = self.gcur[i,kx]

        return y

