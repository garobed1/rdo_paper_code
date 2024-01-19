import numpy as np
import copy
import collections

from optimization.optimizers import optimize
from optimization.robust_objective import RobustSampler
from optimization.opt_subproblem import OptSubproblem
from utils.om_utils import get_om_design_size, om_dict_to_flat_array, grad_opt_feas
from utils.sutils import print_mpi
from optimization.trust_bound_comp import TrustBound
from collections import OrderedDict
import pickle
import os, sys
from smt.utils.options_dictionary import OptionsDictionary

import openmdao.api as om
from openmdao.utils.mpi import MPI

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

con_test = 1

"""
Trust-region approach where m_k is a UQ evaluation (surrogate, SC, etc.) of a 
certain level of fidelity.

"""
class UncertainTrust(OptSubproblem):
    def __init__(self, **kwargs):


        super().__init__(**kwargs)
        self.trust_opt = 0
        self.grad_lhs = []
        self.grad_rhs = []
        self.reflog = []
        self.models = []
        self.radii = []
        self.realizations = []
        self.areds = []
        self.preds = []
        self.loc = []
        self.reflevel = []
        self.duals = None

    def _declare_options(self):
        
        super()._declare_options()
        declare = self.options.declare
        declare(
            "initial_trust_radius", 
            default=0.1, 
            types=float,
            desc="trust radius at the first iteration"
        )
        declare(
            "trust_option", 
            default=1, 
            types=int,
            desc="1: use sphere component, 2: use dv bounds (linear)"
        )


        declare(
            "gamma_1", 
            default=0.5, 
            types=float,
            desc="coefficient for decreasing radius"
        )

        declare(
            "gamma_2", 
            default=2.0, 
            types=float,
            desc="coefficient for increasing radius"
        )

        declare(
            "xi", 
            default=0.1, 
            # default=1.00, 
            types=float,
            desc="coefficient for inexact gradient condition"
        )


        declare(
            "max_trust_radius", 
            default=1000.0, 
            types=float,
            desc="maximum trust radius, no steps taken larger than this"
        )

        declare(
            "flat_refinement", 
            default=5, 
            types=int,
            desc="Flat refinement amount to apply at each outer iter"
        )

        declare(
            "use_truth_to_train", 
            default=False, 
            types=bool,
            desc="If the model uses a surrogate, add the truth evaluations to its training data"
        )

        """
        Error Estimate Options

        By default, use the difference between the model and computed truth gradient to refine the model
        to satisfy the inexact gradient conditions. What we really want is to use the 
        adaptive refinement criteria to approximate this instead
        """
        declare(
            "model_grad_err_est", 
            default=False, 
            types=bool,
            desc="Use an estimate of the model gradient error to drive inexact gradient satisfaction"
        )

        """
        By default, use the difference between the computed truth function and the actual truth function
        to satisfy the predicted reduction error tolerance. What we really want is to use the
        adaptive refinement criteria to approximate this instead (huge potential savings!!!)
        """
        declare(
            "truth_func_err_est", 
            default=False, 
            types=bool,
            desc="Use an estimate of the validation function error to drive validation refinement"
        )
        declare(
            "no_stall", 
            default=True, 
            types=bool,
            desc="Do not stall out the rhs recomputation if it matches the previous iteration closely"
        )
        declare(
            "truth_func_err_est_max", 
            default=5000, 
            types=int,
            desc="Maximum validation sampling level"
        )

        declare(
            "trust_increase_terminate", 
            default=False, 
            types=bool,
            desc="if trust model predicts increase, terminate"
        )

        declare(
            "inexact_gradient_only", 
            default=False, 
            types=bool,
            desc="if true, do not determine convergence based on truth model gradient, use the inexact gradient condition only"
        )

        declare(
            "tol_ignore_sdist", 
            default=False, 
            types=bool,
            desc="if true, instead of min(gerr, sdist), just do gerr"
        )

        declare(
            "min_dual", 
            default=1e-2, 
            types=float,
            desc="for robust constraints, if dual for it is zero or smaller than this, scale the validation tolerance to ensure some refinement"
        )
        
        declare(
            "ref_strategy", 
            default=0, 
            types=int,
            desc="""
                 0: Flat refinement
                 1: Refine by first-order proximity to 0
                 2: Refine by inexact gradient condition (Conn, Kouri papers)
                 """
        )

    def _setup_final(self):
        """
        Configure the trust region settings here

        """
        self.trust_opt = self.options["trust_option"]

        if self.trust_opt == 1 or self.trust_opt == 0:
            # need list of dvs now
            dv_settings = {}
            #TODO: Need to automate this somehow
            dv_settings["x_d"] = OrderedDict({'name':'x_d', 'size':1, 'distributed':False})
            self.prob_model.model.add_subsystem('trust', 
                                      TrustBound(dv_dict=dv_settings, 
                                                    initial_trust_radius=self.options['initial_trust_radius']), 
                                                    promotes_inputs=list(dv_settings.keys()))
            self.prob_model.model.add_constraint('trust.c_trust', lower=0.0)
        elif self.trust_opt == 2:
            pass
        else:
            Exception("No trust region type specified!")

    def _save_iteration(self, iter):

        title = self.title
        path = self.path


        if rank == 0:
            with open(f'{path}/{title}/optimality_{iter}.pickle', 'wb') as f:
                pickle.dump(self.gerr, f)
            with open(f'{path}/{title}/feasability_{iter}.pickle', 'wb') as f:
                pickle.dump(self.gfea, f)
            with open(f'{path}/{title}/grad_lhs_{iter}.pickle', 'wb') as f:
                pickle.dump(self.grad_lhs[-1], f)
            with open(f'{path}/{title}/grad_rhs_{iter}.pickle', 'wb') as f:
                pickle.dump(self.grad_rhs[-1], f)
            with open(f'{path}/{title}/radii_{iter}.pickle', 'wb') as f:
                pickle.dump(self.radii[-1], f)
            with open(f'{path}/{title}/realizations_{iter}.pickle', 'wb') as f:
                pickle.dump(self.realizations[-1], f)
            with open(f'{path}/{title}/areds_{iter}.pickle', 'wb') as f:
                pickle.dump(self.areds[-1], f)
            with open(f'{path}/{title}/preds_{iter}.pickle', 'wb') as f:
                pickle.dump(self.preds[-1], f)
            with open(f'{path}/{title}/loc_{iter}.pickle', 'wb') as f:
                pickle.dump(self.loc[-1], f)
            with open(f'{path}/{title}/models_{iter}.pickle', 'wb') as f:
                pickle.dump(self.models[-1], f)
            with open(f'{path}/{title}/reflog_{iter}.pickle', 'wb') as f:
                pickle.dump(self.reflog[-1], f)
            with open(f'{path}/{title}/prob_truth_points_{iter}.pickle', 'wb') as f:
                pickle.dump(self.prob_truth.model.stat.sampler.current_samples, f)
            with open(f'{path}/{title}/prob_model_points_{iter}.pickle', 'wb') as f:
                pickle.dump(self.prob_model.model.stat.sampler.current_samples, f)
            with open(f'{path}/{title}/duals_{iter}.pickle', 'wb') as f:
                pickle.dump(self.duals, f)

            # surrogate evaluation points, for consistency
            with open(f'{path}/{title}/surr_eval.pickle', 'wb') as f:
                pickle.dump(self.prob_model.model.stat.surr_eval, f)

            # so we know where to start from
            with open(f'{path}/{title}/previous_iter.pickle', 'wb') as f:
                pickle.dump(iter, f)

    # search for the last iteration and load that
    def _load_last_iteration(self):

        title = self.title
        path = self.path

        k = 0
        if os.path.isfile(f'{path}/{title}/previous_iter.pickle'):

            # if rank == 0:
            with open(f'{path}/{title}/previous_iter.pickle', 'rb') as f:
                k = pickle.load(f)
        
            # get surrogate evaluations
            with open(f'{path}/{title}/surr_eval.pickle', 'rb') as f:
                self.prob_model.model.stat.surr_eval = pickle.load(f)

            # check if refining as well
            with open(f'{path}/{title}/refining.pickle', 'rb') as f:
                refining = pickle.load(f)
            

            k = comm.bcast(k)
            refining = comm.bcast(refining)


            self._load_iteration(k)

            k += 1

        # elif os.path.isfile(f'{path}/{title}/first_iter.pickle'): #in case 

        # return the outer iteration counter
        return k

    # load information from the last iteration and start from there
    def _load_iteration(self, iter):

        title = self.title
        path = self.path

        # if rank == 0:
        with open(f'{path}/{title}/optimality_{iter}.pickle', 'rb') as f:
            self.gerr = pickle.load(f)
        with open(f'{path}/{title}/feasability_{iter}.pickle', 'rb') as f:
            self.gfea = pickle.load(f)
        with open(f'{path}/{title}/grad_lhs_{iter}.pickle', 'rb') as f:
            self.grad_lhs.append(pickle.load(f))
        with open(f'{path}/{title}/grad_rhs_{iter}.pickle', 'rb') as f:
            self.grad_rhs.append(pickle.load(f))
        with open(f'{path}/{title}/radii_{iter}.pickle', 'rb') as f:
            self.radii.append(pickle.load(f))
        with open(f'{path}/{title}/realizations_{iter}.pickle', 'rb') as f:
            self.realizations.append(pickle.load(f))
        with open(f'{path}/{title}/areds_{iter}.pickle', 'rb') as f:
            self.areds.append(pickle.load(f))
        with open(f'{path}/{title}/preds_{iter}.pickle', 'rb') as f:
            self.preds.append(pickle.load(f))
        with open(f'{path}/{title}/loc_{iter}.pickle', 'rb') as f:
            self.loc.append(pickle.load(f))
        with open(f'{path}/{title}/duals_{iter}.pickle', 'rb') as f:
            self.duals = pickle.load(f)
        with open(f'{path}/{title}/models_{iter}.pickle', 'rb') as f:
            self.models.append(pickle.load(f))
        with open(f'{path}/{title}/reflog_{iter}.pickle', 'rb') as f:
            self.reflog.append(pickle.load(f))
        # with open(f'{path}/{title}/prob_truth_points_{iter}.pickle', 'rb') as f:
        #     # pickle.dump(self.prob_truth.model.stat.sampler.current_samples, f)
        #     ptcur = pickle.load(f)
            
        # rebuild sampler object
        dvsettings = self.prob_model.model.get_design_vars()
        dvsize = get_om_design_size(dvsettings)
        for i in range(iter + 1):

            with open(f'{path}/{title}/loc_{i}.pickle', 'rb') as f:
                # pickle.dump(self.prob_model.model.stat.sampler.current_samples, f)
                lcur = pickle.load(f)

            with open(f'{path}/{title}/prob_model_points_{i}.pickle', 'rb') as f:
                # pickle.dump(self.prob_model.model.stat.sampler.current_samples, f)
                pmcur = pickle.load(f)

            self.prob_model.model.stat.sampler.initialize(reset=True)
            self.prob_model.model.stat.sampler.set_design(om_dict_to_flat_array(lcur, dvsettings, dvsize))
            self.prob_model.model.stat.sampler.add_data(pmcur, replace_current=True)
        
        self.prob_model.model.stat.surrogate = self.models[-1]

    def solve_full(self):


        """
        PARAMS:

        tru : truth function
        mdl : model function

        delta_k : current trust radius
        eta_k : acceptance measure, (tru(z_k) - tru(z_k + s_k))/(mdl_k(z_k) - mdl_k(z_k + s_k))
        eta_k+1 : prediction of next step's acceptance measure, if k step is accepted, with updated model
        eta_0 : acceptance measure threshold
        eta_1 : if eta_k+1 is lower, shrink the trust radius
        eta_2 : if eta_k+1 is higher, increase the trust radius
        
        # NOTE: Scipy either doubles or quarters the radius, hard coded
        # Make these options like Kouri (2013) paper
        gamma_1 : trust radius retraction coeff for failed step
        gamma_2 : trust radius modifier coeff for successful step

        STEPS:

        1. Solve subproblem for s_k
        2. Check acceptance theshold, eta_eval \geq eta_0, decide to move or not
        3. Refine model according to gradient difference, some level of adding points
        4. Update trust radius based on success of step and model refinement

        3. to 4. is the RTR algorithm from Kouri (2013)
        4. to 3. instead is the CTR algorithm
        
        """

        self.gtol = self.options['gtol']
        stol = self.options['stol']
        self.ctol = self.options['ctol'] # feasability
        miter = self.options['max_iter']
        max_trust_radius = self.options['max_trust_radius']
        initial_trust_radius = self.options['initial_trust_radius']
        eta_0 = self.options['eta']
        eta_1 = self.options['eta_1']
        eta_2 = self.options['eta_2']
        gamma_1 = self.options['gamma_1']
        gamma_2 = self.options['gamma_2']
        self.xi = self.options["xi"]
        self.no_stall = self.options["no_stall"]
        

        if not (0 <= eta_0 < 1.0):
            raise Exception('invalid acceptance stringency')
        if not (0 <= eta_1 < 1.0 and 0 <= eta_2 < 1.0 and eta_1 < eta_2):
            raise Exception('invalid radius update stringencies')
        # if not (0 < gamma_1 < 1.0 and 0 < gamma_2 < 1.0 and gamma_1 <= gamma_2):
        #     raise Exception('invalid radius update coefficients')
        if max_trust_radius <= 0:
            raise Exception('the max trust radius must be positive')
        if initial_trust_radius <= 0:
            raise ValueError('the initial trust radius must be positive')
        if initial_trust_radius >= max_trust_radius:
            raise ValueError('the initial trust radius must be less than the '
                         'max trust radius')


        """
        Check for problem constraints
        
        """
        self.have_cons = False
        if self.prob_model.driver._cons:
            self.have_cons = True

        # optimization index
        k = 0
   
        # ensure that we are not in refining mode to start
        refining = False

        # flagged for failed optimization
        fail = 0
        succ = 0
        
        # design variable meta settings
        dvsettings = self.prob_model.model.get_design_vars()
        dvsize = get_om_design_size(dvsettings)


        # region movement index (only incremented if we accept a step)
        # deprecated use
        tsteps = 0

        if self.options["print"]:
            fetext = '-'
            getext = '-'

        # get DV bounds
        dvbl = {}
        dvbu = {}
        dvsc = {}
        for name in dvsettings:
            dvbl[name] = copy.deepcopy(dvsettings[name]['lower'])
            dvbu[name] = copy.deepcopy(dvsettings[name]['upper'])
            dvsc[name] = dvbu[name] - dvbl[name]


        ###
        # try loading the previous state
        ###
        k = self._load_last_iteration()

        # if no previous iterations were found, start from scratch
        if k == 0:
            # initial guess in dict format, whatever the current state of the OM system is
            z0 = self.prob_model.driver.get_design_var_values()

            # store error measures, including the first one
            gerr0 = 0
            ftru = None
            ferr = 1e6
            self.gerr = 1e6


            # we assume that fidelity belongs to the top level system
            # calling it stat for now
            self.reflevel.append(self.prob_model.model.stat.get_fidelity())

            # initialize trust radius
            trust_radius = initial_trust_radius
            zk = z0

            # validate the first point before starting
            if not self.options["model_grad_err_est"]:
                self._eval_truth(zk)
                ftru = copy.deepcopy(self.prob_truth.get_val(self.prob_outs[0]))

                # Lagrangian, Optimality, Feasability
                if con_test:
                    self.gtru, self.gerr, self.gfea, dummy = grad_opt_feas(self.prob_truth, self.have_cons, self.ctol)
                else:
                    self.gtru = self.prob_truth.compute_totals(return_format='array')
                    self.gerr = np.linalg.norm(self.gtru)

                gerr0 += self.gerr
                self.grange = gerr0 - self.gtol
                if self.options["print"]:
                    getext = str(self.gerr)

            self.grad_lhs.append(None)
            self.grad_rhs.append(None)
            self.radii.append(trust_radius)
            self.realizations.append(self.prob_model.model.stat.get_fidelity())
            self.areds.append(None)
            self.preds.append(None)
            self.loc.append(zk)

        else:
            # NOTE: WONT WORK WITHOUT model_grad_err_est

            # last design iteration
            zk = self.loc[-1]
            om_dict_to_flat_array
            trust_radius = self.radii[-1]


            # we assume that fidelity belongs to the top level system
            # calling it stat for now
            self.reflevel.append(self.prob_model.model.stat.get_fidelity())

            # store error measures, including the first one
            gerr0 = 0
            ftru = None
            ferr = 1e6
            # gmod, self.gerr, self.gfea, duals = grad_opt_feas(self.prob_model, self.have_cons, self.ctol)

        print_mpi(f"___________________________________________________________________________")
        print_mpi(f"Optimization Parameters")
        print_mpi(f"Current Step               = {k}")
        print_mpi(f"Gradient Tolerance         = {self.gtol}")
        print_mpi(f"Step Tolerance             = {stol}")
        print_mpi(f"Maximum Outer Iterations   = {miter}")
        print_mpi(f"Initial Trust Radius       = {initial_trust_radius}")
        print_mpi(f"Number of Design Variables = {dvsize}")
        print_mpi(f"Number of UQ Variables     = {self.prob_model.model.stat.sampler.x_u_dim}")


        print_mpi(f"___________________________________________________________________________")
        print_mpi(f"Beginning Full Optimization-Under-Uncertainty Loop")



        # =================================================================
        # Begin
        # =================================================================
        while (self.gerr > self.gtol) and (k < miter):
            fail = 0
            if self.options["print"]:
                print_mpi("\n")
                print_mpi(f"Outer Iteration {k} ")
                print_mpi(f"Trust Region {tsteps} ")
                print_mpi(f"-------------------")
                print_mpi(f"    OBJ ERR: {fetext}")
                # Add constraint loop as well
                print_mpi(f"    -")
                print_mpi(f"    GRD ERR: {getext}")
                print_mpi(f"    Fidelity: {self.reflevel[-1]}")
                
                self.prob_model.list_problem_vars()
            
                print_mpi(f"___________________________________________________________________________")
                print_mpi(f"")
                print_mpi(f"OOO     Step 1: Solving subproblem...")
            # =================================================================
            # 1.
            #   find a solution to the subproblem
            # =================================================================

            # 
            zk_cent = copy.deepcopy(zk)
            # 
            #TODO: SCALED TRUST REGION
            if self.trust_opt == 1:
                self.prob_model.model.trust.set_center(zk_cent)
            else: #BOX, self.trust_opt == 2
                for name in dvsettings:
                    llmt = np.maximum(zk_cent[name] - trust_radius*dvsc[name], dvbl[name])
                    ulmt = np.minimum(zk_cent[name] + trust_radius*dvsc[name], dvbu[name])
                    #TODO: Annoying absolute name path stuff
                    self.prob_model.model.set_design_var_options(name.split('.')[-1], lower=llmt, upper=ulmt)


            ### THE SUBPROBLEM SOLVE    
            # 
            # self.prob_model.model.stat.check_partials_flag = True
            # self.prob_model.check_partials()
            # self.prob_model.model.stat.check_partials_flag = False
            # 
            fmod_cent = copy.deepcopy(self.prob_model.get_val(self.prob_outs[0]))
            self._solve_subproblem(zk)  
            fmod_cand = copy.deepcopy(self.prob_model.get_val(self.prob_outs[0]))
            zk_cand = self.prob_model.driver.get_design_var_values()
            # self.prob_model.model.stat.check_partials_flag = True
            # self.prob_model.check_partials()
            # self.prob_model.model.stat.check_partials_flag = False
            # 

            # compute predicted reduction
            predicted_reduction = fmod_cent - fmod_cand #kouri method
            # predicted_reduction = ftru_cent - fmod_cand #TODO: how to handle this?
            self.pred = predicted_reduction
            
            # retrieve gradients information
            # 
            # this needs to be the lagrangian gradient with constraints
            
            if con_test:
                gmod, gerrm, gfeam, duals = grad_opt_feas(self.prob_model, self.have_cons, self.ctol)
                gfeam = abs(gfeam)
            else:
                gmod = self.prob_model.compute_totals(return_format='array')
                gerrm = np.linalg.norm(gmod)
            
        
            # need distance of prediction, and if its on edge of radius
            # the dv arrays originate from OrderedDict objects, so this should be fine
            zce_arr = om_dict_to_flat_array(zk_cent, dvsettings, dvsize)
            zca_arr = om_dict_to_flat_array(zk_cand, dvsettings, dvsize)
            sc_arr = om_dict_to_flat_array(dvsc, dvsettings, dvsize)
            s = zca_arr - zce_arr
            s_sc = np.zeros_like(s)
            for i in range(s_sc.size):
                s_sc[i] = s[i]/sc_arr[i]
            self.sdist = np.linalg.norm(s)
            self.sdist_sc = np.linalg.norm(s_sc)
            if self.options["print"]:
                print_mpi(f"o       Step Size = {self.sdist}, Relative = {self.sdist_sc}, Step Tol = {stol}")
            
            

            if self.options["model_grad_err_est"]:
                # Compare to refined model
                if self.options["print"]:
                    print_mpi(f"___________________________________________________________________________")
                    print_mpi(f"")
                    print_mpi(f"OOO     Step 2: Validating by Refining Model...")
                
                # Estimate the inexact gradient condition now rather than later
                # lhs0 = lhs = self.prob_model.model.stat.sampler.rcrit.get_energy()#np.linalg.norm(gmod-gtru)
                lhs0 = None
                rhs0 = min(gerrm, self.sdist) #use unscaled sdist, since the lagrangian won't be scaled either
                if self.options["tol_ignore_sdist"]:
                     rhs0 = gerrm
                # xi_calc = lhs0/rhs0

                # if we time out during refinement, ensure that we know that when restarting
                refining = True
                if rank == 0:
                    with open(f'{self.path}/{self.title}/refining.pickle', 'wb') as f:
                        pickle.dump(refining, f)

                lhs, rhs = self.model_refiner(lhs0, rhs0)

                refining = False
                if rank == 0:
                    with open(f'{self.path}/{self.title}/refining.pickle', 'wb') as f:
                        pickle.dump(refining, f)
                


                # recompute model
                self.prob_model.run_model()
                fmod_cand_star = copy.deepcopy(self.prob_model.get_val(self.prob_outs[0]))

                actual_reduction = fmod_cent - fmod_cand_star
            
            
            else:
                #Eval Truth
                if self.options["print"]:
                    print_mpi(f"___________________________________________________________________________")
                    print_mpi(f"")
                    print_mpi(f"OOO     Step 2: Validating by Computing Truth...")

                ftru_cent = copy.deepcopy(self.prob_truth.get_val(self.prob_outs[0]))
                self._eval_truth(zk_cand)
                ftru_cand = copy.deepcopy(self.prob_truth.get_val(self.prob_outs[0]))

                # compute actual/approximate reduction
                actual_reduction = ftru_cent - ftru_cand
            self.ared = actual_reduction
        
            # =================================================================
            # 3.
            #   compute and compare the improvement ratio to the acceptance threshold
            # =================================================================
            
            if self.options["print"]:
                print_mpi(f"___________________________________________________________________________")
                print_mpi(f"")
                print_mpi(f"OOO     Step 3: Validating Subproblem Reduction with Truth...")
            
            
            

            if predicted_reduction <= 0 and self.options['trust_increase_terminate']:
                fail = 2
                break
            # if predicted_reduction <= 0:
            eta_k = actual_reduction/predicted_reduction

            # choose to accept or not
            # NOTE: Forget this, no point
            accept = False
            if eta_k > eta_0:
                tsteps += 1
                zk = zk_cand
                accept = True
            else:
                zk = zk_cand
                accept = True
                # zk = zk_cent

            #NOTE: this may be classic, as it is now, or retroactive, based on
            # updated/refined model, need to make this an option
            eta_k_act = eta_k




            # =================================================================
            # 4.
            #   adjust trust radius
            # =================================================================
            
            if self.options["print"]:
                print_mpi(f"___________________________________________________________________________")
                print_mpi(f"")
                print_mpi(f"OOO     Step 4: Modifying Trust Bounds...")
                print_mpi(f"O       Prior Radius = {trust_radius}")
            
            # a few ways of doing this
            # bad prediction, reduce radius to fraction of candidate distance
            # right now this is Rodriguez (1998), if eta_0 and eta_1 are the same
            if not accept:
                # trust_radius = gamma_1*trust_radius
                trust_radius = gamma_1*self.sdist_sc
            # remaining conditions apply for accepted steps
            elif eta_k_act < eta_1:
                trust_radius = gamma_1*trust_radius
                if self.options["print"]:
                    print_mpi(f"O       eta = {eta_k_act} < {eta_1}, Reducing")
            elif eta_k_act > eta_2:
                # check if trust constraint is active AKA we are on the boundary
                if self.trust_opt == 1:
                    trustconval = self.prob_model.get_val('trust.c_trust') 
                else:
                    # find out if we're on a bound or not
                    w1 = np.min(np.abs(zca_arr - llmt))
                    w2 = np.min(np.abs(zca_arr - ulmt))
                    trustconval = min(w1, w2)
                if trustconval < 1e-6: #NOTE: should change how we do this
                    trust_radius = gamma_2*trust_radius

                if self.options["print"]:
                    print_mpi(f"O       eta = {eta_k_act} > {eta_2}, Increasing")
            else:
                if self.options["print"]:
                    print_mpi(f"O       eta = {eta_k_act} < {eta_2}, > {eta_1}, Not Changing")

            if self.options["print"]:
                print_mpi(f"O       New Radius = {trust_radius}")

            if self.trust_opt == 1:
                self.prob_model.model.trust.set_radius(trust_radius)






            # define the LHS, compute gtru depending on how we do this
            if not self.options["model_grad_err_est"]: # get the actual gradient error
                if con_test:
                    self.gtru, self.gerr, self.gfea, dummy = grad_opt_feas(self.prob_model, self.have_cons, self.ctol)
                else:
                    self.gtru = self.prob_model.compute_totals(return_format='array')
                    self.gerr = np.linalg.norm(self.gtru)

                lhs0 = np.linalg.norm(gmod-self.gtru)
                rhs0 = min(gerrm, self.sdist)
                if self.options["tol_ignore_sdist"]:
                     rhs0 = gerrm
                xi_calc = lhs0/rhs0
                # def compute_lhs():
            else:
                # self.gtru = gmod
                # self.gerr = gerrm 
                lhs0 = lhs
                rhs0 = rhs

            # ferr = abs(fmod-ftru)

            # try the condition from Kouri (2013)?
            # import pdb; pdb.set_trace(

            # also introduce a step tolerance, self.sdist < stol
            if k == 0:
                gerr0 += self.gerr
                self.grange = gerr0 - self.gtol

            fetext = str(ferr)
            getext = str(self.gerr)


            

            

            # if we used the "truth" function to check refinement, we need to do this step now
            if not self.options["model_grad_err_est"]:
                if self.options["print"]:
                    print_mpi(f"___________________________________________________________________________")
                    print_mpi(f"")
                    print_mpi(f"OOO     Step 5: Refining to Meet Inexact Gradient Conditions...")
            
            
                lhs, rhs = self.model_refiner()
            
            
            
            # store important stuff for plotting
            self.outer_iter = k
            self.gfea = gfeam
            self.grad_lhs.append(lhs)
            self.grad_rhs.append(rhs)
            self.radii.append(trust_radius)
            self.realizations.append(self.prob_model.model.stat.get_fidelity())
            self.areds.append(actual_reduction)
            self.preds.append(predicted_reduction)
            # self.loc.append( om_dict_to_flat_array(zk, dvsettings, dvsize))
            self.loc.append(zk)
            if self.prob_model.model.stat.surrogate is not None:
                self.models.append(self.prob_model.model.stat.surrogate)
            print_mpi(f"O       Radius: {trust_radius}")
            print_mpi(f"O       LHS: {lhs}")
            print_mpi(f"O       RHS: {rhs}")
            print_mpi(f"O       xi*RHS - LHS = {rhs*self.xi - lhs}")
            
            
            # save iteration info to disk
            self._save_iteration(k)

            k += 1            

            #If g truth metrics are not met, 
            if self.gerr < self.gtol and gfeam < self.ctol and not self.options["inexact_gradient_only"]:
                fail = 0
                succ = 1
                break

            # Alternatively, use the inexact gradient condition
            if lhs0 < self.xi*rhs0 and gerrm < self.gtol and gfeam < self.ctol:
                fail = 0
                succ = 2
                break

            # If the predicted step size is small enough
            if self.sdist_sc < stol and gfeam < self.ctol and not self.options["tol_ignore_sdist"]:
                fail = 0
                succ = 3
                break

            # if we reach the reflevel limit, exit the optimization here and continue with full accuracy
            #NOTE: MAY NEED TO BE CAREFUL WITH RESPECT TO FEASABILITY
            if self.options["model_grad_err_est"] and self.reflevel[-1] > self.options["truth_func_err_est_max"]:
                fail = 3
                succ = 0
                break


        if k >= miter:
            succ = 0
            fail = 3

        if fail:
            failure_messages = (
                f'unsuccessfully, true gradient norm above tolerance: {getext}',
                f'unsuccessfully, trust region predicts increase',
                f'unsuccessfully, maximum refinement level reached',
                f'unsuccessfully, maximum outer iterations reached'
            )
            message = failure_messages[fail-1]

        else:
            success_messages = (
                f'successfully!',
                f'successfully, true gradient norm below tolerance: {getext}',
                f'successfully, inexact gradient condition met and model gradient norm below tolerance: {gerrm}',
                f'successfully, step size below tolerance: {self.sdist}'
            )
            message = success_messages[succ]

        zk = self.prob_model.driver.get_design_var_values()
        self.result_cur = om_dict_to_flat_array(zk, dvsettings, dvsize)

        self.grad_lhs.append(lhs)
        self.grad_rhs.append(rhs)
        self.radii.append(trust_radius)
        self.realizations.append(self.prob_model.model.stat.get_fidelity())
        self.areds.append(actual_reduction)
        self.preds.append(predicted_reduction)
        # self.loc.append(om_dict_to_flat_array(zk, dvsettings, dvsize))
        self.loc.append(zk)
        if self.prob_model.model.stat.surrogate is not None:
            self.models.append(self.prob_model.model.stat.surrogate)

        print_mpi("\n")
        print_mpi(f"Optimization terminated {message}")
        print_mpi(f"-------------------")
        print_mpi(f"    Outer Iterations: {self.outer_iter}")
        # Add constraint loop as well
        print_mpi(f"    -")
        print_mpi(f"    Final design vars: {zk}")
        print_mpi(f"    Final robust quant: {ftru}")
        print_mpi(f"    Final truth gradient norm: {getext}")
        print_mpi(f"    Final model gradient norm: {self.gerr}")
        print_mpi(f"    Final model error: {fetext}")
        print_mpi(f"    Final model level: {self.reflevel[-1]}")

        # print(f"    Total model samples: {self.model_iters}")
        # print(f"    Total truth samples: {self.truth_iters}")
        # print(f"    Total samples: {self.model_iters + self.truth_iters}")

        return succ, fail
            

    def model_refiner(self, lhs0, rhs0):
        """
        Refine the model based on options

        """

        # only use if we have a truth model
        rcap = self.prob_truth.model.stat.get_fidelity()
        if(self.options["ref_strategy"] == 1) and not self.options["model_grad_err_est"]:
            grel = self.gerr-self.gtol
            gclose = 1. - grel/self.grange
            rmin = self.options["flat_refinement"] #minimum improvement

            fac = gclose - self.reflevel[-1]/rcap

            refjump = rmin + max(0, int(fac*rcap))
            self.reflevel.append(refjump)
            if self.options["print"]:
                print_mpi(f"O       Strategy = Proximity to Convergence")
                print_mpi(f"O       Relative Proximity = {gclose}| Minimum Refinement = {rmin}")
                print_mpi(f"O       Current Level = {self.reflevel[-1]}/{rcap}| Jump = {refjump}| New Level = {reflevel+refjump}/{rcap}")
            # 


        # Ensure that the inexact gradient condition is met by adding enough points
        # \|\Nabla m_k(z_k) - \Nabla J(z_k)\| \leq \xi min(\|\Nabla m_k(z_k)\|, self.sdist)
        # or lhs = \xi rhs
        # \xi = 0.01 default
        # Question remains on how to ensure this with enough points
        # Report this for each iteration
        lhs = lhs0
        rhs = rhs0
        if(self.options["ref_strategy"] == 2):
            # for now, fully calculate the new lhs and rhs
            refjump = 0
            rmin = self.options["flat_refinement"] #minimum improvement
            rcap = self.prob_truth.model.stat.get_fidelity()
            # rcap = self.options["ref_cap"]
            if self.options["print"]:
                print_mpi(f"O       Strategy = Satisfy Inexact Gradient Condition")
                print_mpi(f"O       Minimum Refinement = {rmin}| xi = {self.xi}")
            rk = 0
            if lhs == None:
                lhs = 1.e6
            reflog = None
            # while(lhs > self.xi*rhs and refjump < rcap): # self.reflevel[-1][-1]
            if 1: # self.reflevel[-1][-1]
                rk += 1
                refjump = rmin
                estat = 'flat'
                      # 
                if self.options["model_grad_err_est"]:
                    estat = 'adaptive'
                    refjump_max = max(rmin, self.options["truth_func_err_est_max"])

                    # the right hand side is now the model gradient as it updates
                    # this doubles up as a verification step and a guidance for refinement
                    
                    self.cur_tol = -1.0
                    self.stop_update = False
                    self.duals = None # start with duals uninitialized
                    def xirhs(nmodel):
                        if not self.stop_update: # if the gradient isn't changing much, there's no point to constantly recomputing it
                            self.prob_model.model.stat.surrogate = nmodel
                            
                            if con_test:
                                #NOTE: need to recompute totals? TODO only do this for the robust surrogate part
                                dummy = self.prob_model.compute_totals(of="stat.musigma", wrt="x_d", return_format='array')
                                gmod, gerrm, gfeam, self.duals = grad_opt_feas(self.prob_model, self.have_cons, self.ctol, duals_given=self.duals, no_trust=True)
                            else:
                                gmod = self.prob_model.compute_totals(return_format='array')
                                gerrm = np.linalg.norm(gmod)

                            new_tol = self.xi*min(self.sdist, gerrm)
                            if self.options["tol_ignore_sdist"]:
                                new_tol = self.xi*gerrm
                            if abs(new_tol - self.cur_tol) < 1e-6 and not self.no_stall:
                                self.stop_update = True
                            # if constrained, we need to divide this by the appropriate dual
                            # if the constraint isnt active, multiply by a large number

                            # if the robust quantity is a constraint
                            if 'stat.musigma' in self.prob_model.driver._cons:
                                # import pdb; pdb.set_trace()

                                # determine factor based on lagrange multiplier value
                                fac = 1.
                                #NOTE: if we start with active constraints, but they turn off, 
                                # because we keep duals constant, this doesn't go away
                                # i think we want to keep this behavior, otherwise refinement quits
                                # as soon as we're feasible again
                                if 'stat.musigma' in self.duals:
                                    # if new_tol > 1e-15:
                                    #     import pdb; pdb.set_trace()
                                    fac_div = max(self.options['min_dual'], self.duals['stat.musigma'])
                                else:
                                    fac_div = self.options['min_dual']
                                
                            self.cur_tol = new_tol
                        return self.cur_tol*(fac/fac_div)

                    # refjump = self.prob_model.model.stat.refine_model(refjump_max, self.xi*rhs)
                    refjump, reflog = self.prob_model.model.stat.refine_model(refjump_max, xirhs, f'{self.path}/{self.title}/ref_progress.pickle')
                    # gerrm = reflog[-1,1]/self.xi
                    gerrm = self.cur_tol
                    # import pdb; pdb.set_trace()

                else:
                    refjump, reflog = self.prob_model.model.stat.refine_model(refjump)
                    self.prob_model.run_model()
                    if con_test:
                        gmod, gerrm, gfeam = grad_opt_feas(self.prob_model, self.have_cons, self.ctol)
                    else:
                        gmod = self.prob_model.compute_totals(return_format='array')
                        gerrm = np.linalg.norm(gmod)
                    
                self.reflevel.append(reflog[-1,2])
                
                if not self.options["model_grad_err_est"]:
                    lhs = np.linalg.norm(gmod-self.gtru)
                else:
                    # lhs = self.prob_model.model.stat.sampler.rcrit.get_energy(self.prob_model.model.stat.sampler.xlimits)
                    lhs = reflog[-1,0]
                rhs = min(gerrm, self.sdist)
                if self.options["tol_ignore_sdist"]:
                    rhs = gerrm

                if self.options["print"]:
                    print_mpi(f"o       Inexact Iter: {rk} | Energy = {estat}| Levels = {self.reflevel}| LHS = {lhs}| RHS: {rhs} | xi*RHS - LHS = {rhs*self.xi - lhs}")
            if self.options["print"]:
                if lhs < self.xi*rhs:
                    print_mpi(f"o       Inexact Gradient Condition Satisfied in {rk} Iterations with {refjump} Points")
                else:
                    print_mpi(f"o       Max Iterations Reached before Satisfaction, Continue with {refjump} Points")

        else:

            #NOTE: Replacing this with earlier refinement, see Step 2.
            # # grab sample data from the truth model if we are using a surrogate
            # if self.prob_model.model.stat.surrogate and self.options["use_truth_to_train"]:
            #     truth_eval = self.prob_truth.model.stat.sampler.current_samples
            #     refjump = truth_eval['x'].shape[0]
            #     if self.options["print"]:
            #         print(f"O       Strategy = Refine with Validation Points")
            #         print(f"O       Level Remains Static| Points Added = {refjump}")
            #     self.prob_model.model.stat.refine_model(truth_eval)
            #     reflevel = self.prob_model.model.stat.xtrain_act.shape[0]
            # else:

            if self.options["print"]:
                print_mpi(f"O       Strategy = Flat Refinement")
                print_mpi(f"O       Current Level = {self.reflevel[-1]}/{rcap}| Jump = {refjump}| New Level = {self.reflevel+refjump}/{rcap}")
            refjump, reflog = self.prob_model.model.stat.refine_model(refjump)
            self.reflevel.append(refjump)

        self.reflog.append(reflog)
        self.gerr = gerrm
        # self.gtru = gmod
        return lhs, rhs
    


    # def lagrangian_eval(problem):



    #     return dL, LO, LF