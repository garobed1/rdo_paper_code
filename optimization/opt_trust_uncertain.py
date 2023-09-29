import numpy as np
import copy
import collections

from optimization.optimizers import optimize
from optimization.robust_objective import RobustSampler
from optimization.opt_subproblem import OptSubproblem
from utils.om_utils import get_om_design_size, om_dict_to_flat_array
from optimization.trust_bound_comp import TrustBound
from collections import OrderedDict

from smt.utils.options_dictionary import OptionsDictionary

import openmdao.api as om
from openmdao.utils.mpi import MPI




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

    def _declare_options(self):
        
        super()._declare_options()
        
        declare = self.options.declare
        



        declare(
            "initial_trust_radius", 
            default=1.0, 
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
            # default=0.01, 
            default=1.00, 
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
            "truth_func_err_est_max", 
            default=500, 
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
        miter = self.options['max_iter']
        max_trust_radius = self.options['max_trust_radius']
        initial_trust_radius = self.options['initial_trust_radius']
        eta_0 = self.options['eta']
        eta_1 = self.options['eta_1']
        eta_2 = self.options['eta_2']
        gamma_1 = self.options['gamma_1']
        gamma_2 = self.options['gamma_2']
        self.xi = self.options["xi"]
        

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


        self.grad_lhs = []
        self.grad_rhs = []        

        # initial guess in dict format, whatever the current state of the OM system is
        z0 = self.prob_model.driver.get_design_var_values()
        # DICT TO ARRAY (OR NOT)

        # design variable meta settings
        dvsettings = self.prob_model.model.get_design_vars()
        dvsize = get_om_design_size(dvsettings)

        # flagged for failed optimization
        fail = 0

        # optimization index
        k = 0

        # region movement index (only incremented if we accept a step)
        tsteps = 0

        # TODO: make parallel, starting with printing only on rank 0
        if self.options["print"]:
            fetext = '-'
            getext = '-'

        # store error measures, including the first one
        gerr0 = 0
        ferr = 1e6
        self.gerr = 1e6

        # get DV bounds
        dvbl = {}
        dvbu = {}
        for name in dvsettings:
            dvbl[name] = copy.deepcopy(dvsettings[name]['lower'])
            dvbu[name] = copy.deepcopy(dvsettings[name]['upper'])

        # we assume that fidelity belongs to the top level system
        # calling it stat for now
        self.reflevel = self.prob_model.model.stat.get_fidelity()
        refjump = self.options["flat_refinement"]
        #TODO: Need constraint conditions as well

        # initialize trust radius
        trust_radius = initial_trust_radius
        zk = z0

        # validate the first point before starting
        if not self.options["model_grad_err_est"]:
            self._eval_truth(zk)
            ftru = copy.deepcopy(self.prob_truth.get_val(self.prob_outs[0]))
            #TODO: this needs to be the lagrangian gradient with constraints
            self.gtru = self.prob_truth.compute_totals(return_format='array')
            self.gerr = np.linalg.norm(self.gtru)
            gerr0 += self.gerr
            self.grange = gerr0 - self.gtol
            if self.options["print"]:
                getext = str(self.gerr)

        print(f"___________________________________________________________________________")
        print(f"Optimization Parameters")
        print(f"Gradient Tolerance         = {self.gtol}")
        print(f"Step Tolerance             = {stol}")
        print(f"Maximum Outer Iterations   = {miter}")
        print(f"Initial Trust Radius       = {initial_trust_radius}")
        print(f"Number of Design Variables = {dvsize}")
        print(f"Number of UQ Variables     = {self.prob_model.model.stat.sampler.x_u_dim}")


        print(f"___________________________________________________________________________")
        print(f"Beginning Full Optimization-Under-Uncertainty Loop")

        while (self.gerr > self.gtol) and (k < miter):
            fail = 100
            if self.options["print"]:
                print("\n")
                print(f"Outer Iteration {k} ")
                print(f"Trust Region {tsteps} ")
                print(f"-------------------")
                print(f"    OBJ ERR: {fetext}")
                # Add constraint loop as well
                print(f"    -")
                print(f"    GRD ERR: {getext}")
                print(f"    Fidelity: {self.reflevel}")
                
                self.prob_model.list_problem_vars()
            
                print(f"___________________________________________________________________________")
                print(f"")
                print(f"OOO     Step 1: Solving subproblem...")
            # =================================================================
            # 1.
            #   find a solution to the subproblem
            # =================================================================

            # 
            zk_cent = copy.deepcopy(zk)
            # import pdb; pdb.set_trace()
            #TODO: SCALED TRUST REGION
            if self.trust_opt == 1:
                self.prob_model.model.trust.set_center(zk_cent)
            else: #BOX, self.trust_opt == 2
                for name in dvsettings:
                    llmt = np.maximum(zk_cent[name] - trust_radius, dvbl[name])
                    ulmt = np.minimum(zk_cent[name] + trust_radius, dvbu[name])
                    #TODO: Annoying absolute name path stuff
                    self.prob_model.model.set_design_var_options(name.split('.')[-1], lower=llmt, upper=ulmt)


            ### THE SUBPROBLEM SOLVE    
            # import pdb; pdb.set_trace()
            # self.prob_model.model.stat.check_partials_flag = True
            # self.prob_model.check_partials()
            # self.prob_model.model.stat.check_partials_flag = False
            # import pdb; pdb.set_trace()
            fmod_cent = copy.deepcopy(self.prob_model.get_val(self.prob_outs[0]))
            self._solve_subproblem(zk)  
            fmod_cand = copy.deepcopy(self.prob_model.get_val(self.prob_outs[0]))
            zk_cand = self.prob_model.driver.get_design_var_values()
            # self.prob_model.model.stat.check_partials_flag = True
            # self.prob_model.check_partials()
            # self.prob_model.model.stat.check_partials_flag = False
            # import pdb; pdb.set_trace()

            # compute predicted reduction
            predicted_reduction = fmod_cent - fmod_cand #kouri method
            # predicted_reduction = ftru_cent - fmod_cand #TODO: how to handle this?
            self.pred = predicted_reduction
            
            # retrieve gradients information
            # import pdb; pdb.set_trace()
            # this needs to be the lagrangian gradient with constraints
            gmod = self.prob_model.compute_totals(return_format='array')
            gerrm = np.linalg.norm(gmod)
        
            # need distance of prediction, and if its on edge of radius
            # the dv arrays originate from OrderedDict objects, so this should be fine
            zce_arr = om_dict_to_flat_array(zk_cent, dvsettings, dvsize)
            zca_arr = om_dict_to_flat_array(zk_cand, dvsettings, dvsize)
            s = zca_arr - zce_arr
            self.sdist = np.linalg.norm(s)
            if self.options["print"]:
                print(f"o       Step Size = {self.sdist}, Step Tol = {stol}")
            
            if self.options["model_grad_err_est"]:
                # Compare to refined model
                if self.options["print"]:
                    print(f"___________________________________________________________________________")
                    print(f"")
                    print(f"OOO     Step 2: Validating by Refining Model...")
                
                # Estimate the inexact gradient condition now rather than later
                # lhs0 = lhs = self.prob_model.model.stat.sampler.rcrit.get_energy()#np.linalg.norm(gmod-gtru)
                lhs0 = None
                rhs0 = min(gerrm, self.sdist)
                # xi_calc = lhs0/rhs0

                lhs, rhs = self.model_refiner(lhs0, rhs0)

                # recompute model
                self.prob_model.run_model()
                fmod_cand_star = copy.deepcopy(self.prob_model.get_val(self.prob_outs[0]))

                actual_reduction = fmod_cent - fmod_cand_star
            
            
            else:
                #Eval Truth
                if self.options["print"]:
                    print(f"___________________________________________________________________________")
                    print(f"")
                    print(f"OOO     Step 2: Validating by Computing Truth...")

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
                print(f"___________________________________________________________________________")
                print(f"")
                print(f"OOO     Step 3: Validating Subproblem Reduction with Truth...")
            
            
            # If the predicted step size is small enough
            if self.sdist < stol:
                fail = 0
                succ = 3
                break

            if predicted_reduction <= 0 and self.options['trust_increase_terminate']:
                fail = 2
                break
            # if predicted_reduction <= 0:
            eta_k = actual_reduction/predicted_reduction

            # choose to accept or not
            accept = False
            if eta_k > eta_0:
                tsteps += 1
                zk = zk_cand
                accept = True
            else:
                zk = zk_cent

            #NOTE: this may be classic, as it is now, or retroactive, based on
            # updated/refined model, need to make this an option
            eta_k_act = eta_k




            # =================================================================
            # 4.
            #   adjust trust radius
            # =================================================================
            
            if self.options["print"]:
                print(f"___________________________________________________________________________")
                print(f"")
                print(f"OOO     Step 4: Modifying Trust Bounds...")
                print(f"O       Prior Radius = {trust_radius}")
            
            # a few ways of doing this
            # bad prediction, reduce radius to fraction of candidate distance
            # right now this is Rodriguez (1998), if eta_0 and eta_1 are the same
            if not accept:
                # trust_radius = gamma_1*trust_radius
                trust_radius = gamma_1*self.sdist
            # remaining conditions apply for accepted steps
            elif eta_k_act < eta_1:
                trust_radius = gamma_1*trust_radius
                if self.options["print"]:
                    print(f"O       eta = {eta_k_act} < {eta_1}, Reducing")
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
                    print(f"O       eta = {eta_k_act} > {eta_2}, Increasing")
            else:
                if self.options["print"]:
                    print(f"O       eta = {eta_k_act} < {eta_2}, > {eta_1}, Not Changing")

            if self.options["print"]:
                print(f"O       New Radius = {trust_radius}")

            if self.trust_opt == 1:
                self.prob_model.model.trust.set_radius(trust_radius)






            # define the LHS, compute gtru depending on how we do this
            if not self.options["model_grad_err_est"]: # get the actual gradient error
                self.gtru = self.prob_truth.compute_totals(return_format='array')
                self.gerr = np.linalg.norm(self.gtru)

                lhs0 = np.linalg.norm(gmod-self.gtru)
                rhs0 = min(gerrm, self.sdist)
                xi_calc = lhs0/rhs0
                # def compute_lhs():
            else:
                self.gtru = gmod
                self.gerr = gerrm 
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


            #If g truth metrics are not met, 
            if self.gerr < self.gtol and not self.options["inexact_gradient_only"]:
                fail = 0
                succ = 1
                break

            # Alternatively, use the inexact gradient condition
            if lhs0 < self.xi*rhs0 and gerrm < self.gtol:
                fail = 0
                succ = 2
                break
            import pdb; pdb.set_trace()

            # if we used the "truth" function to check refinement, we need to do this step now
            if not self.options["model_grad_err_est"]:
                if self.options["print"]:
                    print(f"___________________________________________________________________________")
                    print(f"")
                    print(f"OOO     Step 5: Refining to Meet Inexact Gradient Conditions...")
            
            
                lhs, rhs = self.model_refiner()

            k += 1            
            self.outer_iter = k
            self.grad_lhs.append(lhs)
            self.grad_rhs.append(rhs)
            print(f"O       LHS: {lhs}")
            print(f"O       RHS: {rhs}")
            print(f"O       xi*RHS - LHS = {rhs*self.xi - lhs}")
            # import pdb; pdb.set_trace()

        if fail:
            failure_messages = (
                f'unsuccessfully, true gradient norm above tolerance: {getext}',
                f'unsuccessfully, trust region predicts increase'
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
        self.result_cur = zk

        print("\n")
        print(f"Optimization terminated {message}")
        print(f"-------------------")
        print(f"    Outer Iterations: {self.outer_iter}")
        # Add constraint loop as well
        print(f"    -")
        print(f"    Final design vars: {zk}")
        print(f"    Final objective: {ftru}")
        print(f"    Final truth gradient norm: {getext}")
        print(f"    Final model gradient norm: {gerrm}")
        print(f"    Final model error: {fetext}")
        print(f"    Final model level: {self.reflevel}")

        print(f"    Total model samples: {self.model_iters}")
        print(f"    Total truth samples: {self.truth_iters}")
        print(f"    Total samples: {self.model_iters + self.truth_iters}")
            


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

            fac = gclose - self.reflevel/rcap

            refjump = rmin + max(0, int(fac*rcap))
            if self.options["print"]:
                print(f"O       Strategy = Proximity to Convergence")
                print(f"O       Relative Proximity = {gclose}| Minimum Refinement = {rmin}")
                print(f"O       Current Level = {self.reflevel}/{rcap}| Jump = {refjump}| New Level = {reflevel+refjump}/{rcap}")
            # import pdb; pdb.set_trace()


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

            rmin = self.options["flat_refinement"] #minimum improvement
            rcap = self.prob_truth.model.stat.get_fidelity()
            # rcap = self.options["ref_cap"]
            if self.options["print"]:
                print(f"O       Strategy = Satisfy Inexact Gradient Condition")
                print(f"O       Minimum Refinement = {rmin}| xi = {self.xi}")
            rk = 0
            if lhs == None:
                lhs = 1.e6
            while(lhs > self.xi*rhs and self.reflevel < rcap):
                rk += 1
                refjump = rmin
                estat = 'flat'
                      # import pdb; pdb.set_trace()
                if self.options["model_grad_err_est"]:
                    estat = 'adaptive'
                    refjump_max = max(rmin, self.options["truth_func_err_est_max"])
                    refjump = self.prob_model.model.stat.refine_model(refjump_max, self.xi*rhs)
                else:
                    refjump = self.prob_model.model.stat.refine_model(refjump)
                    
                self.reflevel += refjump
                self.prob_model.run_model()
                
                import pdb; pdb.set_trace()

                gmod = self.prob_model.compute_totals(return_format='array')
                gerrm = np.linalg.norm(gmod)
                if not self.options["model_grad_err_est"]:
                    lhs = np.linalg.norm(gmod-self.gtru)
                else:
                    lhs = self.prob_model.model.stat.sampler.rcrit.get_energy(self.prob_model.model.stat.sampler.xlimits)
                rhs = min(gerrm, self.sdist)

                if self.options["print"]:
                    print(f"o       Inexact Iter: {rk} | Energy = {estat}| New Level = {self.reflevel}| LHS = {lhs}| RHS: {rhs}| xi*RHS - LHS = {rhs*self.xi - lhs}")
            if self.options["print"]:
                if lhs < self.xi*rhs:
                    print(f"o       Inexact Gradient Condition Satisfied in {rk} Iterations with {self.reflevel} Points")
                else:
                    print(f"o       Max Iterations Reached before Satisfaction, Continue with {self.reflevel} Points")

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
                print(f"O       Strategy = Flat Refinement")
                print(f"O       Current Level = {self.reflevel}/{rcap}| Jump = {refjump}| New Level = {self.reflevel+refjump}/{rcap}")
            self.prob_model.model.stat.refine_model(refjump)
            self.reflevel += refjump

        return lhs, rhs