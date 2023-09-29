import numpy as np
import copy
from collections import defaultdict
from utils.error import stat_comp, _gen_var_lists
from scipy.special import legendre, hermite, jacobi, roots_legendre, roots_hermite, roots_jacobi
from smt.sampling_methods import LHS
from infill.getxnew import adaptivesampling
from infill.refinement_picker import GetCriteria
from smt.utils.options_dictionary import OptionsDictionary
from utils.sutils import convert_to_smt_grads

# this one better suited for surrogate model


_poly_root_types = {
    "uniform": roots_legendre,
    "norm": roots_hermite,
    "beta": roots_jacobi
}

"""
Provides an interface to a consistent set of sampling points for a function with
design inputs and uncertain inputs, using LHS
"""
class RobustSampler():
    def __init__(self, x_d_init, N, **kwargs):
        self.has_points = False #are uncertain samples generated at the current design?

        self.x_d_cur = x_d_init # current design point, same length as x_d_ind
        self.x_d_ind = None # indices of the design variables in the function call
        self.x_u_ind = None 

        self.x_d_dim = 0
        self.x_u_dim = 0


        # self.x_samples = None
        # self.f_samples = None
        # self.g_samples = None
        self.current_samples = defaultdict(dict)
        self.history = {}
        self.design_history = []
        self.nested_ref_ind = None #list of indices of current iteration that existed in previous
        self.func_computed = False #do we have function data at the samples?
        self.grad_computed = False #do we have gradient data at the samples?
        self.stop_generating = True
        self.dont_reset = False #for adaptive sampler, don't reset attributes if we've computed them through sampling
        self.func = None #set function for adaptive refinement

        self._attribute_reset()

        self.sampling = None #SMT sampler object

        """
        Max index to keep track of previous iterations. Increments either when 
        design is changed, or UQ is refined/changed
        """
        self.iter_max = -1

        self.comp_track = 0 #track total number of evaluations since instantiation
        self.grad_track = 0

        self.options = OptionsDictionary()
        self.options.declare(
            "xlimits",
            types=np.ndarray,
            desc="The interval of the domain in each dimension with shape dim x 2 (required)",
        )

        self.options.declare(
            "name",
            default='',
            types=str,
            desc="The interval of the domain in each dimension with shape dim x 2 (required)",
        )
        
        self.options.declare(
            "probability_functions",
            types=list,
            desc="gives pdfs of uncertain variables, implicit list of design variables",
        )

        self.options.declare(
            "retain_uncertain_points",
            types=bool,
            default=True,
            desc="keep the same points in the uncertain space as we traverse the design space",
        )


        self.options.declare(
            "name",
            types=str,
            default='sampler',
            desc="keep the same points in the uncertain space as we traverse the design space",
        )

        self.options.declare(
            "external_only",
            types=bool,
            default=False,
            desc="only use with surrogate. when design is updated, don't add new points at all",
        )   #TODO: This will likely need tweaking, and allow for both kinds of training data updates


        self._declare_options()
        self.options.update(kwargs)

        self.N = N

        self.initialize()

    def initialize(self):
        # run sampler for the first time on creation
        pdfs = self.options["probability_functions"]
        xlimits = self.options["xlimits"]
        

        pdf_list, uncert_list, static_list, scales, pdf_name_list = _gen_var_lists(pdfs, xlimits)
        
        self.x_u_dim = len(uncert_list)
        self.x_d_dim = len(static_list)
        self.x_u_ind = uncert_list
        self.x_d_ind = static_list
        self.pdf_list = pdf_list
        self.pdf_name = pdf_name_list
        self.dim = self.x_u_dim + self.x_d_dim
        self.scales = scales

        self._initialize()
        self.generate_uncertain_points(self.N)    


    """
    Start of methods to override
    """
    def _initialize(self):
        
        

        
        # run sampler for the first time on creation
        xlimits = self.options["xlimits"]

        u_xlimits = xlimits[self.x_u_ind]
        self.sampling = LHS(xlimits=u_xlimits, criterion='maximin')

    def _declare_options(self):
        # add exclusive options
        
        self.options.declare(
            "design_noise",
            types=float,
            default=0.0,
            desc="only use with surrogate. when sampling the uncertain space, add random noise to the dvs, scaled by this option",
        )

    def _new_sample(self, N):
        #TODO: options for this
        u_tx = self.sampling(N)
        tx = np.zeros([N, self.dim])
        tx[:, self.x_u_ind] = u_tx
        tx[:, self.x_d_ind] = self.x_d_cur#[self.x_d_cur[i] for i in self.x_d_ind]


        return tx

    def _refine_sample(self, N, e_tol=-1.):
        tx = self.current_samples['x']
        noise = self.options["design_noise"]

        # just produce new LHS
        newsize = N + tx.shape[0]
        u_tx = self.sampling(newsize)
        tx = np.zeros([newsize, self.dim])
        tx[:, self.x_u_ind] = u_tx
        tx[:, self.x_d_ind] = self.x_d_cur
        # track matching points #TODO: standardize this
        # self.nested_ref_ind = range(tx.shape[0]).tolist()
        
        """
        function this out
        
        if noise > 1e-12:
            perturb = np.random.rand(self.x_d_dim)
            xlimits = self.options["xlimits"]
            d_xlimits = xlimits[self.x_d_ind]
            bound = np.zeros([self.x_d_dim, 2])
            # for i in range(self.x_d_dim):
            
            
            

            scale = d_xlimits[:,1] - d_xlimits[:,0]
            nlower = self.x_d_cur[:] - noise*scale
            bound[:,0] = np.maximum(d_xlimits[:][0], nlower)
            nupper = self.x_d_cur[:] + noise*scale
            bound[:,1] = np.minimum(d_xlimits[:][1], nupper)
            bscale = bound[:,1] - bound[:,0]
            bperturb = perturb*bscale - bound[:,0]

            tx[:, self.x_d_ind] += bperturb 
        """
            



        return tx

    """
    End of methods to override
    """

    def set_design(self, x_d_new):
        
        x_d_buf = x_d_new
        ret = 0
        print(f"o       {self.options['name']} Iteration {self.iter_max}: Design {x_d_buf}", end='')
        if np.allclose(x_d_buf, self.x_d_cur, rtol = 1e-15, atol = 1e-15):
            print(f": No change in design, no data added")
            return ret # indicates that we have not moved, useful for gradient evals, avoiding retraining
        # import pdb; pdb.set_trace()
        if not self.options["external_only"]:
            self.has_points = False
            if self.options["retain_uncertain_points"]:
                tx = copy.deepcopy(self.current_samples['x'])
                self._internal_save_state()
                self.current_samples['x'] = tx
                self.current_samples['x'][:, self.x_d_ind] = x_d_buf
                # self.x_samples[:, self.x_d_ind] = self.x_d_cur#[self.x_d_cur[i] for i in self.x_d_ind]
                self.has_points = True
                print(f": Design is changed, retaining data from previous locations")
            else:
                print(f": Design is changed, ready to add new data")
            ret = 1
        else:
            print(f": Design is changed, but no data added")
            ret = 0
        self.x_d_cur = x_d_buf
        return ret # indicates that we have not moved, useful for gradient evals, avoiding retraining


    #NOTE: both generate_ and refine_ need an option to introduce noise to the sample
    def generate_uncertain_points(self, N, func=None, model=None):
        """
        First of two functions that will increment the sampling iteration

        Generates 

        Parameters
        ----------
        N: int or float or list
            generic reference to sampling level, by default just number of points to sample

        """
        # check if we already have them
        if self.has_points:
            print(f"o       {self.options['name']} Iteration {self.iter_max}: Already have points, no points generated")
            return 0

        print(f"o       {self.options['name']} Iteration {self.iter_max}: Generating {N} new points for UQ evaluation")
        self.func = func
        self.model = model
        tx = self._new_sample(N)

        # archive previous dataset
        self._internal_save_state(refine=False)

        self.current_samples['x'] = tx
        self.has_points = True

        return 1

    def refine_uncertain_points(self, N, tol=-1., func=None, model=None):
        """
        Second of two functions that will increment the sampling iteration
        Add more UQ points to the current design. Usually this is nested

        N: int or float or list
            generic reference to sampling level, by default just number of points to add to sample
        func: SMT Function
            If refinery requires function evaluations (e.g. adaptive sampling), compute from this
        """
        # check if we already have them NOTE: EVEN IF A DIFFERENT NUMBER IS REQUESTED FOR NOW
        if self.has_points and N is None:
            print(f"{self.options['name']} Iter {self.iter_max}: No refinement requested, no points generated")
            return 0

        told = self.current_samples['x'].shape[0]

        print(f"o       {self.options['name']} Iteration {self.iter_max}: Refining {N} new points for UQ evaluation")
        self.func = func
        self.model = model
        tx = self._refine_sample(N, e_tol=tol)

        # archive previous dataset
        self._internal_save_state(refine=True)
        
        self.current_samples['x'] = tx
        self.has_points = True

        tdiff = tx.shape[0] - told

        return tdiff

    def set_evaluated_func(self, f):

        self.current_samples['f'] = f
        self.comp_track += f.shape[0]
        self.func_computed = True

    def set_evaluated_grad(self, g):

        self.current_samples['g'] = g
        self.grad_track += g.shape[0]
        self.grad_computed = True

    # resets attributes as space is traversed
    def _attribute_reset(self):

        self.current_samples['x'] = None
        self.current_samples['f'] = None
        self.current_samples['g'] = None
        self.func_computed = False
        self.grad_computed = False
        self.has_points = False


    # saving to a dict of dicts
    def _internal_save_state(self, refine=False, insert=False):
        """
        Internal version, increments sample counter since it's called just
        before updating the state

        Parameters:
        -----------
        refine: bool
            True if refining in place, false if traversing

        insert: bool
            True if points are added external to sampler generation
        """
        if self.iter_max < 0:
            self.iter_max += 1
            return

        affix = '_mov'
        if refine:
            affix = '_ref'
        if insert:
            affix = '_ins'

        name = self.options['name'] + '_' + str(self.iter_max) + affix
        self.save_state(name)

        # update design history, simple list
        self.design_history.append(self.x_d_cur)

        # increment iteration counter, reset if not using adaptive sampling
        if not self.dont_reset:
            self._attribute_reset()
        self.dont_reset = False
        self.iter_max += 1

    # add samples to this object from outside, in the format of current_samples
    #
    def add_data(self, new_samples, replace_current=False):

        # check that new_samples is good
        assert 'x' in new_samples

        if replace_current:

            # archive previous dataset
            self._internal_save_state(insert=True)

            self.current_samples['x'] = copy.deepcopy(new_samples['x'])
            self.has_points = True
            if 'f' in new_samples:
                self.set_evaluated_func(new_samples['f'])
            if 'g' in new_samples:
                self.set_evaluated_grad(new_samples['g'])

            self.stop_generating = False
        else:
            print(f"{self.options['name']} Iter {self.iter_max}: Adding points without replacing not implemented!")



    def save_state(self, name):
        """
        Save current state to a dict of dicts

        Parameters:
        -----------

        name: None or str
            custom name for this entry
        
        """

        #TODO: option to write to file, probably

        #This default naming scheme allows built in methods to access iterations
        if not name:
            name = self.options['name'] + '_' + str(self.iter_max) + '_saved'

        self.history[name] = copy.deepcopy(self.current_samples)
        self.history[name]['func_computed'] = copy.deepcopy(self.func_computed)
        self.history[name]['grad_computed'] = copy.deepcopy(self.grad_computed)


        # This tracks full points between refinement, so we know what we don't have to recompute
        self.history[name]['nested_ref_ind'] = copy.deepcopy(self.nested_ref_ind)

        # also keep track of if func/grad computed, other stuff?

    def load_state(self, name, filename=None):
        """
        Load state from name, set attributes as current

        Parameters:
        -----------

        name: None or str
            name of entry to load, if 

        """

        pass


"""
Sampler object for stochastic collocation points. N represents total polynomial order, though it may be adapted
on a per-direction basis
"""
class CollocationSampler(RobustSampler):
    


    def _initialize(self):
        # add exclusive options
        # self.options.declare(
        #     "external_only",
        #     types=bool,
        #     default=False,
        #     desc="only use with surrogate. when design is updated, don't add new points at all",
        # )   #TODO: This will likely need tweaking, and allow for both kinds of training data updates
        
        # run sampler for the first time on creation
        self.xlimits = self.options["xlimits"]

        

        # given pdfs, generate list of appropriate polynomials
        #TODO: Only works for uniform/legendre and normal/hermite, beta needs to pass args
        poly_list = []
        for i in range(self.x_u_dim):
            j = self.x_u_ind[i]
            pname = self.pdf_name[i]
            if pname == "beta":
                # import pdb; pdb.set_trace()
                poly_list.append(lambda n: _poly_root_types["beta"](n, 
                                                                    alpha=self.pdf_list[j].args[1]-1,
                                                                    beta=self.pdf_list[j].args[0]-1)) # need to subtract one?
                                                                    # alpha=self.pdf_list[j].args[1]-1,
                                                                    # beta=self.pdf_list[j].args[0]-1)) # need to subtract one and reverse?
                # print(poly_list[i](20))
                # from scipy.stats.distributions import beta
                # print(beta())
                # import pdb; pdb.set_trace()
            else: # don't require additional args
                poly_list.append(_poly_root_types[pname])

        self.poly_list = poly_list
        self.weights = None
        
        self.absc_nsc = None
        self.weig_ind = None

        self.N_act = None
        self.jumps = None
        # N represents the order of the polynomial basis functions, not the samples directly (N_act)
        # If int, that is the order for all directions. If list (of length x_u_dim), apply to each 



        # u_xlimits = xlimits[self.x_u_ind]
        # self.sampling = LHS(xlimits=u_xlimits, criterion='maximin')

    def _declare_options(self):
        pass

    def _new_sample(self, N):

        self.N = N
        xlimits = self.options["xlimits"]
        if isinstance(N, int):
            self.N = self.x_u_dim*[N]

        # use recursion to form full tensor products
        N_act = self._recurse_total_points(0, self.N, 1)
        self.N_act = N_act

        # use recursion to get jumps for each dimension
        jumps = np.zeros(self.x_u_dim, dtype=int)

        # gather list of all abscissae in each direction
        absc = []
        absc_nsc = [] #no scale
        weig = []
        #TODO: ONLY WORKS FOR DISTS RANGING -1 to 1, NORMAL DIST IS UNCLEAR
        for i in range(self.x_u_dim):
            x_nsc, w = self.poly_list[i](self.N[i])
            pname = self.pdf_name[i]
            if pname == 'beta':
                w *= 2./np.sum(w) #AREA CORRECTION
            # import pdb; pdb.set_trace()
            # x = x*(self.scales[i]/2) + (0.5*self.scales[i] + xlimits[i,0])
            absc_nsc.append(x_nsc)
            x = 0.5*(x_nsc + 1.)*self.scales[i] + xlimits[i,0]
            absc.append(x)
            weig.append(w)
            jumps[i] = self._recurse_total_points(i, self.N, 1)/self.N[i]

        self.u_tx = np.zeros([N_act, self.x_u_dim])
        self.weights = np.ones(N_act)
        
        si = np.zeros(self.x_u_dim, dtype=int)
        self._recurse_sc_formation(0, si, jumps, absc, weig, self.N)

        # N_act = len(u_tx)
        # u_tx = np.array(u_tx)
        # save abscissae as well
        self.jumps = jumps
        self.absc_nsc = absc_nsc
        self.weig_ind = weig
        self.weights /= np.power(2, self.x_u_dim)
        tx = np.zeros([N_act, self.dim])
        tx[:, self.x_u_ind] = self.u_tx
        tx[:, self.x_d_ind] = self.x_d_cur#[self.x_d_cur[i] for i in self.x_d_ind]

        # import pdb; pdb.set_trace()
        return tx

    def _refine_sample(self, N, e_tol=-1.):
        N_old = self.N

        if isinstance(N, int):
            N_add = self.x_u_dim*[N]
        else:
            N_add = N

        N_new = np.array(N_old) + np.array(N_add)
        N_new = N_new.tolist()

        tx = self._new_sample(N)

        return tx

    def _recurse_sc_formation(self, di, si, jumps, absc, weig, N):
        
        N_cur = N[di]
        
        for i in range(N_cur):
            super_ind = np.dot(si, jumps)
            # recurse if more dimensions
            if di < self.x_u_dim - 1:
                self._recurse_sc_formation(di+1, si, jumps, absc, weig, N)
                si[di+1:] = 0
            
            self.weights[super_ind] = 1.0
            for j in range(self.x_u_dim):
                try:
                    self.u_tx[super_ind][j] = absc[j][si[j]]
                    self.weights[super_ind] *= weig[j][si[j]]
                except:
                    print("SC Recursion Failure!")
                    import pdb; pdb.set_trace()

            si[di] += 1
            

    def _recurse_total_points(self, di, N, tot):
        
        tot *= N[di]
        # recurse if more dimensions
        if di < self.x_u_dim - 1:
            tot = self._recurse_total_points(di+1, N, tot)

        return tot




"""
Sampler object for adaptively sampled surrogate points. Only works for surrogate-based stat comp

Need a refinement criteria object, but we also need the surrogate used in stat comp
"""
class AdaptiveSampler(RobustSampler):
    


    def _initialize(self):
        
        # run sampler for the first time on creation
        self.xlimits = self.options["xlimits"]

        # establish criteria
        self.rset = self.options['criteria']
        self.rcrit = None # can only initialize this once we initialize the surrogate

        u_xlimits = self.xlimits[self.x_u_ind]
        # self.sampling = LHS(xlimits=self.xlimits, criterion='maximin')
        self.sampling = LHS(xlimits=u_xlimits, criterion='maximin')

        self.surrogate_initialized = False

    def _declare_options(self):
        # the criteria contains the model, we should tell stat comp to get its surrogate from this
        self.options.declare(
            "criteria",
            default=None,
            desc="REQUIRED: refinement criteria function for adaptive sampling/infill, given as the list of settings",
        )

        self.options.declare(
            "max_batch",
            default=1,
            types=int,
            desc="maximum allowable batch size",
        )

        self.options.declare(
            "full_refine",
            default=False,
            types=bool,
            desc="if true, operate refinement over combined design/uncertain spaces",
        )

        self.options.declare(
            "as_options",
            default=None,
            desc="options dict for adaptive sampling iterator",
        )   


    # NOTE: jk we're overriding for now, TODO need option to initialize over udim or full dim FOR THE BASE CLASS
    def _new_sample(self, N):


        # if model already exists, use _refine_sample instead
        if self.model is not None:
            u_tx = self._refine_sample(N)
        else:
            u_tx = self.sampling(N)
        
        
        tx = np.zeros([N, self.dim])
        tx[:, self.x_u_ind] = u_tx
        tx[:, self.x_d_ind] = self.x_d_cur#[self.x_d_cur[i] for i in self.x_d_ind]


        return tx


    def _refine_sample(self, N, e_tol=-1.):
        
        max_batch = self.options['max_batch']
        as_options = self.options['as_options']

        batch_use = max_batch
        if N < max_batch:
            batch_use = N

        if self.model == None:
            raise ValueError("Model not supplied!")
        if self.func == None:
            raise ValueError("Function not supplied!")

        bounds = self.xlimits
        sset = self.options['criteria']
        self.rcrit = GetCriteria(sset, self.model, convert_to_smt_grads(self.model), bounds, self.x_u_ind)
        if not self.options['full_refine']:
            # bounds = self.xlimits[self.x_u_ind]
            self.rcrit.set_static(self.x_d_cur[:,0])

        modelset = copy.deepcopy(self.rcrit.model) # grab a copy of the current model

        # func set in self.func, should not be none
        if self.func == None:
            raise ValueError("func not set in AdaptiveSampler")

        # perform adaptive sampling
        # import pdb; pdb.set_trace()
        mf, rF, d1, d2, d3 = adaptivesampling(self.func, modelset, self.rcrit, bounds, N, e_tol = e_tol, batch=batch_use, options=as_options)


        self.rcrit = rF


        # set evaluations internally
        tx = mf.training_points[None][0][0]
        tf = mf.training_points[None][0][1]
        tg = convert_to_smt_grads(mf)
        self.set_evaluated_func(tf)
        self.set_evaluated_grad(tg)

        # make sure we don't reset attributes
        self.dont_reset = True


        return tx









if __name__ == '__main__':

    x_init = 0.
    # N = [5, 3, 2]
    # xlimits = np.array([[-1., 1.],
    #            [-1., 1.],
    #            [-1., 1.],
    #            [-1., 1.]])
    # pdfs =  [x_init, 'uniform', 'uniform', 'uniform']
    # samp1 = CollocationSampler(np.array([x_init]), N=N,
    #                             xlimits=xlimits, 
    #                             probability_functions=pdfs, 
    #                             retain_uncertain_points=True)
    
    
    from smt.problems import Rosenbrock
    from utils.sutils import convert_to_smt_grads
    N = [5, 3]
    pdfs = ['uniform', 'uniform']
    func = Rosenbrock(ndim=2)
    xlimits = func.xlimits
    samp = CollocationSampler(np.array([x_init]), N=N,
                                xlimits=xlimits, 
                                probability_functions=pdfs, 
                                retain_uncertain_points=True)
    


    xt = samp.current_samples['x']
    ft = func(xt)
    gt = convert_to_smt_grads(func, xt)
    samp.set_evaluated_func(ft)
    samp.set_evaluated_grad(gt)

    from utils.stat_comps import _mu_sigma_comp, _mu_sigma_grad
    stats, vals = _mu_sigma_comp(func, xt.shape[0], xt, xlimits, samp.scales, pdfs, tf = ft, weights=samp.weights)
    # gstats, grads = _mu_sigma_grad(func, xt.shape[0], xt, xlimits, samp.scales, pdfs, tf = ft, tg=gt, weights=samp.weights)
    import pdb; pdb.set_trace()
    #TODO: Make a test for this

