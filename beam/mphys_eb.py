import numpy as np
import copy
import openmdao.api as om
from mphys.builder import Builder
from beam.beam_solver import EulerBeamSolver
from scipy.sparse.linalg.dsolve import spsolve
from mphys import Multipoint
from mphys.scenario_structural import ScenarioStructural
import mphys_comp.impinge_setup as default_impinge_setup

from beam.om_beamdvs import beamDVComp

"""
Wrapper for the beam solver, incorporating as a structural solver in mphys
"""

class EBMesh(om.IndepVarComp):
    """
    Component to read the initial mesh coordinates for the Euler-Bernoulli solver, in the shape of the aero mesh
    """
    def initialize(self):
        self.options.declare('eb_solver', default = None, desc='the beam_solver object', recordable=False)

    def setup(self):
        eb_solver = self.options['eb_solver']
        xpts = eb_solver.getMeshPoints()

        pts = np.zeros(xpts.size*3)
        for i in range(xpts.size):
            pts[3*i] = xpts[i]

        # for the sake of the 2D impingement problem, double up this vector and shift by 1 in the y direction
        pts2 = np.zeros(xpts.size*3)
        for i in range(xpts.size):
            pts2[3*i+1] = 1.0
            pts2[3*i] = xpts[i]
        pts = np.append(pts, pts2)

        self.add_output('x_struct0', distributed=False, val=pts, shape=pts.size, desc='structural node coordinates', tags=['mphys_coordinates'])

class EBSolver(om.ImplicitComponent):
    """
    Component to perform Euler-Bernoulli structural analysis

        - The steady residual is R = K * u_s - f_s = 0

    """
    def initialize(self):

        self.options.declare('struct_objects', recordable=False)
        self.options.declare('check_partials')        

        self.res = None
        self.ans = None
        self.struct_rhs = None
        self.x_save = None

        self.transposed = False
        self.check_partials = False

        self.ndv = None
        self.npnt = None
        self.ndof = None

        self.beam_solver = None
        self.beam_dict = None

        self.old_dvs = None
        self.old_xs = None

    def setup(self):
        self.check_partials = self.options['check_partials']
        self.beam_solver = self.options["struct_objects"][1]
        self.beam_dict = self.options["struct_objects"][0]

        self.ndv =  self.beam_dict['ndv'] #this is just beam thickness for now
        self.npnt = self.beam_dict['ndv']
        self.ndof = self.beam_dict['ndof']

        # create some vectors that we'll need
        self.res        = np.zeros(self.ndof)
        self.Iyy        = np.zeros(self.npnt)
        self.force      = np.zeros(self.npnt)
        self.ans        = np.zeros(2*(self.npnt-2))
        self.struct_rhs = np.zeros(2*(self.npnt-2))

        # OpenMDAO setup
        state_size = len(self.ans)
        # inputs
        self.add_input('dv_struct', distributed=False, shape=self.ndv, desc='tacs design variables', tags=['mphys_input'])
        self.add_input('x_struct0', distributed=False, shape_by_conn=True, desc='structural node coordinates',tags=['mphys_coordinates'])
        self.add_input('struct_force',  distributed=False, val=np.ones(self.npnt), desc='structural load vector', tags=['mphys_coupling'])

        # outputs
        # its important that we set this to zero since this displacement value is used for the first iteration of the aero
        self.add_output('struct_states', distributed=False, shape=state_size, val = np.ones(state_size), desc='structural state vector', tags=['mphys_coupling'])

        # partials
        self.declare_partials('struct_states',['dv_struct','struct_force','struct_states'])

    def _need_update(self,inputs):

        update = False

        if self.old_dvs is None:
            self.old_dvs = inputs['dv_struct'].copy()
            update =  True

        for dv, dv_old in zip(inputs['dv_struct'],self.old_dvs):
            if np.abs(dv - dv_old) > 0.:#1e-7:
                self.old_dvs = inputs['dv_struct'].copy()
                update =  True

        if self.old_xs is None:
            self.old_xs = inputs['x_struct0'].copy()
            update =  True

        for xs, xs_old in zip(inputs['x_struct0'],self.old_xs):
            if np.abs(xs - xs_old) > 0.:#1e-7:
                self.old_xs = inputs['x_struct0'].copy()
                update =  True

        return update

    def _update_internal(self,inputs,outputs=None):
        #if self._need_update(inputs):
            # update Iyy
        self.beam_solver.computeRectMoment(np.array(inputs['dv_struct']))
        self.beam_solver.setThickness(inputs['dv_struct'])
            # have this function call setIyy internally

            # update force
            # import pdb; pdb.set_trace()
            # self.beam_solver.setLoad(np.array(inputs['struct_force']))

        if outputs is not None:
            self.beam_solver.u = outputs['struct_states']
        self.beam_solver.setLoad(np.array(inputs['struct_force']))


    def apply_nonlinear(self, inputs, outputs, residuals):
        
        self._update_internal(inputs,outputs)
        
        res  = self.beam_solver.getResidual(outputs['struct_states'])
        residuals['struct_states'] = res

    def solve_nonlinear(self, inputs, outputs):

        self._update_internal(inputs,outputs)

        ans  = self.beam_solver()
        outputs['struct_states'] = ans

    def linearize(self, inputs, outputs, partials):

        self._update_internal(inputs,outputs)
        self.beam_solver.assemble()
        partials['struct_states','struct_states'] = copy.deepcopy(self.beam_solver.A.real.todense())

        #ans  = self.beam_solver()
        dAudth, dbdf = self.beam_solver.evalAssembleSens()
        partials['struct_states','struct_force'] = dbdf
        partials['struct_states','dv_struct'] = dAudth
        #import pdb; pdb.set_trace()

    def solve_linear(self,d_outputs,d_residuals,mode):

        if mode == 'fwd':
            if self.check_partials:
                print ('solver fwd')
            else:
                raise ValueError('forward mode requested but not implemented')

        if mode == 'rev':
            res_array = d_outputs['struct_states']
            d_residuals['struct_states'] = spsolve(self.beam_solver.A.T, res_array)
        
        if self.comm.rank == 0:
            print("Out of solve linear in mphys_eb", flush=True)

class EBGroup(om.Group):
    def initialize(self):
        self.options.declare('solver_objects', recordable=False)
        self.options.declare("aero_coupling", default=False)
        self.options.declare('check_partials')

    def setup(self):
        self.struct_objects = self.options['solver_objects']
        self.aero_coupling = self.options['aero_coupling']
        self.check_partials = self.options['check_partials']

        if self.aero_coupling:
            self.add_subsystem(
                "force",
                EBForce(struct_objects=self.struct_objects),
                promotes_inputs=["f_struct"],
                promotes_outputs=["struct_force"],
            )

        self.add_subsystem('solver', EBSolver(
            struct_objects=self.struct_objects,
            check_partials=self.check_partials),
            promotes_inputs=['struct_force', 'x_struct0', 'dv_struct'],
            promotes_outputs=['struct_states']
        )

        if self.aero_coupling:
            self.add_subsystem(
                "disp",
                EBDisp(struct_objects=self.struct_objects),
                promotes_inputs=["struct_states"],
                promotes_outputs=["u_struct"],
            )

class EBFuncsGroup(om.Group):
    def initialize(self):
        self.options.declare('solver_objects', recordable=False)
        self.options.declare('check_partials')

    def setup(self):
        self.struct_objects = self.options['solver_objects']
        self.check_partials = self.options['check_partials']
        self.func_list = self.struct_objects[0]['get_funcs']
        func_no_mass = []
        for func in self.func_list:
            if func not in ['mass']:
                func_no_mass.append(func)

        self.add_subsystem('funcs', EBFunctions(
            struct_objects=self.struct_objects,
            check_partials=self.check_partials),
            promotes_inputs=['x_struct0', 'struct_states','dv_struct'],
            promotes_outputs=func_no_mass
        )

        self.add_subsystem('mass', EBMass(
            struct_objects=self.struct_objects,
            check_partials=self.check_partials),
            promotes_inputs=['x_struct0', 'dv_struct'],
            promotes_outputs=['mass'],
        )
        

class EBBuilder(Builder):

    def __init__(self, options, check_partials=False):
        self.options = options
        self.check_partials = check_partials

    def initialize(self, comm):
        solver_dict={}

        name = self.options['name']

        ndv = (self.options['Nelem']+1)

        ndof = 3 #bad hacky way to do this

        number_of_nodes = (self.options['Nelem']+1)

        solver_dict['ndv']    = ndv
        solver_dict['ndof']   = ndof
        solver_dict['number_of_nodes'] = number_of_nodes
        solver_dict['get_funcs'] = self.options['get_funcs']

        # create solver
        beam_solver_obj = EulerBeamSolver(self.options)

        # check if the user provided a load function
        if 'force' in self.options.keys():
            solver_dict['force'] = self.options['force']

        self.solver_objects = [0, 0]
        self.solver_objects[0] = solver_dict
        self.solver_objects[1] = beam_solver_obj

    def get_coupling_group_subsystem(self):
        return EBGroup(solver_objects=self.solver_objects,
                        aero_coupling=True,
                        check_partials=self.check_partials)

    def get_mesh_coordinate_subsystem(self):
        return EBMesh(eb_solver=self.solver_objects[1])

    def get_post_coupling_subsystem(self):
        return EBFuncsGroup(solver_objects=self.solver_objects,
                            check_partials=self.check_partials)

    def get_ndof(self):
        return self.solver_objects[0]['ndof']

    def get_number_of_nodes(self):
        return 2*self.solver_objects[0]['number_of_nodes']

    def get_ndv(self):
        return self.solver_objects[0]['ndv']






class EBForce(om.ExplicitComponent):
    """
    OpenMDAO component that wraps z forces to the beam solver

    """

    def initialize(self):
        self.options.declare('struct_objects', recordable=False)

    def setup(self):
        # self.set_check_partial_options(wrt='*',directional=True)

        self.solver = self.options["struct_objects"][1]
        solver = self.solver

        self.add_input("f_struct", distributed=False, shape_by_conn=True, tags=["mphys_coupling"])

        local_coord_size = solver.getMeshPoints().size

        self.add_output("struct_force", distributed=False, shape=local_coord_size, val=np.zeros(local_coord_size), tags=["mphys_coupling"])

        self.declare_partials("*","*")

    def compute(self, inputs, outputs):

        solver = self.solver

        f = inputs["f_struct"]
        
        f_z = np.zeros(int(len(f)/6))
        for i in range(len(f_z)):
            f_z[i] = -f[3*i+2]

        # u_struct = np.zeros(2*len(u_z)*3)
        # for i in range(len(u_z)):
        #     u_struct[3*i+2] = u_z[i]
        #     u_struct[len(u_z)*3 + 3*i+2] = u_z[i]
        #import pdb; pdb.set_trace()
        outputs["struct_force"] = f_z

    def compute_partials(self, inputs, partials):

        solver = self.solver

        f = inputs["f_struct"]
        
        f_z = np.zeros(int(len(f)/6))
        for i in range(len(f_z)):
            f_z[i] = f[3*i+2]

        dfz = np.zeros([int(len(f)/6),len(f)])
        for i in range(len(f_z)):
            dfz[i,3*i+2] = -1.


        if self.comm.rank == 0:
            print("Out of compute partials in mphys_eb forces", flush=True)
        partials["struct_force","f_struct"] = dfz


class EBDisp(om.ExplicitComponent):
    """
    OpenMDAO component that wraps z displacements from the beam solver

    """

    def initialize(self):
        self.options.declare('struct_objects', recordable=False)

    def setup(self):
        # self.set_check_partial_options(wrt='*',directional=True)

        self.solver = self.options["struct_objects"][1]
        solver = self.solver

        self.add_input("struct_states", distributed=False, shape_by_conn=True, tags=["mphys_coupling"])

        local_coord_size = 2*solver.getMeshPoints().size*3
        #import pdb; pdb.set_trace()
        self.add_output("u_struct", distributed=False, shape=local_coord_size, val=np.zeros(local_coord_size), tags=["mphys_coupling"])

        self.declare_partials("*","*")

    def compute(self, inputs, outputs):

        solver = self.solver

        #u = solver.getSolution() #should actually get this from inputs instead
        u = inputs['struct_states']
        u_z = np.zeros(int(len(u)/2 + 2))
        for i in range(1,len(u_z)-1):
            u_z[i] = u[2*i-2]

        u_struct = np.zeros(2*len(u_z)*3)
        for i in range(len(u_z)):
            u_struct[3*i+2] = u_z[i]
            u_struct[len(u_z)*3 + 3*i+2] = u_z[i]

        outputs["u_struct"] = u_struct

    def compute_partials(self, inputs, partials):

        solver = self.solver

        #u = solver.getSolution() #should actually get this from inputs instead
        u = inputs['struct_states']
        u_z = np.zeros(int(len(u)/2 + 2))

        u_struct = np.zeros(2*len(u_z)*3)

        duz = np.zeros([len(u_z),len(u)])
        for i in range(1,len(u_z)-1):
            duz[i,2*i-2] = 1.

        dus = np.zeros([len(u_struct),len(u_z)])
        for i in range(len(u_z)):
            dus[3*i+2,i] = 1.
            dus[len(u_z)*3 + 3*i+2, i] = 1.
        
        if self.comm.rank == 0:
            print("Out of compute partials in mphys_eb disp", flush=True)

        partials["u_struct","struct_states"] = np.matmul(dus, duz)



class EBFunctions(om.ExplicitComponent):
    """
    Component to compute functions of the Euler-Bernoulli solver
    """
    def initialize(self):
        self.options.declare('struct_objects', recordable=False)
        self.options.declare('check_partials')
        
        self.check_partials = False

    def setup(self):

        self.struct_objects = self.options['struct_objects']
        self.check_partials = self.options['check_partials']

        self.ndv = self.struct_objects[0]['ndv']
        self.func_list = self.struct_objects[0]['get_funcs']
        self.beam_solver = self.struct_objects[1]

        # OpenMDAO part of setup
        # TODO move the dv_struct to an external call where we add the DVs
        self.add_input('dv_struct', distributed=False, shape = self.ndv, desc='design variables', tags=['mphys_input'])
        self.add_input('x_struct0', distributed=False, shape_by_conn=True, desc='structural node coordinates',tags=['mphys_coordinates'])
        self.add_input('struct_states', distributed=False, shape_by_conn=True, desc='structural state vector', tags=['mphys_coupling'])

        # Remove the mass function from the func list if it is there
        # since it is not dependent on the structural state
        func_no_mass = []
        for func in self.func_list:
            if func not in ['mass']:
                func_no_mass.append(func)

        self.func_list = func_no_mass
        for item in self.func_list:
            if item == 'stress':
                ishape = self.beam_solver.Nelem + 1
            else:
                ishape = 1
            self.add_output(item, distributed=False, shape=ishape, tags=['mphys_result'])
            # declare the partials
            self.declare_partials(item,['dv_struct','struct_states'])

    def setup_outs(self):
        self.func_list = self.struct_objects[0]['get_funcs']
        func_no_mass = []
        for func in self.func_list:
            if func not in ['mass']:
                func_no_mass.append(func)

        self.func_list = func_no_mass
        for item in self.func_list:
            if item == 'stress':
                ishape = self.beam_solver.Nelem + 1
            else:
                ishape = 1
            self.add_output(item, distributed=False, shape=ishape, tags=['mphys_result'])
            # declare the partials
            self.declare_partials(item,['dv_struct','struct_states'])
    def _update_internal(self,inputs):
        # update Iyy
        self.beam_solver.computeRectMoment(np.array(inputs['dv_struct']))
        self.beam_solver.setThickness(inputs['dv_struct'])
        self.beam_solver.setStates(inputs['struct_states'])
        # have this function call setIyy internally

    def compute(self,inputs,outputs):
        self._update_internal(inputs)

        thing = self.beam_solver.evalFunctions(self.func_list)
        for item in self.func_list:
            outputs[item] = thing[item]

    def compute_partials(self, inputs, partials):
        self._update_internal(inputs)

        self.beam_solver.evalFunctions(self.func_list)
        thingth = self.beam_solver.evalFuncThSens(self.func_list)
        thingst = self.beam_solver.evalFuncStateSens(self.func_list)
        for item in self.func_list:
            partials[item,'dv_struct'] = thingth[item]
            partials[item,'struct_states'] = thingst[item]

        if self.comm.rank == 0:
            print("Out of compute partials in mphys_eb functions", flush=True)


class EBMass(om.ExplicitComponent):
    """
    Component to compute TACS mass
    """
    def initialize(self):
        self.options.declare('struct_objects', recordable=False)
        self.options.declare('check_partials')

        self.mass = False

        self.check_partials = False

    def setup(self):

        self.struct_objects = self.options['struct_objects']
        self.check_partials = self.options['check_partials']

        #self.set_check_partial_options(wrt='*',directional=True)

        self.beam_solver = self.struct_objects[1]

        self.ndv  = self.struct_objects[0]['ndv']

        # OpenMDAO part of setup
        self.add_input('dv_struct', distributed=False, shape=self.ndv, desc='design variables', tags=['mphys_input'])
        self.add_input('x_struct0', distributed=False, shape_by_conn=True, desc='structural node coordinates', tags=['mphys_coordinates'])
        self.add_output('mass', val=0.0, distributed=False, desc = 'structural mass', tags=['mphys_result'])
        self.declare_partials('mass',['dv_struct'])

    def _update_internal(self,inputs):
        # update Iyy
        self.beam_solver.computeRectMoment(np.array(inputs['dv_struct']))
        self.beam_solver.setThickness(inputs['dv_struct'])
        # have this function call setIyy internally


    def compute(self,inputs,outputs):
        self._update_internal(inputs)

        if 'mass' in outputs:
            thing = self.beam_solver.evalFunctions(['mass'])['mass']
            outputs['mass'] = thing

    
    def compute_partials(self, inputs, partials):
        if self.check_partials:
            self._update_internal(inputs)

        self.beam_solver.evalFunctions(['mass'])['mass']
        dm_ddv = self.beam_solver.evalMassThSens()

        if self.comm.rank == 0:
            print("Out of compute partials in mphys_eb mass", flush=True)
        partials['mass','dv_struct'] = dm_ddv#list(self.beam_solver.evalthSens(['mass']).values())




class Top(Multipoint):

    def _declare_options(self):
        self.options.declare('problem_settings', default=default_impinge_setup,
                             desc='default settings for the shock impingement problem, including solver settings.')    


    def setup(self):
        self.impinge_setup = self.options["problem_settings"]
        ################################################################################
        # Euler Bernoulli Setup
        ################################################################################
        struct_options = self.impinge_setup.structOptions
        struct_builder = EBBuilder(struct_options)
        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # MPHYS Setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
       
        # force format annoyance
        fm = struct_options["force"].size
        f0 = np.zeros(fm*6)
        for i in range(fm):
            f0[3*i+2] = struct_options["force"][i]

        dvs.add_output("f_struct", f0)
        dvs.add_output("dv_struct_TRUE", struct_options["th_true"])

        # thickness interp subsys
        self.add_subsystem("dv_interp", beamDVComp(ndv = struct_options["ndv_true"], method='bsplines'))

        nonlinear_solver = om.NonlinearBlockGS(maxiter=2, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        scenario = "test"
        self.mphys_add_scenario(
            scenario,
            ScenarioStructural(
                struct_builder=struct_builder
            ),
            nonlinear_solver,
            linear_solver,
        )

        for discipline in ["struct"]:
            self.mphys_connect_scenario_coordinate_source("mesh_%s" % discipline, scenario, discipline)

        

    def configure(self):
        # create the aero problem 
        impinge_setup = self.impinge_setup

        self.connect("f_struct", f"test.f_struct")
        # self.connect("dv_struct", f"test.dv_struct")
        # self.connect("dv_struct_TRUE", "dv_interp.DVS")
        self.connect("dv_interp.th", "test.dv_struct")




if __name__ == '__main__':

    ################################################################################
    # OpenMDAO setup
    ################################################################################


    nelem = 117
    problem_settings = default_impinge_setup
    problem_settings.nelem = nelem
    ndv_true = problem_settings.ndv_true
    problem_settings.structOptions["Nelem"] = nelem
    problem_settings.structOptions["force"] = np.ones((nelem+1))*1000000000.0
    problem_settings.structOptions["th"] = np.ones(nelem+1)*0.0004
    problem_settings.structOptions['smax'] = 5e5
    problem_settings.structOptions['E'] = 69000000000
    # problem_settings.structOptions['smax'] = .20
    prob = om.Problem()
    prob.model = Top(problem_settings=problem_settings)
    

    prob.model.add_design_var("dv_struct_TRUE")
    prob.model.add_design_var("dv_interp.DVS")
    # prob.model.add_objective("test.struct_post.func_struct")
    # prob.model.add_objective("test.struct_post.stresscon")

    
    prob.setup(mode='rev')
    # om.n2(prob, show_browser=False, outfile="mphys_as_adflow_eb_%s_2pt.html")
    #prob.set_val("mach", 2.)
    # prob.set_val("dv_struct_TRUE", np.ones(ndv_true)*0.001)
    prob.run_model()
    # print(prob.get_val("test.mass"))
    # # import pdb; pdb.set_trace()

    # prob.set_val("dv_struct_TRUE", np.ones(ndv_true)*0.0015)
    # prob.run_model()
    # print(prob.get_val("test.mass"))
    # # import pdb; pdb.set_trace()
    # # f1 = copy.deepcopy(prob.get_val("test.struct_post.func_struct"))
    # # prob.set_val("dv_struct", np.ones(nelem+1)*0.001)
    # # prob.run_model()
    # # f2 = copy.deepcopy(prob.get_val("test.struct_post.func_struct"))
    # #prob.set_val("beta", 7.)
    # #x = np.linspace(2.5, 3.5, 10)

    # print(prob.get_val("test.struct_states"))
    # print(np.mean(prob.get_val("test.struct_states")))
    #prob.model.approx_totals()


    # # prob.check_totals(step_calc='rel_avg')

    prob.check_partials(step_calc = 'rel_avg', minimum_step = 1e-15) # settings are important since displacements are so small
    # print(prob.get_val('test.stress'))
    # print(prob.get_val('test.stresscon'))
    # import pdb; pdb.set_trace()
    #prob.model.list_outputs()

    # if MPI.COMM_WORLD.rank == 0:
    #     print("cd = %.15f" % prob["test.aero_post.cd_def"])
    #     print(y)
    #     prob.model.