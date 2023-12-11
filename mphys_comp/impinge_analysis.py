import imp
import numpy as np
import argparse
from mpi4py import MPI
import sys

import openmdao.api as om

from mphys import Multipoint
from mphys_comp.shock_angle_comp import ShockAngleComp
from mphys_comp.inflow_comp import InflowComp
from mphys.scenario_aerostructural import ScenarioAeroStructural

# these imports will be from the respective codes' repos rather than mphys
from mphys.mphys_adflow import ADflowBuilder
from beam.mphys_eb import EBBuilder
from beam.mphys_onetoone import OTOBuilder
from beam.om_beamdvs import beamDVComp
#from mphys.mphys_meld import MeldBuilder
#from mphys.mphys_rlt import RltBuilder

from baseclasses import AeroProblem

# from tacs import elements, constitutive, functions

# contains all options, aero, opt, struct, uq, warp
import mphys_comp.impinge_setup as default_impinge_setup

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank

class Top(Multipoint):


    def _declare_options(self):
        self.options.declare('problem_settings', default=default_impinge_setup,
                             desc='default settings for the shock impingement problem, including solver settings.')    
        self.options.declare('full_free', default=False,
                             desc='apply far field conditions at inflow (everywhere)')    
        self.options.declare('use_inflow_comp', default=True,
                             desc='determine downstream settings ')    
        self.options.declare('use_shock_comp', default=False,
                             desc='determine upstream settings using the oblique shock component')    
        self.options.declare('subsonic', default=False,
                             desc='use subsonic mesh specification instead')    

    def setup(self):
        self.impinge_setup = self.options["problem_settings"]

        impinge_setup = self.impinge_setup

        opt_options = impinge_setup.optOptions

        ################################################################################
        # ADflow Setup
        ################################################################################
        aero_options = impinge_setup.aeroOptions
        warp_options = impinge_setup.warpOptions
        if(self.options["full_free"]):
            def_surf = ['symp1','symp2','wall1','wall2','wall3','far','outflow']
        else:
            def_surf = ['symp1','symp2','wall1','wall2','wall3','inflow','far','outflow']
        struct_surf = 'wall2'

        aero_builder = ADflowBuilder(options=aero_options,  scenario="aerostructural", def_surf=def_surf, struct_surf=struct_surf) #mesh_options=warp_options,
        aero_builder.initialize(self.comm)
        # aero_builder.solver.addFunction('cd','wall2','cd_def')
        # aero_builder.solver.addFunction('cdv','wall2','cdv_def')
        # aero_builder.solver.addFunction('cdp','wall2','cdp_def')
        # aero_builder.solver.addFunction('cdm','wall2','cdm_def')
        aero_builder.solver.addFunction('drag','allWalls','d_def')
        # aero_builder.solver.addFunction('drag',None,'d_def')
        aero_builder.solver.addFunction('dragviscous','allWalls', 'dv_def')
        # aero_builder.solver.addFunction('dragviscous',None, 'dv_def')
        aero_builder.solver.addFunction('dragpressure','allWalls','dp_def')
        # aero_builder.solver.addFunction('dragpressure',None,'dp_def')
        aero_builder.solver.addFunction('dragmomentum','allWalls','dm_def')
        # aero_builder.solver.addFunction('dragmomentum',None,'dm_def')
        # aero_builder.solver.addFunction('cd',aero_builder.solver.allWallsGroup,'cd_def')
        # aero_builder.solver.addFunction('cdv',aero_builder.solver.allWallsGroup,'cdv_def')
        # aero_builder.solver.addFunction('cdp',aero_builder.solver.allWallsGroup,'cdp_def')
        # aero_builder.solver.addFunction('cdm',aero_builder.solver.allWallsGroup,'cdm_def')

        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # Euler Bernoulli Setup
        ################################################################################
        struct_options = impinge_setup.structOptions

        struct_builder = EBBuilder(struct_options)
        struct_builder.initialize(self.comm)
        ndv_struct = struct_builder.get_ndv()

        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())

        ################################################################################
        # Transfer Scheme Setup
        ################################################################################

        self.onetoone = True

        if(self.onetoone):
            ldxfer_builder = OTOBuilder(aero_builder, struct_builder)
            ldxfer_builder.initialize(self.comm)
        else:
            isym = -1
            ldxfer_builder = MeldBuilder(aero_builder, struct_builder, isym=isym, n=2)
            ldxfer_builder.initialize(self.comm)

        ################################################################################
        # MPHYS Setup
        ################################################################################

        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        # # component to determine post-shock flow properties for ADFlow
        if(self.options["use_shock_comp"]):
            dvs.add_output("shock_angle", opt_options["shock_angle"])
            self.add_subsystem("shock", ShockAngleComp())

        else:
            dvs.add_output("M1", val=impinge_setup.mach)
            dvs.add_output("beta", val=impinge_setup.beta)
            dvs.add_output("P1", val=impinge_setup.P)
            dvs.add_output("T1", val=impinge_setup.T)
            # dvs.add_output("r1", val=impinge_setup.rho)

        if(self.options["use_inflow_comp"]):
            dvs.add_output("M0", impinge_setup.M0)
            dvs.add_output('P0', impinge_setup.P0)
            dvs.add_output('T0', impinge_setup.T0)
            self.add_subsystem("upstream", InflowComp())
        else:
            dvs.add_output("vx0", impinge_setup.VX)
            dvs.add_output('P0', impinge_setup.P0)
            dvs.add_output('r0', impinge_setup.r0)

        

        #dvs.add_output("beta", impinge_setup.beta)
        dvs.add_output("dv_struct_TRUE", struct_options["th_true"])
        dvs.add_output("rsak", aero_options["SAConsts"][0])

        dvs.add_output("mass")
        dvs.add_output("stresscon")

        self.add_subsystem("dv_interp", beamDVComp(ndv = struct_options["ndv_true"], method='bsplines'))


        nonlinear_solver = om.NonlinearBlockGS(maxiter=2, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        linear_solver = om.LinearBlockGS(maxiter=25, iprint=2, use_aitken=True, rtol=1e-14, atol=1e-14)
        scenario = "test"
        self.mphys_add_scenario(
            scenario,
            ScenarioAeroStructural(
                aero_builder=aero_builder, struct_builder=struct_builder, ldxfer_builder=ldxfer_builder
            ),
            nonlinear_solver,
            linear_solver,
        )

        for discipline in ["aero", "struct"]:
            self.mphys_connect_scenario_coordinate_source("mesh_%s" % discipline, scenario, discipline)

        

    def configure(self):
        # create the aero problem 
        impinge_setup = self.impinge_setup

        ap = AeroProblem(
            name=impinge_setup.probName,
            mach=impinge_setup.mach,
            alpha =impinge_setup.alpha,
            beta =impinge_setup.beta,
            areaRef = 1.0,
            chordRef = 1.0,
            T = impinge_setup.T, 
            P = impinge_setup.P, 
            evalFuncs=["d_def", "dv_def","dm_def","dp_def"],
        )    

        BCVarFuncs = ["Pressure", "PressureStagnation", "Temperature", "Density", "DensityStagnation", "VelocityX", "TemperatureStagnation", "Thrust", "Heat"]
        ap.possibleBCDVs = set(BCVarFuncs)

        ap.addDV("mach", value=impinge_setup.mach, name="mach1")
        ap.addDV("beta", value=impinge_setup.beta, name="beta")
        ap.addDV("P", value=impinge_setup.P, name="pressure1")
        ap.addDV("T", value=impinge_setup.T, name="temperature1")

        # set BC vars that are hard coded in the CGNS mesh, upstream properties
        if(not self.options["full_free"]):
            if self.options["subsonic"]:
                ap.setBCVar("PressureStagnation", impinge_setup.P0, "inflow")
                ap.setBCVar("DensityStagnation",  impinge_setup.r0, "inflow")

                ap.addDV("PressureStagnation", family="inflow", name="pressure0")
                ap.addDV("DensityStagnation",  family="inflow", name="density0")
            else:
                ap.setBCVar("Pressure", impinge_setup.P0, "inflow")
                ap.setBCVar("Density",  impinge_setup.r0, "inflow")
                ap.setBCVar("VelocityX", impinge_setup.VX, "inflow")

                ap.addDV("Pressure", family="inflow", name="pressure0")
                ap.addDV("Density",  family="inflow", name="density0")
                ap.addDV("VelocityX", family="inflow", name="velocityx0")

        self.test.coupling.aero.mphys_set_ap(ap)
        self.test.aero_post.mphys_set_ap(ap)
        # self.test.struct_post.funcs.setup_outs()

        # TODO: Customize SA var connection
        # self.connect("rsak", "test.coupling.aero.rsak")
        self.connect("dv_struct_TRUE", "dv_interp.DVS")
        self.connect("dv_interp.th", "test.dv_struct")
        # ### NOTE TODO ALERT TODO NOTE ### THIS IS MESSED UP
        # WHAT DO YOU MEAN THEY DONT EXIST
        # self.connect("mass","test.coupling.struct.mass")
        # self.connect("stresscon","test.coupling.struct.stresscon")
        # import pdb; pdb.set_trace()
        # self.connect("mass","test.aero_post.d_def")
        # self.connect("mass","test.struct_post.mass.mass")
        # self.connect("stresscon","test.struct_post.funcs.stresscon")

        if(self.options["use_shock_comp"]):
            self.connect("shock_angle", "shock.shock_angle")
            self.connect("M0", "shock.mach0")
            self.connect("P0", "shock.P0")
            self.connect("T0", "shock.T0")

            self.connect("shock.mach1", "test.coupling.aero.mach1")
            self.connect("shock.flow_angle", "test.coupling.aero.beta")
            self.connect("shock.T1", "test.coupling.aero.temperature1")
            self.connect("shock.P1", "test.coupling.aero.pressure1")
            self.connect("shock.mach1", "test.aero_post.mach1")
            self.connect("shock.flow_angle", "test.aero_post.beta")
            self.connect("shock.T1", "test.aero_post.temperature1")
            self.connect("shock.P1", "test.aero_post.pressure1")
        else:
            self.connect("M1", "test.coupling.aero.mach1")
            self.connect("M1", "test.aero_post.mach1")
            self.connect("beta", "test.coupling.aero.beta")
            self.connect("beta", "test.aero_post.beta")
            self.connect("P1", "test.coupling.aero.pressure1")
            self.connect("P1", "test.aero_post.pressure1")
            self.connect("T1", "test.coupling.aero.temperature1")
            self.connect("T1", "test.aero_post.temperature1")
        if(not self.options["full_free"]):
            if(self.options["use_inflow_comp"]):
                self.connect("M0", "upstream.mach0")
                self.connect("P0", "upstream.P0")
                self.connect("T0", "upstream.T0")

                if not self.options["subsonic"]:
                    self.connect("upstream.VelocityX", "test.coupling.aero.velocityx0")
                    self.connect("upstream.Density", "test.coupling.aero.density0")
                    self.connect("upstream.Pressure", "test.coupling.aero.pressure0")
                if not self.options["subsonic"]:
                    self.connect("upstream.VelocityX", "test.aero_post.velocityx0")
                    self.connect("upstream.Density", "test.aero_post.density0")
                    self.connect("upstream.Pressure", "test.aero_post.pressure0")
            else:
                if not self.options["subsonic"]:
                    self.connect("vx0", "test.coupling.aero.velocityx0")
                    self.connect("r0", "test.coupling.aero.density0")
                    self.connect("P0", "test.coupling.aero.pressure0")
                if not self.options["subsonic"]:
                    self.connect("vx0", "test.aero_post.velocityx0")
                    self.connect("r0", "test.aero_post.density0")
                    self.connect("P0", "test.aero_post.pressure0")
        
        

# use as scratch space for playing around
if __name__ == '__main__':

    ################################################################################
    # OpenMDAO setup
    ################################################################################

    use_shock = True
    use_inflow = True
    full_far = False
    subsonic = False

    problem_settings = default_impinge_setup
    # problem_settings.aeroOptions['equationType'] = 'laminar NS'
    # problem_settings.aeroOptions['equationType'] = 'Euler'
    problem_settings.aeroOptions['NKSwitchTol'] = 1e-6 #1e-6
    # problem_settings.aeroOptions['NKSwitchTol'] = 1e-3 #1e-6
    problem_settings.aeroOptions['nCycles'] = 5000000
    problem_settings.aeroOptions['L2Convergence'] = 1e-15
    problem_settings.aeroOptions['printIterations'] = False
    problem_settings.aeroOptions['printTiming'] = True
    problem_settings.mach = 2.7
    # problem_settings.beta = 0.8
    # problem_settings.mach = 0.8


    if full_far:
        aeroGridFile = f'../meshes/imp_TEST_73_73_25.cgns'
    elif subsonic:
        aeroGridFile = f'../meshes/imp_subs_73_73_25.cgns'
    else:
        aeroGridFile = f'../meshes/imp_mphys_73_73_25.cgns'
    problem_settings.aeroOptions['gridFile'] = aeroGridFile
    nelem = 30
    problem_settings.nelem = nelem
    problem_settings.structOptions['Nelem'] = nelem
    problem_settings.structOptions['force'] = np.ones(nelem+1)*1.0
    problem_settings.structOptions["th"] = np.ones(nelem+1)*0.0005

    prob = om.Problem()
    prob.model = Top(problem_settings=problem_settings, subsonic=subsonic,
                                                         use_shock_comp=use_shock, 
                                                         use_inflow_comp=use_inflow, 
                                                         full_free=full_far)
    
    # prob.model.add_design_var("M1")
    # prob.model.add_design_var("beta")
    # prob.model.add_design_var("P1")
    # prob.model.add_design_var("T1")
    # prob.model.add_design_var("r1")

    # prob.model.add_design_var("P0")
    # prob.model.add_design_var("M0")
    # prob.model.add_design_var("T0")
    # prob.model.add_design_var("vx0")
    # prob.model.add_design_var("r0")
    # prob.model.add_design_var("P0")

    prob.model.add_design_var("dv_struct")
    # prob.model.add_design_var("shock_angle")

    # prob.model.add_objective("test.aero_post.cd_def")
    # prob.model.add_objective("test.aero_post.cdv_def")
    # prob.model.add_objective("test.aero_post.cdp_def")
    # prob.model.add_objective("test.aero_post.cdm_def")
    # prob.model.add_objective("test.aero_post.d_def")
    # prob.model.add_objective("test.aero_post.dv_def")
    prob.model.add_objective("test.aero_post.dp_def")
    # prob.model.add_objective("test.struct_post.stress_con")
    # prob.model.add_objective("test.aero_post.dm_def")
    # prob.model.add_objective("test.aero_post.drag")
    # prob.model.add_objective("test.aero_post.dragviscous")
    # prob.model.add_objective("test.aero_post.dragpressure")
    # prob.model.add_objective("test.aero_post.dragmomentum")
    
    prob.setup(mode='rev')
    om.n2(prob, show_browser=False, outfile="mphys_as_adflow_eb_%s_2pt.html")
    #prob.set_val("mach", 2.)
    #prob.set_val("dv_struct", impinge_setup.structOptions["th"])
    #prob.set_val("beta", 7.)
    #x = np.linspace(2.5, 3.5, 10)

    #y = np.zeros(10)
    #for i in range(10):
    #prob.set_val("M0", x[i])
    # prob.set_val("shock_angle", 25.)
    
    # import pdb; pdb.set_trace()
    #prob.model.approx_totals()
    prob.run_model()
    # import copy
    # y0 = copy.deepcopy(prob.get_val("test.aero_post.cd_def"))
    # #totals1 = prob.compute_totals(wrt='rsak')
    # #prob.model.approx_totals()
    # totals2 = prob.compute_totals(of='test.aero_post.cd_def', wrt=['shock.mach1','shock_angle','rsak'])

    # h = 1e-8

    # prob.set_val("rsak", 0.41 + h)
    # prob.run_model()
    # y1k = copy.deepcopy(prob.get_val("test.aero_post.cd_def"))
    # prob.set_val("rsak", 0.41)
    # prob.set_val("shock.mach1", default_impinge_setup.mach+h)
    # prob.run_model()
    # y1s = copy.deepcopy(prob.get_val("test.aero_post.cd_def"))
    
    # fds = (y1s-y0)/h
    # fdk = (y1k - y0)/h

    prob.check_totals(step_calc='rel_avg')

    #prob.check_partials()
    import pdb; pdb.set_trace()
    #prob.model.list_outputs()

    # if MPI.COMM_WORLD.rank == 0:
    #     print("cd = %.15f" % prob["test.aero_post.cd_def"])
    #     print(y)
    #     prob.model.