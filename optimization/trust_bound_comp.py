import openmdao.api as om
import numpy as np

from utils.om_utils import get_om_design_size

class TrustBound(om.ExplicitComponent):
    """
    Component defining proximity to and violation of a trust radius for design
    traversal
    """
    def initialize(self):
    
        # radius and location are updated here externally to match with the 
        # proper optimization

        
        self.radius = -1.

        # location can't just be the current state of the problem, it needs
        # to stay consistent during a driver run
    
        self.location = None
    
        # need to initialize this component with a list of dvs
        self.options.declare("dv_dict", default=None, recordable=False)
        self.options.declare("initial_trust_radius", default=-1., recordable=False)

    def setup(self):
        dv_settings = self.options["dv_dict"]
        self.radius = self.options["initial_trust_radius"]
        # if dv_settings is None:
        #     Exception("TrustBound component requires an OpenMDAO list of DVs and metadata!")
        
        

        # # add each design variable as an input, according to the passed dv_dict
        self.add_output('c_trust')
        # import pdb; pdb.set_trace()
        for name, meta in dv_settings.items():
            self.add_input(name, shape=meta['size']) 
            self.declare_partials('c_trust', name)
            # self.connect() # do this via promotes_inputs
    
         # no values in here

        # NOTE: this is computed as proximity to trust bound, i.e. rad - norm(sk - zk) = c
        # constraint should be set with lower=0.0

    def compute(self, inputs, outputs):

        location = self.location
        radius = self.radius
        
        location_list = self.options["dv_dict"]


        # no bound
        if radius < 0.0 or location == None:
            outputs['c_trust'] = 10.
            return
    
        tsize = get_om_design_size(location_list)

        # bound
        center_desvar_array = np.zeros(tsize)
        candid_desvar_array = np.zeros(tsize)

        i = 0
        # import pdb; pdb.set_trace()
        for name, meta in location_list.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            center_desvar_array[i:i + size] = location[f'dvs.{name}']
            candid_desvar_array[i:i + size] = inputs[name]
            i += size

        dist_flat = candid_desvar_array - center_desvar_array
        dist = np.linalg.norm(dist_flat)
        # import pdb; pdb.set_trace()
        outputs['c_trust'] = radius**2 - dist**2



    
    def compute_partials(self, inputs, partials):

        # no bound
        location = self.location
        radius = self.radius

        location_list = self.options["dv_dict"]

        # no bound
        if radius < 0.0 or location == None:
            for name, meta in location.items():
                partials['c_trust', name] = 0.0
            return
        
        tsize = get_om_design_size(location_list)

        # bound
        center_desvar_array = np.zeros(tsize)
        candid_desvar_array = np.zeros(tsize)

        i = 0
        for name, meta in location_list.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']
            center_desvar_array[i:i + size] = location[f'dvs.{name}']
            candid_desvar_array[i:i + size] = inputs[name]
            i += size

        dist_flat = candid_desvar_array - center_desvar_array
        dist = np.linalg.norm(dist_flat)

        i = 0
        for name, meta in location_list.items():
            size = meta['global_size'] if meta['distributed'] else meta['size']

            partials['c_trust', name] = -2.*dist_flat[i:i + size]*dist
            i += size
        # import pdb; pdb.set_trace()


    def set_center(self, zk):

        """
        Set the center of the trust ball. Should be placed before calling run_driver

        zk : dict
            location to center the trust radius about, in driver dv dict form
        """
        self.location = zk
    
    def set_radius(self, rad):
        """
        Set the radius/bound of the trust ball

        rad : float
            trust radius (all directions for now). If negative, we treat as a 
            flag to return 
        """

        self.radius = rad
        