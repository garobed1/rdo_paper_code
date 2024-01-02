import openmdao.api as om
import os, copy
from functions.problem_picker import GetProblem
from utils.sutils import convert_to_smt_grads
import numpy as np


class SMTComponent(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('prob', desc="smt problem")
        self.options.declare('dim_base', default=[], desc="base dimension")
        self.options.declare('prob_opts', default=[], desc="smt problem options")
        self.options.declare("sub_index", None, desc="indices of dimensions to connect")

        self.func = None
        
    
    def setup(self):

        # get function
        self.func = GetProblem(self.options['prob'], self.options['dim_base'],
                               *self.options['prob_opts'])

        # get sub indices
        self.dim_t = self.func.ndim
        self.sub_ind = np.arange(0, self.dim_t).tolist()
        self.fix_ind = []
        self.fix_val = []
        if self.options['sub_index'] is not None:
            self.sub_ind = self.options['sub_index']
            self.fix_ind = [x for x in np.arange(0, self.dim_t) if x not in self.sub_ind]
            self.fix_val = [0.]*len(self.fix_ind)
        self.dim = len(self.sub_ind)

        # inputs
        self.add_input('x', shape=self.dim,
                              desc='Current design point')
        
        self.add_output('y', shape=1,
                                   desc='smt function output')

        self.declare_partials('*','*')

    def set_static(self, x):
        len_given = x.shape[0]
        dim_t = len(self.sub_ind) + len(self.fix_ind)
        dim_r = len(self.fix_ind)
        if len_given == dim_t:
            self.fix_val = x[self.fix_ind]
        elif len_given == dim_r:
            self.fix_val = x
        else:
            ValueError(f'Invalid number of inputs given ({len_given} != total dim {dim_t}, {len_given} != reduced dim {dim_r})')

        return 
    
    def compute(self, inputs, outputs):

        x_sub = inputs['x']

        x_full = np.zeros(self.dim_t)

        c = 0
        for i in self.sub_ind:
            x_full[i] = x_sub[c]
            c += 1

        c = 0
        for i in self.fix_ind:
            x_full[i] = self.fix_val[c]
            c += 1


        y = self.func(x_full)

        outputs['y'] = y


        
    def compute_partials(self, inputs, partials):

        x_sub = inputs['x']

        x_full = np.zeros(self.dim_t)

        c = 0
        for i in self.sub_ind:
            x_full[i] = x_sub[c]
            c += 1

        c = 0
        for i in self.fix_ind:
            x_full[i] = self.fix_val[c]
            c += 1


        dy = convert_to_smt_grads(self.func, x_full)

        partials['y','x'] = dy[self.sub_ind]


        