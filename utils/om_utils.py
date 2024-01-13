
import numpy as np

from collections import OrderedDict
from utils.sutils import print_mpi
#TODO: Function to convert om dicts to smt ordering

# etc.



def get_om_dict(xnew, smt_map):


    # smt_map ('name', inds (int or np.ndarray))

    # from onerahub, om to smt bounds
    om_dvs = []

    for name, inds in smt_map.items():
       
        size = inds.shape[0]

        vals = np.zeros(size)

        for i in range(inds.shape[0]):
            vals[i] = xnew[inds[i]]
        
        om_dvs.append(name, vals)

    return om_dvs

def map_om_to_smt(self, dvs):
    """
    Map OM design var dict to a particular SMT ordering

    Do this once and pass the map as metadata to SMT-based components and 
    drivers

    Parameters
    ----------
    dvs : OrderedDict
        openmdao design variables

    Returns
    -------
    smt_map : dict
        indices for each design variable in a single vector
        
    xlimits : list
        list of ordered bounds
    """

    #dvs = self._designvars

    #{name:index}
    # establish this here

    smt_map = OrderedDict([(name, np.zeros(_get_size(meta), dtype=int))
                               for name, meta in dvs.items()])

    # from onerahub, om to smt bounds
    xlimits = []

    count = 0
    for name, meta in dvs.items():

        size = meta["size"]
        meta_low = meta["lower"]
        meta_high = meta["upper"]

        for j in range(size):
            if isinstance(meta_low, np.ndarray):
                p_low = meta_low[j]
            else:
                p_low = meta_low

            if isinstance(meta_high, np.ndarray):
                p_high = meta_high[j]
            else:
                p_high = meta_high

            xlimits.append((p_low, p_high))

            smt_map[name][j] = count
            count += 1

            

    return smt_map, xlimits


def get_om_design_size(dv_settings):
    ndesvar = 0
    for name, meta in dv_settings.items():
        size = meta['global_size'] if meta['distributed'] else meta['size']
        ndesvar += size
    return ndesvar

# Assume items in OM DV dict originate from OrderedDict
def om_dict_to_flat_array(dct, dv_settings, ndesvar):
    desvar_array = np.empty(ndesvar)
    i = 0
    for name, meta in dv_settings.items():
        size = meta['global_size'] if meta['distributed'] else meta['size']
        desvar_array[i:i + size] = dct[name]
        i += size
    return desvar_array

def _get_size(dct):
    # Returns global size of the variable if it is distributed, size otherwise.
    return dct['global_size'] if dct['distributed'] else dct['size']




"""
Tucker's code. Modify it to do what I need it to do

"""

def l1_merit_function2(driver, penalty, feas_tol):
    obj_vals = driver.get_objective_values()

    for obj in obj_vals:
        phi = np.copy(obj_vals[obj])
        break

    con_vals = driver.get_constraint_values()
    con_violation = constraint_violation2(
        driver, driver._cons, con_vals, feas_tol)

    print_mpi(f"merit con_violation: {con_violation}")
    for error in con_violation.values():
        print_mpi(f"merit error: {error}")
        print_mpi(f"abs error: {np.absolute(error)}")
        print_mpi(f"sum abs error: {np.sum(np.absolute(error))}")
        print_mpi(f"phi: {phi}")
        print_mpi(f"penalty: {penalty}")
        phi += penalty * np.sum(np.absolute(error))

    return phi


def get_active_constraints2(cons, con_vals, dvs, dv_vals, feas_tol, no_trust=False):
    active_cons = {}
    for con in cons.keys():
        con_name = cons[con]['name']
        con_val = con_vals[con]
        if cons[con]['equals'] is not None:
            # print_mpi(f"constraint {con} is equality!")
            active_cons[con_name] = np.ones_like(con_val, dtype=bool)
        else:
            con_ub = cons[con].get("upper", np.inf)
            con_lb = cons[con].get("lower", -np.inf)
            # print_mpi(f"{con} lower bound: {con_lb}, upper bound: {con_ub}, value: {con_vals[con]}")

            # Initially assume all cons are inactive
            active_con = np.zeros_like(con_val, dtype=bool)

            # find indices of constraint that violate the upper bound
            active_con_ind = np.asarray(con_val > con_ub).nonzero()
            active_con[active_con_ind] = True

            # find indices of constraint that are on the upper bound
            active_con_ind = np.asarray(np.isclose(
                con_val, con_ub, atol=feas_tol, rtol=feas_tol)).nonzero()
            active_con[active_con_ind] = True

            # find indices of constraint that violate the upper bound
            active_con_ind = np.asarray(con_val < con_lb).nonzero()
            active_con[active_con_ind] = True

            # find indices of constraint that are on the upper bound
            active_con_ind = np.asarray(np.isclose(
                con_val, con_lb, atol=feas_tol, rtol=feas_tol)).nonzero()
            active_con[active_con_ind] = True

            active_cons[con_name] = active_con

            # if con_val > con_ub or np.isclose(con_val, con_ub, atol=feas_tol, rtol=feas_tol):
            #     active_cons[con_name] = True
            # elif con_val < con_lb or np.isclose(con_val, con_lb, atol=feas_tol, rtol=feas_tol):
            #     active_cons[con_name] = True

    if not no_trust:
        for dv in dvs.keys():
            dv_val = dv_vals[dv]
            dv_ub = dvs[dv].get("upper", np.inf)
            dv_lb = dvs[dv].get("lower", -np.inf)

            # Initially assume all dv bounds are inactive
            active_dv = np.zeros_like(dv_val, dtype=bool)

            # find indices of dv that violate the upper bound
            active_dv_ind = np.asarray(dv_val > dv_ub).nonzero()
            active_dv[active_dv_ind] = True

            # find indices of dv that are on the upper bound
            active_dv_ind = np.asarray(np.isclose(
                dv_val, dv_ub, atol=feas_tol, rtol=feas_tol)).nonzero()
            active_dv[active_dv_ind] = True

            # find indices of dv that violate the lower bound
            active_dv_ind = np.asarray(dv_val < dv_lb).nonzero()
            active_dv[active_dv_ind] = True

            # find indices of dv that are on the lower bound
            active_dv_ind = np.asarray(np.isclose(
                dv_val, dv_lb, atol=feas_tol, rtol=feas_tol)).nonzero()
            active_dv[active_dv_ind] = True

            active_cons[dv] = active_dv

        # if dv_val > dv_ub or np.isclose(dv_val, dv_ub, atol=feas_tol, rtol=feas_tol):
        #     active_cons[dv] = True
        # elif dv_val < dv_lb or np.isclose(dv_val, dv_lb, atol=feas_tol, rtol=feas_tol):
        #     active_cons[dv] = True

    return active_cons


def constraint_violation2(driver, cons, con_vals, feas_tol):
    con_violation = {}

    for con in cons.keys():
        con_val = con_vals[con]
        if cons[con]['equals'] is not None:
            con_target = cons[con]["equals"]
            print_mpi(f"{con} target: {con_target}, value: {con_vals[con]}")
            if not np.isclose(con_val, con_target, atol=feas_tol, rtol=feas_tol):
                # print_mpi(f"violates equality constraint!")
                con_violation[con] = con_val - con_target
        else:
            con_ub = cons[con].get("upper", np.inf)
            con_lb = cons[con].get("lower", -np.inf)
            print_mpi(
                f"{con} lower bound: {con_lb}, upper bound: {con_ub}, value: {con_vals[con]}")

            # find indices of constraint that violate the upper bound
            con_viol_ind = np.asarray(con_val > con_ub).nonzero()
            # reduce that set to those values that are not on the constraint bound within the tolerance
            con_viol_ind = np.asarray(np.invert(np.isclose(
                con_val[con_viol_ind], con_ub, atol=feas_tol, rtol=feas_tol))).nonzero()

            # set the constraint violation
            con_viol = np.zeros_like(con_val)
            con_viol[con_viol_ind] = con_val[con_viol_ind] - con_ub

            # find indices of constraint that violate the lower bound
            con_viol_ind = np.asarray(con_val < con_lb).nonzero()
            # reduce that set to those values that are not on the constraint bound within the tolerance
            con_viol_ind = np.asarray(np.invert(np.isclose(
                con_val[con_viol_ind], con_lb, atol=feas_tol, rtol=feas_tol))).nonzero()

            # set the constraint violation
            con_viol[con_viol_ind] = con_val[con_viol_ind] - con_lb

            con_violation[con] = con_viol
            # if con_val > con_ub:
            #     if not np.isclose(con_val, con_ub, atol=feas_tol, rtol=feas_tol):
            #         # print_mpi(f"violates upper bound!")
            #         con_violation[con] = con_val - con_ub
            # elif con_val < con_lb:
            #     if not np.isclose(con_val, con_lb, atol=feas_tol, rtol=feas_tol):
            #         # print_mpi(f"violates lower bound!")
            #         con_violation[con] = con_val - con_lb
    return con_violation


def estimate_lagrange_multipliers2(obj, active_cons, dvs, totals):
    duals = {}

    n = 0
    for metadata in dvs.values():
        n += metadata['global_size']

    obj_grad = np.zeros(n)
    offset = 0
    for dv in dvs.keys():
        dv_size = dvs[dv]['global_size']
        obj_grad[offset:offset +
                 dv_size] = np.reshape(totals[obj, dv], dv_size)
        offset += dv_size

    print_mpi(f"dv size: {n}")
    n_con = 0
    for con, active in active_cons.items():
        n_con += np.count_nonzero(active)

    active_cons_jac = np.zeros((n_con, n))
    i = 0
    for con, active in active_cons.items():
        for active_con in active:
            if not active_con:
                continue

            if con in dvs.keys():
                con_grad = {dv: np.array([1.0]) if dv == con else np.array(
                    [0.0]) for dv in dvs.keys()}
            else:
                con_grad = {dv: totals[con, dv] for dv in dvs.keys()}
            print_mpi(f"{con} grad: {con_grad}")
            offset = 0
            for dv in dvs.keys():
                dv_size = dvs[dv]['global_size']
                print_mpi(
                    f"dest shape: {active_cons_jac[i, offset:offset + dv_size].shape}")
                print_mpi(f"target shape: {con_grad[dv].shape}")
                active_cons_jac[i, offset:offset + dv_size] = con_grad[dv]
                offset += dv_size
            i += 1

    duals_vec, optimality, a, b = np.linalg.lstsq(
        active_cons_jac.T, obj_grad, rcond=None)
    print_mpi(duals_vec)

    i = 0
    for con, active in active_cons.items():
        if i < n_con:
            duals[con] = np.zeros(active.shape)
            for con_i, active_con in enumerate(active):
                if not active_con:
                    continue

                duals[con][con_i] = duals_vec[i]
                i += 1

    print_mpi(f"estimated multipliers: {duals}")
    return duals


def optimality2(responses, obj, active_cons, dvs, duals, totals):
    n = 0
    for metadata in dvs.values():
        n += metadata['global_size']

    obj_grad = np.zeros(n)
    offset = 0
    for dv in dvs.keys():
        dv_size = dvs[dv]['size']
        obj_grad[offset:offset + dv_size] = totals[obj, dv]
        offset += dv_size

    n_con = 0
    for con, active in active_cons.items():
        n_con += np.count_nonzero(active)
    active_cons_jac = np.zeros((n_con, n))

    print_mpi(f"opt n_con: {n_con}")
    dual_vec = np.zeros(n_con)
    dual_offset = 0

    response_map = {}
    for response, meta in responses.items():
        response_map[meta['name']] = response

    # for i, con in enumerate(active_cons):
    #     if con in dvs.keys():
    #         con_grad = {dv: np.array([1.0]) if dv == con else np.array(
    #             [0.0]) for dv in dvs.keys()}
    #     else:

    #         con_grad = {dv: totals[response_map[con], dv] for dv in dvs.keys()}
    #     offset = 0
    #     for dv in dvs.keys():
    #         dv_size = dvs[dv]['size']
    #         active_cons_jac[offset:offset + dv_size, i] = con_grad[dv]
    #         offset += dv_size

    #     dual_size = duals[con].size
    #     dual_vec[dual_offset:dual_offset + dual_size] = duals[con]
    #     dual_offset += dual_size

    i = 0
    for con, active in active_cons.items():
        for active_con in active:
            if not active_con:
                continue

            if con in dvs.keys():
                con_grad = {dv: np.array([1.0]) if dv == con else np.array(
                    [0.0]) for dv in dvs.keys()}
            else:
                con_grad = {dv: totals[response_map[con], dv]
                            for dv in dvs.keys()}
            print_mpi(f"{con} grad: {con_grad}")
            offset = 0
            for dv in dvs.keys():
                dv_size = dvs[dv]['global_size']
                print_mpi(
                    f"dest shape: {active_cons_jac[i, offset:offset + dv_size].shape}")
                print_mpi(f"target shape: {con_grad[dv].shape}")
                active_cons_jac[i, offset:offset + dv_size] = con_grad[dv]
                offset += dv_size
            i += 1

    i = 0
    for con, active in active_cons.items():
        if i < n_con:
            for con_i, active_con in enumerate(active):
                if not active_con:
                    continue

            dual_vec[i] = duals[con][con_i]
            i += 1

    full_vec = obj_grad - dual_vec @ active_cons_jac
    optimality = np.linalg.norm(full_vec, np.inf)

    # if optimality > 1e-7:
    # import pdb; pdb.set_trace()
    return optimality, full_vec


def get_active_constraints(prob, constraints, des_vars, feas_tol=1e-6):
    active_cons = list()
    for constraint in constraints.keys():
        constraint_value = prob[constraint]
        if constraints[constraint]['equals'] is not None:
            # print_mpi(f"constraint {constraint} is equality!")
            active_cons.append(constraint)
        else:
            constraint_upper = constraints[constraint].get("upper", np.inf)
            constraint_lower = constraints[constraint].get("lower", -np.inf)
            if constraint_value > constraint_upper or np.isclose(constraint_value, constraint_upper, atol=feas_tol, rtol=feas_tol):
                active_cons.append(constraint)
            elif constraint_value < constraint_lower or np.isclose(constraint_value, constraint_lower, atol=feas_tol, rtol=feas_tol):
                active_cons.append(constraint)

    for des_var in des_vars.keys():
        des_var_scaler = des_vars[des_var]['scaler'] or 1
        des_var_adder = des_vars[des_var]['adder'] or 0
        des_var_value = prob[des_var]
        des_var_upper = des_vars[des_var].get(
            "upper", np.inf) / des_var_scaler - des_var_adder
        des_var_lower = des_vars[des_var].get(
            "lower", -np.inf) / des_var_scaler - des_var_adder
        # print_mpi(f"{des_var} lower bound: {des_var_lower}, upper bound: {des_var_upper}, value: {des_var_value}")
        if des_var_value > (des_var_upper / des_var_scaler - des_var_adder) or np.isclose(des_var_value, des_var_upper, atol=feas_tol, rtol=feas_tol):
            # print_mpi(f"upper bound active!")
            active_cons.append(des_var)
        elif des_var_value < des_var_lower or np.isclose(des_var_value, des_var_lower, atol=feas_tol, rtol=feas_tol):
            # print_mpi(f"lower bound active!")
            active_cons.append(des_var)

    return active_cons


def constraint_violation(prob, constraints, feas_tol=1e-6):
    constraint_error = dict()
    scaled_constraint_error = dict()
    for constraint in constraints.keys():
        constraint_value = prob[constraint]
        if constraints[constraint]['equals'] is not None:
            constraint_target = constraints[constraint]["equals"]
            if not np.isclose(constraint_value, constraint_target, atol=feas_tol, rtol=feas_tol):
                constraint_error[constraint] = constraint_value - \
                    constraint_target
                scaled_constraint_error[constraint] = (
                    constraint_value - constraint_target) / constraint_target
        else:
            constraint_value = prob[constraint]
            constraint_upper = constraints[constraint].get("upper", np.inf)
            constraint_lower = constraints[constraint].get("lower", -np.inf)
            # print_mpi(f"{constraint} lower bound: {constraint_lower}, upper bound: {constraint_upper}, value: {constraint_value}")
            if constraint_value > constraint_upper:
                if not np.isclose(constraint_value, constraint_upper, atol=feas_tol, rtol=feas_tol):
                    # print_mpi(f"violates upper bound!")
                    constraint_error[constraint] = constraint_value - \
                        constraint_upper
                    scaled_constraint_error[constraint] = (
                        constraint_value - constraint_upper) / constraint_upper
            elif constraint_value < constraint_lower:
                if not np.isclose(constraint_value, constraint_lower, atol=feas_tol, rtol=feas_tol):
                    # print_mpi(f"violates lower bound!")
                    constraint_error[constraint] = constraint_value - \
                        constraint_lower
                    scaled_constraint_error[constraint] = (
                        constraint_value - constraint_lower) / constraint_lower
    return constraint_error, scaled_constraint_error


def l1_merit_function(prob, objective, constraints, penalty_parameter, feas_tol=1e-6, maximize=False):
    phi = np.copy(prob[objective])

    if maximize:
        phi *= -1.0

    constraint_error, scaled_constraint_error = constraint_violation(
        prob, constraints, feas_tol)
    # for error in constraint_error.values():
    #     phi += penalty_parameter * np.absolute(error)
    for error in scaled_constraint_error.values():
        phi += penalty_parameter * np.absolute(error)

    return phi


def optimality(totals, objective, active_constraints, des_vars, multipliers, maximize=False):
    grad_f = {input: totals[objective, input] for input in des_vars.keys()}

    n = 0
    for input in grad_f.keys():
        n += grad_f[input].size

    grad_f_vec = np.zeros((n))
    offset = 0
    for input in grad_f.keys():
        input_size = grad_f[input].size
        grad_f_vec[offset:offset + input_size] = grad_f[input]
        offset += input_size

    if maximize:
        grad_f_vec *= -1.0

    n_con = len(active_constraints)
    active_cons_mat = np.zeros((n, n_con))
    multipliers_vec = np.zeros((n_con))
    multiplier_offset = 0
    for i, constraint in enumerate(active_constraints):
        if constraint in des_vars.keys():
            constraint_grad = {input: np.array([1.0]) if input == constraint else np.array([
                0.0]) for input in des_vars.keys()}
        else:
            constraint_grad = {
                input: totals[constraint, input] for input in des_vars.keys()}
        # print_mpi(f"{constraint} grad: {constraint_grad}")
        offset = 0
        for input in constraint_grad.keys():
            input_size = constraint_grad[input].size
            active_cons_mat[offset:offset +
                            input_size, i] = constraint_grad[input]
            offset += input_size

        multiplier_size = multipliers[constraint].size
        multipliers_vec[multiplier_offset:multiplier_offset +
                        multiplier_size] = multipliers[constraint]
        multiplier_offset += multiplier_size

    # multipliers_vec = np.zeros((n_con))
    # offset = 0
    # for input in multipliers.keys():
    #     input_size = multipliers[input].size
    #     multipliers_vec[offset:offset + input_size] = multipliers[input]
    #     offset += input_size

    full_vec = grad_f_vec - active_cons_mat @ multipliers_vec
    optimality = np.linalg.norm(
        full_vec, np.inf)
    return optimality, full_vec


def unscale_lagrange_multipliers(prob, objective, active_constraints, multipliers):
    for response in prob.driver._responses.values():
        if response['name'] == objective:
            obj_ref = response['ref']
            obj_ref0 = response['ref0']

    for constraint in active_constraints:
        if constraint in prob.driver._designvars:
            ref = prob.driver._designvars[constraint]['ref']
            ref0 = prob.driver._designvars[constraint]['ref0']
        else:
            for response in prob.driver._responses.values():
                if response['name'] == constraint:
                    ref = response['ref']
                    ref0 = response['ref0']

        if obj_ref is None:
            obj_ref = 1.0
        if obj_ref0 is None:
            obj_ref0 = 0.0

        if ref is None:
            ref = 1.0
        if ref0 is None:
            ref0 = 0.0

        multipliers[constraint] = multipliers[constraint] * \
            (obj_ref - obj_ref0) / (ref - ref0)

    return multipliers


def estimate_lagrange_multipliers(prob, objective, active_constraints, totals, des_vars, unscaled=False):
    multipliers = dict()

    grad_f = {input: totals[objective, input] for input in des_vars.keys()}

    n = 0
    for input in grad_f.keys():
        n += grad_f[input].size

    grad_f_vec = np.zeros((n))
    offset = 0
    for input in grad_f.keys():
        input_size = grad_f[input].size
        grad_f_vec[offset:offset + input_size] = grad_f[input]
        offset += input_size

    n_con = len(active_constraints)
    active_cons_mat = np.zeros((n, n_con))
    for i, constraint in enumerate(active_constraints):
        if constraint in des_vars.keys():
            constraint_grad = {input: np.array([1.0]) if input == constraint else np.array([
                0.0]) for input in des_vars.keys()}
        else:
            constraint_grad = {
                input: totals[constraint, input] for input in des_vars.keys()}
        # print_mpi(f"{constraint} grad: {constraint_grad}")
        offset = 0
        for input in constraint_grad.keys():
            input_size = constraint_grad[input].size
            active_cons_mat[offset:offset +
                            input_size, i] = constraint_grad[input]
            offset += input_size

    # print_mpi(f"grad_f_vec: {grad_f_vec}")
    # print_mpi(f"active_cons_mat: {active_cons_mat}")
    multipliers_vec, optimality, a, b = np.linalg.lstsq(
        active_cons_mat, grad_f_vec, rcond=None)
    # multipliers_vec, optimality, a, b = np.linalg.lstsq(active_cons_mat, grad_f_vec, rcond=-1)
    # print_mpi(f"lstsq output: {a}, {b}")
    # print_mpi(f"multipliers vec: {multipliers_vec}")
    # print_mpi(f"optimality: {np.linalg.norm(grad_f_vec - active_cons_mat @ multipliers_vec)}")
    # print_mpi(f"Estimated optimality squared: {optimality}")
    # print_mpi(f"Estimated optimality: {np.sqrt(optimality)}")
    offset = 0
    for constraint in active_constraints:
        constraint_size = 1
        multipliers[constraint] = multipliers_vec[offset:offset + constraint_size]
        offset += constraint_size

    if unscaled:
        unscale_lagrange_multipliers(
            prob, objective, active_constraints, multipliers)

    return multipliers


    
    
# """
# Evaluate/Predict the Lagrangian gradient at the current design, also return 
# optimality and feasability if applicable
# If no constraints are present, this is just the objective gradient and its
# norm. If constraints are present, we employ techniques to estimate the
# lagrange multipliers \pi and compute the constraint jacobian as well (A)
# \Nabla\mathcal{L} = \Nabla f + A^T \pi
# Whether the robust quantity is an objective or constraint, the uncertainty
# will propagate to this result

# Inputs:
#     problem : openmdao Problem for which to evaluate the Lagrangian. The 
#         constraint method will only work properly for the "low-fidelity"
#         subproblem model, however
#    have_cons :
#    feas_tol : 
#    opt_tol : if the original optimization was solved to this tolerance, solve the
#        lagrangian multiplier lstsq problem to this tolerance as well
#    duals_given : if values for duals are given, do not estimate them
# Outputs:
#     grad : ndarray Lagrangian Gradient
#     opt : float Optimality
#     feas : float Feasability
# """
def grad_opt_feas(problem, have_cons, feas_tol, opt_tol=1e-16, duals_given = None, no_trust=False):

    # check if there are constraints first
    if not have_cons:
        grad = problem.compute_totals(return_format='array')
        opt = np.linalg.norm(grad, np.inf)
        feas = 0.0
        duals = []
        return grad, opt, feas, duals
    
    problem.run_model()
    driver = problem.driver
    model = problem.model
    dvs = driver._designvars
    dv_vals = driver.get_design_var_values()
    responses = driver._responses
    cons = driver._cons
    con_vals = driver.get_constraint_values()
    
    # gather constraint and objective info
    for response, meta in responses.items():
        if meta['type'] == 'obj':
            obj = response
        # if meta['type'] == 'con':

    # active constraints
    active_cons = get_active_constraints2(cons, con_vals, dvs, dv_vals, feas_tol, no_trust=no_trust)
    
    # total derivatives
    totals = problem.compute_totals([*responses.keys()],
                                           [*dvs.keys()],
                                           driver_scaling=False)

    # lagrange multipliers of active constraints
    # only do this at the subproblem optima, don't do it while we're adding points to the surrogate
    # however, do recompute if the constraint becomes active
    if duals_given is None:
        duals = estimate_lagrange_multipliers2(obj, active_cons, dvs, totals)
    elif set(duals_given.keys()) != set(active_cons.keys()):
        duals = estimate_lagrange_multipliers2(obj, active_cons, dvs, totals)
    else:
        duals = duals_given
    # optimality
    opt, grad = optimality2(responses, obj, active_cons, dvs, duals, totals)

    feas = 0.0
    if len(duals) > 0:
        d_list = [value for values in duals.values() for value in values]
        penalty = 2.*abs(max(d_list, key=abs))
        feas = l1_merit_function2(driver, penalty, feas_tol)

    #NOTE: replace totals with lagrangian grad?
    # import pdb; pdb.set_trace()
    return grad, opt, feas, duals
