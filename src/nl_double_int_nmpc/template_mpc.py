
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

"""Nominal MPC"""

def template_mpc(model, silence_solver = True):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    mpc.settings.n_robust = 0
    mpc.settings.n_horizon = 10
    mpc.settings.t_step = 1
    mpc.settings.store_full_solution =True

    if silence_solver:
        mpc.settings.supress_ipopt_output()


    mterm = model.aux['terminal_cost']
    lterm = model.aux['stage_cost'] # terminal cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1)


    mpc.bounds['lower','_x','x'] = np.array([[0.0], [0.0]])
    mpc.bounds['upper','_x','x'] = np.array([[10.0], [10.0]])

    mpc.bounds['lower','_u','u'] = -np.array([[10.0], [5.0]])
    mpc.bounds['upper','_u','u'] =  np.array([[10.0], [5.0]])


    # parameters
    p_val = mpc.get_p_template(1)
    p_val['_p'] = np.array([0.15,5,2])#x_sp_val
    def p_fun(n):
        return p_val
    mpc.set_p_fun(p_fun)

    mpc.setup()

    return mpc
