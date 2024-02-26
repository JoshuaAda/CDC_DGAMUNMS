
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

"""Simulator with uncertain parameter p"""
def template_simulator(model):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)


    simulator.set_param(t_step = 1)

    # uncertainties
    #p_val = simulator.get_p_template()

    #def p_fun(t_now):
    #    return p_val
    #simulator.set_p_fun(p_fun)

    # uncertain parameters
    p_num = simulator.get_p_template()
    p_num["x_sp"] = np.array([[5],[1]])
    p_num['p'] = np.array([[0.25]])
    def p_fun(t_now):
        return p_num
    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator
