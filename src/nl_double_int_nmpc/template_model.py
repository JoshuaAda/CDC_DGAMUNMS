"""Info:
https://doi.org/10.1016/j.sysconle.2007.06.013 Lazar et al. 2008
"""


import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


"""Nonlinear double integrator model"""
def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(2,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(2,1))
    p = model.set_variable(var_type='_p', var_name='p', shape=(1, 1))
    x_sp = model.set_variable(var_type='_p', var_name='x_sp', shape=(2, 1))
    
    A = np.array([[ 1,  1],
                  [ 0,  1]])

    B = np.array([[1,0],
                  [ 0,1 ]])

    F = vertcat(p,p)

    Q = np.array([[1, 0],
                  [0, 1]])

    R = np.array([[0, 0],
                  [0, 0]])

    # x_sp = np.array([[2],[2]])
    stage_cost = (_x-x_sp).T@Q@(_x-x_sp) + _u.T@R@_u
    terminal_cost = (_x-x_sp).T@Q@(_x-x_sp)

    # stage_cost = _x[0] + _u.T@R@_u
    # terminal_cost = _x[0]
    # stage_cost = _x[0]+_x[1] + _u.T@R@_u
    # terminal_cost = _x[0]+_x[1]

    # stage_cost = _x.T@Q@_x + _u.T@R@_u
    # terminal_cost = _x.T@Q@_x

    model.set_expression(expr_name='stage_cost', expr=stage_cost)
    model.set_expression(expr_name='terminal_cost', expr=terminal_cost)

    x_next = A@_x + B@_u + F@sqrt(_x.T@_x)
    model.set_rhs('x', x_next)

    model.setup()

    return model
