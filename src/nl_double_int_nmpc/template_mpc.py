
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

def template_mpc(model, silence_solver = False):
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

    #max_x = np.array([[10.0], [10.0]])

    mpc.bounds['lower','_x','x'] = np.array([[0.0], [0.0]])# np.array([[-4.49345e-09], [-4.62047e-09]])#np.array([[0.0], [0.0]])
    mpc.bounds['upper','_x','x'] = np.array([[10.0], [10.0]]) #np.array([[10], [6.43288]])#np.array([[10.0], [10.0]])

    mpc.bounds['lower','_u','u'] = -np.array([[10.0], [5.0]])
    mpc.bounds['upper','_u','u'] =  np.array([[10.0], [5.0]])

    # uncertainties
    #tvp_val = mpc.get_tvp_template()
    #tvp_val['_tvp',:] = np.array([0.25])
    #def tvp_fun(t_now):
    #    return tvp_val
    #mpc.set_tvp_fun(tvp_fun)

    # parameters
    #x_sp_val = np.array([[5],[1],[0.25]])
    p_val = mpc.get_p_template(1)
    p_val['_p'] = np.array([0.15,5,2])#x_sp_val
    #p_val['_p',1] = np.array([0.3,5,2])#np.array([0.25,0.2,0.3])
    def p_fun(n):
        return p_val
    mpc.set_p_fun(p_fun)
    #mpc.set_uncertainty_values(p = p)
    #mpc.prepare_nlp()
    #G=#DM.ones((4,2))
    #for k in range(10):
    #    for m in range(2):
    #        extra_cons = G@mpc.opt_x['_x',k][m][0]-DM.ones((2,1))
    #        mpc.nlp_cons.append(
    #            extra_cons
    #        )
    #mtx = np.zeros(extra_cons.shape)
    #mtx.fill(-inf)
    #mpc.nlp_cons_lb.append(mtx)
    #mpc.nlp_cons_ub.append(np.zeros(extra_cons.shape))
    #mpc.create_nlp()
    mpc.setup()

    return mpc
