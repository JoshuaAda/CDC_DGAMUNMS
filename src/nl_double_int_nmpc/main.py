import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
from approx_MPC import ApproxMPC, ApproxMPCSettings
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
import torch
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
import time
from Robust_MPC import RobustMPC


"""Functions to simulate and find recatngles for the safety filter"""
def f(x,u,p):
    A = np.array([[ 1,  1],
                  [ 0,  1]])

    B = np.array([[1,0],
                  [ 0,1 ]])

    F = np.array([[p],[p]])

    return A@x + B@u + F*sqrt(x.T@x)

u_s=np.load("../u_new.npy")
x_min=np.load("../x_min_new.npy")
x_max=np.load("../x_max_new.npy")
lbx_rect = [x_min[k,0,:] for k in range(len(x_min))]#np.array([[-4.49345e-09], [-4.62047e-09]])#np.array(#np.array([[-3.39036e-09], [-4.46839e-09]])#np.array([[2.38303e-10], [-2.73719e-09]])
ubx_rect = [x_max[k,-1,:] for k in range(len(x_max))]#np.array([[10], [6.43288]]) #np.array([[7.77376], [5.29665]])#np.array([[8.578], [5.02841]])
# SETUP
def in_rectangle(x, lb,ub):
    return np.logical_and(np.logical_and(lb[0] <= x[0], x[0] <= ub[0]), np.logical_and(lb[1] <= x[1], x[1] <= ub[1]))
def find_save_input(x):
    u0=np.array([[0],[0]])
    for k in range(len(u_s)):
        if in_rectangle(x,x_min[k,0],x_max[k,-1]):
            for m in range(len(u_s[k])):
                if in_rectangle(x,x_min[k,m],x_max[k,m]):
                    u0=u_s[k][m]
                    break
    return u0
matplotlib.use('TkAgg')


""" Definition of MPC/RobustMPC/Approximate MPC"""
model = template_model()
mpc = template_mpc(model)
robust_mpc=RobustMPC()
n_layers = 6 #(L = n+1)
n_neurons = 200
n_in = 4
n_out = 2
settings=ApproxMPCSettings(n_in=n_in,n_out=n_out,n_layers=n_layers,n_neurons=n_neurons)
mpc_app=ApproxMPC(settings)
mpc_app.load_state_dict("../approx_mpc_models/run_20")
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

# INFO
lbx, ubx = np.array(mpc.bounds['lower','_x','x']), np.array(mpc.bounds['upper','_x','x'])
lub, ubu = np.array(mpc.bounds['lower','_u','u']), np.array(mpc.bounds['upper','_u','u'])
count=0
count_viol=0
times=[]
stage_costs=[]
M=1000
"""1000 uncertainty realizations"""
for k in range(M):
# CONFIG
    simulator = template_simulator(model)
    np.random.seed(k)
    e = np.ones([model.n_x,1])
    in_set=False
    while not in_set:
        x0 = np.random.uniform(np.array([[0],[0]]),np.array([[10],[10]]))#np.array([[8],[7]])#np.random.uniform(2*e,8*e)
        in_set=any([in_rectangle(x0,lbx_rect[k],ubx_rect[k]) for k in range(len(lbx_rect))])
    u0 = np.array([0.0,0.0]).reshape(2,1)


    N_sim = 20
    p_template = simulator.get_p_template()
    def p_fun(n):
        p_template['p'] = np.random.uniform(0.0,0.3)
        p_template['x_sp']=np.array([[5],[2]])
        return p_template
    simulator.set_p_fun(p_fun)
    mpc.u0 = u0
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()
    print(k)

    # MAIN LOOP


    robust_mpc.u0=u0
    """Simulation of 10 steps"""
    for k in range(N_sim):
        x=torch.Tensor(np.array([x0[0],x0[1],u0[0],u0[1]])).squeeze()
        s = time.time()

        """ Controller evaluation and forward prediction"""
        u0 = mpc_app.make_step(x,clip_outputs=False).reshape((2,1))##robust_mpc.make_step(x0)#robust_mpc.make_step(x0)##mpc.make_step(x0)#
        r = time.time()
        times.append(r-s)
        x0_1=f(x0,u0,0.0)
        x0_2=f(x0,u0,0.3)

        """Safety filter/ if set to Flase evaluation without safety filter"""
        Cond=[not in_rectangle(x0_2,lbx_rect[k],ubx_rect[k]) or not in_rectangle(x0_1,lbx_rect[k],ubx_rect[k]) for k in range(len(lbx_rect))]
        if all(Cond):#False:#not in_rectangle(x0_2,lbx_rect,ubx_rect) or not in_rectangle(x0_1,lbx_rect,ubx_rect):
            count=count+1
            simulator.x0=x0
            estimator.x0=x0
            u0=find_save_input(x0)
            mpc.u0=u0
        y_next = simulator.make_step(u0)
        x0_new = estimator.make_step(y_next)
        x0=x0_new
        if not in_rectangle(x0,lbx,ubx):
            count_viol+=1
    stage_costs.append(sum(simulator.data['_aux', 'stage_cost']))
myu=sum(times)/len(times)
print(myu)
print(count)
print(count_viol)
print(sum(stage_costs)/len(stage_costs))
#print(sum(times)/len(times))

fig1, axs_1 = plt.subplots(2,1)
axs_1[0].plot(simulator.data['_x','x',0])
axs_1[0].hlines(lbx[0],0,N_sim,colors='r',linestyles='dashed')
axs_1[0].hlines(ubx[0],0,N_sim,colors='r',linestyles='dashed')
axs_1[0].set_ylabel('x0')
axs_1[0].set_xlabel('time')

axs_1[1].plot(simulator.data['_x','x',1])
axs_1[1].hlines(lbx[1],0,N_sim,colors='r',linestyles='dashed')
axs_1[1].hlines(ubx[1],0,N_sim,colors='r',linestyles='dashed')
axs_1[1].set_ylabel('x1')
axs_1[1].set_xlabel('time')

fig2, axs_2 = plt.subplots(2,1)
axs_2[0].plot(simulator.data['_u','u',0])
axs_2[0].hlines(lub[0],0,N_sim,colors='r',linestyles='dashed')
axs_2[0].hlines(ubu[0],0,N_sim,colors='r',linestyles='dashed')
axs_2[0].set_ylabel('u0')
axs_2[0].set_xlabel('time')

axs_2[1].plot(simulator.data['_u','u',1])
axs_2[1].hlines(lub[1],0,N_sim,colors='r',linestyles='dashed')
axs_2[1].hlines(ubu[1],0,N_sim,colors='r',linestyles='dashed')
axs_2[1].set_ylabel('u1')
axs_2[1].set_xlabel('time')

fig3, axs_3 = plt.subplots(1,1)
axs_3.plot(simulator.data['_x','x',0],simulator.data['_x','x',1])
axs_3.plot(simulator.data['_x','x',0][0],simulator.data['_x','x',1][0],'x',color='r')
axs_3.plot(simulator.data['_x','x',0][-1],simulator.data['_x','x',1][-1],'o',color='r')
# for i in range(N_sim):
#     axs_3.plot(simulator.data['_x','x',0][i],simulator.data['_x','x',1][i],'o',color='g')
axs_3.set_xlabel('x0')
axs_3.set_ylabel('x1')
axs_3.legend(['trajectory','start','end'])

fig4, axs_4 = plt.subplots(1,1)
axs_4.plot(simulator.data['_aux','stage_cost'])
axs_4.set_ylabel('stage_cost')
axs_4.set_xlabel('time')
print(sum(simulator.data['_aux','stage_cost']))
plt.show(block=False)
fig1.savefig("States_Robust.svg")
fig2.savefig("Inputs_Robust.svg")
fig3.savefig("Trajectory_Robust.svg")
fig4.savefig("Cost_Robust.svg")
input('press any key to exit')
