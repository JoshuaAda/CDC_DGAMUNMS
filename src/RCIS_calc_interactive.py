# %%
# Essentials
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb as pdb
# Tools
from IPython.display import clear_output
import copy
import sys
import pandas as pd
# Specialized packages
from casadi import *
from casadi.tools import *
import control
import time as time
import os.path
from scipy.linalg import solve_discrete_are, inv, eig, block_diag 

import scipy.signal as signal

# For Plotting
from matplotlib.animation import FuncAnimation, ImageMagickFileWriter
from cycler import cycler
import time as time
import ipympl
from matplotlib.widgets import Slider, Button
# Custom packages
import do_mpc



# Customizing Matplotlib:
mpl.use('TkAgg')
mpl.rcParams['font.size'] = 15
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = 'true'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['axes.labelpad'] = 6


# %% [markdown]
# ## Create Model

# %%
dt=1
nx=2 # Two Species
nu=2 #4 Ventilation rates
nd=1 #5 Uncertainties: Lamp Temperatures and Environment temp.

# %%
from nl_double_int_nmpc.template_model import template_model
model = template_model()
x_sp = np.array([[5],[1]])

# %%
model._rhs['x']
system=Function('system',[model.x,model.u,model.p['p']],[model._rhs_fun(model.x,model.u,[],[],vertcat(model.p['p'],x_sp),[])])

# %%
system(model.x,model.u,model.p['p'])

# %%
lb_x = 0*np.ones((nx,1))
ub_x = 10*np.ones((nx,1))
# input constraints
lb_u = np.array([[-10],[-5]])#-10*np.ones((nu,1))
ub_u = np.array([[10],[5]])#10*np.ones((nu,1))

# %% [markdown]
# # Creating the Simulator

# %%
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = dt)


# %%
p_template = simulator.get_p_template()

# %%
p_template.keys()


# %%
p0=np.array([0.15])
p_min=np.array([0.0])
p_max=np.array([0.3])

# %%
def p_fun_max(t):
    p_template['p'] = p_max
    p_template['x_sp'] = x_sp
    return p_template

# %%
def p_fun_0(t):
    p_template['p'] = p0
    p_template['x_sp'] = x_sp
    return p_template

# %%
def p_fun_min(t):
    p_template['p'] = p_min
    p_template['x_sp'] = x_sp
    return p_template

# %%
def p_fun_var(t):
    np.random.seed(1234+int(t))
    p_template['p'] = np.random.uniform(p_min,p_max)
    p_template['x_sp'] = x_sp
    return p_template

# %%
p_fun_var(120)['p']

# %%
def get_p_shape(p_min,p_max,x0_min,x0_max,u0,gs):
    p=[]
    x_next_list=[]
    for j in range(gs**(nx+nd)):
        p0=np.zeros((nd,1))
        x0=np.zeros((nx,1))
        for d in range(nd):
            p0[d]=(p_min[d]+(p_max[d]-p_min[d])*((j//(gs**d))%(nd+nx))/(gs-1)).squeeze()
        for i in range(nx):
            x0[i]=(x0_min[i]+(x0_max[i]-x0_min[i])*((j//(gs**(nd+i)))%(nd+nx))/(gs-1)).squeeze()
        #print(p0,x0)
        x_next=system(x0,u0,p0)
        x_next_list.append(x_next)
        p.append(p0)
    
    return p,x_next_list

# %% [markdown]
# ## Looking at the Uncertainty

# %%
from scipy.spatial import ConvexHull, convex_hull_plot_2d

# %%
def get_xnext(x,u,p):
    return np.array([system(x,u,p)]).reshape(nx,1)



# %%
x0_min=np.array([[0.0,0.5]]).T
x0_max=np.array([[0.5,1]]).T
u0=np.array([[0,0]]).T
x_min=get_xnext(x0_min,u0,p_min)
x_max=get_xnext(x0_max,u0,p_max)
p_test,x_next_list=get_p_shape(p_min,p_max,x0_min,x0_max,u0,3)
x_next_cord=np.concatenate(x_next_list,axis=0).reshape(-1,2)
hull=ConvexHull(x_next_cord)
#p_hull,x_next_hull=get_p_hull(p_min,p_max,x0,u0,10)
#x_next_hull_c=np.concatenate(x_next_hull,axis=0).reshape(-1,2)
fig,ax =plt.subplots(1,1,figsize=(8,6))
#ax.plot(295.5,294.5,'ro',label='set_point')
#convex_hull_plot_2d(hull, ax=ax)
ax.plot(x_next_cord[np.append(hull.vertices,hull.vertices[0]),0], x_next_cord[np.append(hull.vertices,hull.vertices[0]),1], 'ro--', lw=1)
#ax.plot(x0[0],x0[1],'kx',label='x_0')
ax.add_patch(mpl.patches.Rectangle(lb_x, ub_x[0]-lb_x[0],ub_x[1]-lb_x[0], color="None",ec='blue'))
line = ax.add_patch(mpl.patches.Rectangle(x_min, x_max[0]-x_min[0],x_max[1]-x_min[1], color="None",ec='blue',hatch='/'))
ini_rect= ax.add_patch(mpl.patches.Rectangle(x0_min, x0_max[0]-x0_min[0],x0_max[1]-x0_min[1], color="None",ec='gray'))
ax.plot(x_next_cord[:,0], x_next_cord[:,1], 'ko',markersize=2, lw=1)
ax.set_xlim([lb_x[0]-0.2,ub_x[0]+0.2])
ax.set_ylim([lb_x[1]-0.2,ub_x[1]+0.2])
plt.show()



# %%
x0_min=np.array([[0.0,0.5]]).T
x0_max=np.array([[0.5,0.8]]).T
u0=np.array([[0.1,0.1]]).T
x_min=get_xnext(x0_min,u0,p_min)
x_max=get_xnext(x0_max,u0,p_max)
p_test,x_next_list=get_p_shape(p_min,p_max,x0_min,x0_max,u0,3)
x_next_cord=np.concatenate(x_next_list,axis=0).reshape(-1,2)
#hull=ConvexHull(x_next_cord)
fig,ax =plt.subplots(1,1)
#ax.plot(295.5,294.5,'ro',label='set_point')
ax.add_patch(mpl.patches.Rectangle(x0_min, x0_max[0]-x0_min[0],x0_max[1]-x0_min[1], color="None",ec='grey',hatch='x'))
#ax.add_patch(mpl.patches.Rectangle(lb_x, ub_x[0]-lb_x[0],ub_x[1]-lb_x[0], color="None",ec='red'))
line = ax.add_patch(mpl.patches.Rectangle(x_min, x_max[0]-x_min[0],x_max[1]-x_min[1], color="None",ec='green',hatch='/'))
points=ax.plot(x_next_cord[:,0],x_next_cord[:,1],'ko',markersize=2)
#uncert_hull=ax.plot(x_next_cord[np.append(hull.vertices,hull.vertices[0]),0], x_next_cord[np.append(hull.vertices,hull.vertices[0]),1], 'ro--', lw=1)
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='u0',
    valmin=-5,
    valmax=5,
    valinit=0,
)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label="u1",
    valmin=-5,
    valmax=5,
    valinit=0,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    x_min=get_xnext(x0_min,np.array([[freq_slider.val,amp_slider.val]]).T,p_min)
    x_max=get_xnext(x0_max,np.array([[freq_slider.val,amp_slider.val]]).T,p_max)
    line.set(xy=x_min, width=x_max[0]-x_min[0],height=x_max[1]-x_min[1])
    p_test,x_next_list=get_p_shape(p_min,p_max,x0_min,x0_max,np.array([[freq_slider.val,amp_slider.val]]).T,3)
    x_next_cord=np.concatenate(x_next_list,axis=0).reshape(-1,2)
    points[0].set(xdata=x_next_cord[:,0],ydata=x_next_cord[:,1])
    ax.relim()
    ax.autoscale()
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()

# %% [markdown]
# ## Computation of a Robust invariant control set
# The RCIS, we compute here is a N-set RCIS. The cuts in each step are propoesed to be the same

# %% [markdown]
# ### Biggest Set

# %%
x_ref=SX.sym('x_ref',nx,1)
p_plus=SX.sym('p_plus',nd,1)
p_minus=SX.sym('p_minus',nd,1)

# %% Number of steps
N=3
# %%
cuts=np.zeros((nx,1))

cuts[0]=4
cuts[1]=4

ordering_dimension=[1,0]

ns=1
for i in range(nx):
    ns*=(cuts[i]+1)
ns=int(ns[0])
print(ns)


# %%
def create_lis(cuts,ordering_dimension,dim=0,lis=[],last_idx=0):
    # Write a recursive function, that creates a search tree like list of lists with increasing numbers
    # The most inner list contains the index of the number of rectangles=number of cuts+1 of the dimension specified last in ordering dimension
    # dim is the dimension, that denotes the depth of recursion
    # The ordering dimension is a list of the dimensions, that are cut in the order of the list
    for i in range(int(cuts[ordering_dimension[dim]]+1)):
        if dim==len(ordering_dimension)-1:
            #print('Problem in last layer')
            lis.append(last_idx)
            last_idx+=1
        else:
            #print('Problem in layer '+str(dim+1)+' of '+str(len(ordering_dimension))+' layers')
            new_new_lis, last_idx=create_lis(cuts,ordering_dimension,dim+1,[],last_idx)
            lis.append(new_new_lis)
    return lis, last_idx

lis,_=create_lis(cuts,ordering_dimension)



# %%
print(lis)

# %%
opt_x = struct_symSX([
    entry('x_min', shape=nx ,repeat=[N,ns]),
    entry('x_max', shape=nx, repeat=[N,ns]),
    entry('u', shape=nu, repeat=[N,ns])
])

# %%
lb_opt_x = opt_x(0)
ub_opt_x = opt_x(np.inf)

# %%
lb_opt_x['x_min'] = lb_x
lb_opt_x['x_max'] = lb_x
ub_opt_x['x_min'] = ub_x
ub_opt_x['x_max'] = ub_x

lb_opt_x['u'] = lb_u
ub_opt_x['u'] = ub_u

# %%
# These functions are used in a recursive algorithm to create the constraints for the cuts
def flatten(xs):
    if isinstance(xs, list):
        res = []
        def loop(ys):
            for i in ys:
                if isinstance(i, list):
                    loop(i)
                else:
                    res.append(i)
        loop(xs)
    else:
        res=[xs]
    return res

def depth(l):
    if isinstance(l, list):
        return 1 + max(depth(item) for item in l) if l else 1
    else:
        return 0



# Functions to get the index from a vector containing the index of each cut in the respective dimension
def from_count_get_s(count, cuts):
    assert len(cuts)==len(count)
    s=0
    remainder=ns
    for l in ordering_dimension:
        remainder/=(cuts[l]+1)
        s+=remainder*count[l]
    return int(s)
# And the oter way around
def from_s_get_count(idx, cuts):
    count=np.zeros((nx,1))
    remainder=ns
    rest=idx
    for l in ordering_dimension:
        if cuts[l]>0:
            remainder/=(cuts[l]+1)
            count[l]=rest//remainder
            rest-=remainder*count[l]
    return count
# Recursive function to set up the equality constraints defining the cuts

def constraint_function_RCIS(l,ord_dim,opt_x,i,h,lbg,ubg):
    for k in range(len(l)):
        idx=flatten(l[k])
        dim=ord_dim[-depth(l)]
        for s in idx:
            if s==idx[0] and k==0:
                h.append(opt_x['x_min',i,s,dim]-opt_x['x_min',i,0,dim])
                lbg.append(0)
                ubg.append(0)
            else:
                h.append(opt_x['x_min',i,s,dim]-opt_x['x_min',i,idx[0],dim])
                #print(opt_x['alpha',i,s,d_min]-opt_x['alpha',i,idx[0],d_min])
                lbg.append(0)
                ubg.append(0)
            if s==idx[-1] and k==len(l)-1:###????
                h.append(opt_x['x_max',i,s,dim]-opt_x['x_max',i,-1,dim])
                #print(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,-1,d_max])
                lbg.append(0)
                ubg.append(0)
            else:
                h.append(opt_x['x_max',i,s,dim]-opt_x['x_max',i,idx[-1],dim])
                #print(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,idx[-1],d_max])
                lbg.append(0)
                ubg.append(0)
        if k>=1:
            prev_last=flatten(l[k-1])[-1]
            h.append(opt_x['x_min',i,idx[0],dim]-opt_x['x_max',i,prev_last,dim])
            #print(opt_x['alpha',i,idx[0],d_min]+opt_x['alpha',i,prev_last,d_max])
            lbg.append(0)
            ubg.append(0)
        if depth(l) >1:
            h,lbg,ubg=constraint_function_RCIS(l[k],ord_dim,opt_x,i,h,lbg,ubg)
    
    return h,lbg,ubg

# %%
J=0
g=[]
lb_g=[]
ub_g=[]

for i in range(N):
    for s in range(ns):
        x_next_plus = system(opt_x['x_max',i,s], opt_x['u',i,s],p_plus)
        x_next_minus = system(opt_x['x_min',i,s], opt_x['u',i,s],p_minus)
        if i==N-1:
            g.append(opt_x['x_max',0,-1]-x_next_plus)
            g.append(x_next_minus - opt_x['x_min',0, 0])
        else:
            g.append(opt_x['x_max',i+1,-1]-x_next_plus)
            g.append(x_next_minus - opt_x['x_min',i+1, 0])
        lb_g.append(np.zeros((2*nx,1)))
        ub_g.append(inf*np.ones((2*nx,1)))
    # Cutting for RCIS
    g,lb_g,ub_g=constraint_function_RCIS(lis,ordering_dimension,opt_x,i,g,lb_g,ub_g)
    for s in range(ns):
        g.append(opt_x['x_max',i,s]-opt_x['x_min',i,0])
        g.append(opt_x['x_max',i,-1]-opt_x['x_min',i,s])
        g.append(opt_x['x_min',i,s]-opt_x['x_min',i,0])
        g.append(opt_x['x_max',i,-1]-opt_x['x_max',i,s])
        lb_g.append(np.zeros((4*nx,1)))
        ub_g.append(np.ones((4*nx,1))*inf)


#g.append(x_ref-opt_x['x_min',0])
#g.append(-x_ref+opt_x['x_max',1])
#lb_g.append(np.zeros((2*nx,1)))
#ub_g.append(np.ones((2*nx,1))*inf)
J=-1#0
for i in range(1):
    J_mini=-1
    for ix in range(nx):
        J_mini=J_mini*(opt_x['x_max',i,-1,ix]-opt_x['x_min',i,0,ix])
    J+=J_mini
#=-(opt_x['x_max',0,-1]-opt_x['x_min',0,0])
#J-=(opt_x['x_max',1,-1,1]-opt_x['x_min',1,0,1])*10

g = vertcat(*g)
lb_g = vertcat(*lb_g)
ub_g = vertcat(*ub_g)

prob = {'f':J,'x':vertcat(opt_x),'g':g, 'p':vertcat(x_ref,p_plus,p_minus)}
solver_mx_inv_set = nlpsol('solver','ipopt',prob)

# %%
x_set=np.array([[0.1,0.1]]).T
opt_ro_initial=opt_x(0)
opt_ro_initial['x_min']=x_set
opt_ro_initial['x_max']=x_set
results=solver_mx_inv_set(p=vertcat(x_set,p_max,p_min),x0=opt_ro_initial, lbg=lb_g,ubg=ub_g,lbx=lb_opt_x,ubx=ub_opt_x)

# %%
res=opt_x(results['x'])

# %%

x_min_RCIS_num=res['x_min',0,0].full().reshape(nx,1)
    
x_max_RCIS_num=res['x_max',0,-1].full().reshape(nx,1)
for i in range(N):
    print(res['x_min',i,0])
    print(res['x_max',i,-1])

# %%
fig,ax=plt.subplots(1,1)
for i in range(N):
    for s in range(ns):
        x_min_min=system(res['x_min',i,s],res['u',i,s],p_min)
        x_min_max=system(res['x_min',i,s],res['u',i,s],p_max)
        x_max_min=system(res['x_max',i,s],res['u',i,s],p_min)
        x_max_max=system(res['x_max',i,s],res['u',i,s],p_max)
        #ax.plot(vertcat(res['x_min',i,s][0],x_min_min[0]),vertcat(res['x_min',i,s][1],x_min_min[1]),'y',linewidth=1)
        #ax[0].plot(vertcat(res['x_min',s][0],x_min_max[0]),vertcat(res['x_min',s][1],x_min_max[1]))
        #ax[0].plot(vertcat(res['x_max',s][0],x_max_min[0]),vertcat(res['x_max',s][1],x_max_min[1]))
        #ax.plot(vertcat(res['x_max',i,s][0],x_max_max[0]),vertcat(res['x_max',i,s][1],x_max_max[1]),'b',linewidth=1)
        #ax[1].arrow(x=np.array(res['x_min',s][0])[0][0],y=np.array(res['x_min',s][1])[0][0],dx=np.array(x_min_min[0]-res['x_min',s][0])[0][0],dy=np.array(x_min_min[1]-res['x_min',s][1])[0][0])

    for s in range(ns):
        ax.add_patch(mpl.patches.Rectangle(np.array(res['x_min',i,s]), np.array(res['x_max',i,s]-res['x_min',i,s])[0][0] , np.array(res['x_max',i,s]-res['x_min',i,s])[1][0], color="None",ec='grey',hatch='/'))

#ax.add_patch(mpl.patches.Rectangle(lb_x, ub_x[0]-lb_x[0],ub_x[1]-lb_x[0], color="None",ec='red'))
ax.set_ylabel('x_2')
ax.set_xlabel('x_1')
ax.set_xlim([0,10])
ax.set_ylim([0,10])
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.show(block=False)
plt.savefig("RCIS_3.svg")
# %%
#np.save('u_new.npy',[np.array([s.full() for s in res['u',i]]) for i in range(N)])
#np.save('x_min_new.npy',[np.array([s.full() for s in res['x_min',i]]) for i in range(N)])
#np.save('x_max_new.npy',[np.array([s.full() for s in res['x_max',i]]) for i in range(N)])

'''
# %% [markdown]
# # Try to increase the RCIS size by running robust MPCs around it

# %% [markdown]
# Here comes a robust MPC based on monotonicity, but without the condition, that the input needs to be the same in the first tube, as this is just as an extension to allow for larger RCIS

# %%
N = 10 # Prediction horizon 

# %%
cuts=np.zeros((nx,1))

cuts[0]=4
cuts[1]=4

ordering_dimension=[1,0]

ns=1
for i in range(nx):
    ns*=(cuts[i]+1)
ns=int(ns[0])
print(ns)


# %%
lis,_=create_lis(cuts,ordering_dimension,dim=0,lis=[],last_idx=0)



# %%
print(lis)

# %%
opt_x = struct_symSX([
    entry('x_min', shape=nx ,repeat=[N+1,ns]),
    entry('x_max', shape=nx, repeat=[N+1,ns]),
    entry('u', shape=nu, repeat=[N,ns])
])

# %%
lb_opt_x = opt_x(0)
ub_opt_x = opt_x(np.inf)

# %%
lb_opt_x['x_min'] = lb_x
lb_opt_x['x_max'] = lb_x
ub_opt_x['x_min'] = ub_x
ub_opt_x['x_max'] = ub_x

lb_opt_x['u'] = lb_u
ub_opt_x['u'] = ub_u

# %%
# These functions are used in a recursive algorithm to create the constraints for the cuts
def flatten(xs):
    if isinstance(xs, list):
        res = []
        def loop(ys):
            for i in ys:
                if isinstance(i, list):
                    loop(i)
                else:
                    res.append(i)
        loop(xs)
    else:
        res=[xs]
    return res

def depth(l):
    if isinstance(l, list):
        return 1 + max(depth(item) for item in l) if l else 1
    else:
        return 0



# Functions to get the index from a vector containing the index of each cut in the respective dimension
def from_count_get_s(count, cuts):
    assert len(cuts)==len(count)
    s=0
    remainder=ns
    for l in ordering_dimension:
        remainder/=(cuts[l]+1)
        s+=remainder*count[l]
    return int(s)
# And the oter way around
def from_s_get_count(idx, cuts):
    count=np.zeros((nx,1))
    remainder=ns
    rest=idx
    for l in ordering_dimension:
        if cuts[l]>0:
            remainder/=(cuts[l]+1)
            count[l]=rest//remainder
            rest-=remainder*count[l]
    return count
# Recursive function to set up the equality constraints defining the cuts

def constraint_function(l,ord_dim,opt_x,h,lbg,ubg,i):
    for k in range(len(l)):
        idx=flatten(l[k])
        dim=ord_dim[-depth(l)]
        for s in idx:
            if s==idx[0] and k==0:
                h.append(opt_x['x_min',i,s,dim]-opt_x['x_min',i,0,dim])
                lbg.append(0)
                ubg.append(0)
            else:
                h.append(opt_x['x_min',i,s,dim]-opt_x['x_min',i,idx[0],dim])
                #print(opt_x['alpha',i,s,d_min]-opt_x['alpha',i,idx[0],d_min])
                lbg.append(0)
                ubg.append(0)
            if s==idx[-1] and k==len(l)-1:###????
                h.append(opt_x['x_max',i,s,dim]-opt_x['x_max',i,-1,dim])
                #print(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,-1,d_max])
                lbg.append(0)
                ubg.append(0)
            else:
                h.append(opt_x['x_max',i,s,dim]-opt_x['x_max',i,idx[-1],dim])
                #print(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,idx[-1],d_max])
                lbg.append(0)
                ubg.append(0)
        if k>=1:
            prev_last=flatten(l[k-1])[-1]
            h.append(opt_x['x_min',i,idx[0],dim]-opt_x['x_max',i,prev_last,dim])
            #print(opt_x['alpha',i,idx[0],d_min]+opt_x['alpha',i,prev_last,d_max])
            lbg.append(0)
            ubg.append(0)
        if depth(l) >1:
            h,lbg,ubg=constraint_function(l[k],ord_dim,opt_x,h,lbg,ubg,i)
    
    return h,lbg,ubg

# %%
x_init_min=SX.sym('x_init_min',nx,1)
x_init_max=SX.sym('x_init_max',nx,1)
p_minus=SX.sym('p_minus',nd,1)
p_plus=SX.sym('p_plus',nd,1)
x_RCIS_plus=SX.sym('x_RCIS_plus',nx,1)
x_RCIS_minus=SX.sym('x_RCIS_minus',nx,1)

# %%
x=SX.sym('x',nx,1)
u=SX.sym('u',nu,1)
u_bef=SX.sym('u_bef',nu,1)
x_ref=SX.sym('x_ref',nx,1)
stage_cost=(x-x_ref).T@(x-x_ref)+u.T@u+(u-u_bef).T@(u-u_bef)
stage_cost_fcn=Function('stage_cost',[x,u,u_bef,x_ref],[stage_cost])
term_cost=(x-x_ref).T@(x-x_ref)
term_cost_fcn=Function('term_cost',[x,x_ref],[term_cost])

# %%
J = 0
g = []    # constraint expression g
lb_g = []  # lower bound for constraint expression g
ub_g = []  # upper bound for constraint expression g

# First u corresponds to feed directly after measuring x_init


g.append(opt_x['x_max',0,-1]-x_init_max)
g.append(opt_x['x_min',0,0]-x_init_min)
lb_g.append(np.zeros((2*nx,1)))
ub_g.append(np.zeros((2*nx,1)))


for i in range(N):
    # objective
    for s in range(ns):
        if i==0:
            J += stage_cost_fcn(opt_x['x_max',i,s], opt_x['u',i,s],u_bef,x_ref)
            J += stage_cost_fcn(opt_x['x_min',i,s], opt_x['u',i,s],u_bef,x_ref)
        else:
            J += stage_cost_fcn(opt_x['x_max',i,s], opt_x['u',i,s],opt_x['u',i-1,s],x_ref)
            J += stage_cost_fcn(opt_x['x_min',i,s], opt_x['u',i,s],opt_x['u',i-1,s],x_ref)

    # equality constraints (system equation)

    for s in range(ns):
        x_next_max = system(opt_x['x_max',i,s], opt_x['u',i,s],p_max)
        x_next_min = system(opt_x['x_min',i,s], opt_x['u',i,s],p_min)

        g.append(-x_next_max + opt_x['x_max',i+1,-1])
        g.append(x_next_min - opt_x['x_min',i+1,0])
        lb_g.append(np.zeros((2*nx,1)))
        ub_g.append(inf*np.ones((2*nx,1)))

        

for s in range(ns):
    J += term_cost_fcn(opt_x['x_max',-1,s],x_ref)
    J += term_cost_fcn(opt_x['x_min',-1,s],x_ref)
            
        
        
#Cutting 
for i in range(0,N+1):
    g,lb_g,ub_g=constraint_function(lis,ordering_dimension,opt_x,g,lb_g,ub_g,i)
    for s in range(ns):
        g.append(opt_x['x_max',i,s]-opt_x['x_min',i,0])
        g.append(opt_x['x_max',i,-1]-opt_x['x_min',i,s])
        g.append(opt_x['x_min',i,s]-opt_x['x_min',i,0])
        g.append(opt_x['x_max',i,-1]-opt_x['x_max',i,s])
        lb_g.append(np.zeros((4*nx,1)))
        ub_g.append(np.ones((4*nx,1))*inf)



# Introduce RCIS at the end of the prediction horizon!
g.append(x_RCIS_plus - opt_x['x_max', -1,-1])
g.append( opt_x['x_min',-1, 0]-x_RCIS_minus)
lb_g.append(np.zeros((2*nx,1)))
ub_g.append(inf*np.ones((2*nx,1)))

# Concatenate constraints
g = vertcat(*g)
lb_g = vertcat(*lb_g)
ub_g = vertcat(*ub_g)

prob = {'f':J,'x':opt_x.cat,'g':g, 'p':vertcat(x_init_min,x_init_max,x_ref,u_bef,p_plus,p_minus,x_RCIS_plus,x_RCIS_minus)}

rob_mpc_solver = nlpsol('solver','ipopt',prob,{'ipopt.max_iter':4000,'ipopt.linear_solver':'mumps','ipopt.ma86_u':1e-6,'ipopt.print_level':2, 'ipopt.sb': 'yes', 'print_time':0,'ipopt.ma57_automatic_scaling':'yes','ipopt.ma57_pre_alloc':10,'ipopt.ma27_meminc_factor':100,'ipopt.ma27_pivtol':1e-4,'ipopt.ma27_la_init_factor':100,'ipopt.warm_start_init_point':'yes'})#,'ipopt.hessian_approximation':'limited-memory'})


# %% [markdown]
# Create now a sampling environment, that lets one sample around the RCIS in rectangular tubes

# %%
# Some MPC parameters
x_set=np.array([[0.1,0.1]]).T
u_initial=np.array([[0,0]]).T


# %%
delx1=0.1
delx2=0.1
# Round RCIS down to align with grid
x_RCIS_plus_num=np.array([[x_max_RCIS_num[0]-x_max_RCIS_num[0]%delx1 , x_max_RCIS_num[1]-x_max_RCIS_num[1]%delx2]]).reshape(nx)
x_RCIS_minus_num=np.array([[0,0]]).reshape(nx)
layers=10
sol= []
for i in range(layers):
    leng=int((x_RCIS_plus_num[0]-x_RCIS_minus_num[0])//delx1 + (x_RCIS_plus_num[1]-x_RCIS_minus_num[1])//delx2 + (i)*2+1)
    count=0
    for k in range(leng):
        if k<(x_RCIS_plus_num[0]-x_RCIS_minus_num[0])//delx1+i+1:
            x_box_min=np.array([[x_RCIS_minus_num[0]+k*delx1,x_RCIS_plus_num[1]+i*delx2]]).reshape(nx)
            x_box_max=np.array([[x_RCIS_minus_num[0]+(k+1)*delx1,x_RCIS_plus_num[1]+(i+1)*delx2]]).reshape(nx)
        else:
            x_box_min=np.array([[x_RCIS_plus_num[0]+i*delx1, x_RCIS_minus_num[1]+count*delx2]]).reshape(nx)
            x_box_max=np.array([[x_RCIS_plus_num[0]+(i+1)*delx1, x_RCIS_minus_num[1]+(count+1)*delx2]]).reshape(nx)
            count+=1
        print('Layer {}/{}; inner iteration {}/{}'.format(i+1,layers,k,leng))
        # Solve MPC problem
        opt_x_initial=opt_x(0)
        opt_x_initial['x_min']=x_box_min
        opt_x_initial['x_max']=x_box_max
        results=rob_mpc_solver(p=vertcat(x_box_min,x_box_max,x_set,u_initial,p_max,p_min,x_max_RCIS_num,x_min_RCIS_num),x0=opt_x_initial, lbg=lb_g,ubg=ub_g,lbx=lb_opt_x,ubx=ub_opt_x)
        res=opt_x(results['x'])
        sol.append({'x_min':x_box_min,'x_max':x_box_max,'res':res,'optimal':rob_mpc_solver.stats()['success']})
        



# %%
# Plot the results
fig,ax=plt.subplots(1,1)
# First plot RCIS
ax.add_patch(mpl.patches.Rectangle(x_min_RCIS_num, x_max_RCIS_num[0]-x_min_RCIS_num[0],x_max_RCIS_num[1]-x_min_RCIS_num[1], color="None",ec='grey',hatch='/'))
# Then plot the rectangles of the MPC solution and color them green if the solution was optimal else red
for i in range(len(sol)):
    if sol[i]['optimal']:
        ax.add_patch(mpl.patches.Rectangle(sol[i]['x_min'], sol[i]['x_max'][0]-sol[i]['x_min'][0],sol[i]['x_max'][1]-sol[i]['x_min'][1], color="None",ec='green'))
    else:
        ax.add_patch(mpl.patches.Rectangle(sol[i]['x_min'], sol[i]['x_max'][0]-sol[i]['x_min'][0],sol[i]['x_max'][1]-sol[i]['x_min'][1], color="None",ec='red'))
ax.set_ylim([0,4])
ax.set_xlim([0,6])

# %%
x_max_RCIS_num

# %%
'''


