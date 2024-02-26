import numpy as np
from casadi import *
from casadi.tools import *

class RobustMPC:
    def __init__(self):
        def create_lis(cuts, ordering_dimension, dim=0, lis=[], last_idx=0):
            # Write a recursive function, that creates a search tree like list of lists with increasing numbers
            # The most inner list contains the index of the number of rectangles=number of cuts+1 of the dimension specified last in ordering dimension
            # dim is the dimension, that denotes the depth of recursion
            # The ordering dimension is a list of the dimensions, that are cut in the order of the list
            for i in range(int(cuts[ordering_dimension[dim]] + 1)):
                if dim == len(ordering_dimension) - 1:
                    # print('Problem in last layer')
                    lis.append(last_idx)
                    last_idx += 1
                else:
                    # print('Problem in layer '+str(dim+1)+' of '+str(len(ordering_dimension))+' layers')
                    new_new_lis, last_idx = create_lis(cuts, ordering_dimension, dim + 1, [], last_idx)
                    lis.append(new_new_lis)
            return lis, last_idx
        from nl_double_int_nmpc.template_model import template_model
        model = template_model()

        # %%
        p0 = np.array([0.25])
        p_min = np.array([0.2])
        p_max = np.array([0.3])
        model._rhs['x']
        system = Function('system', [model.x, model.u, model.tvp], [model._rhs['x']])

        # %%
        system(model.x, model.u, model.tvp)
        dt = 1
        nx = 2  # Two Species
        nu = 2  # 4 Ventilation rates
        nd = 1  # 5 Uncertainties: Lamp Temperatures and Environment temp.
        lb_x = 0 * np.ones((nx, 1))
        ub_x = 10 * np.ones((nx, 1))
        # input constraints
        lb_u = np.array([[-10], [-5]])  # -10*np.ones((nu,1))
        ub_u = np.array([[10], [5]])  # 10*np.ones((nu,1))
        N = 10  # Prediction horizon

        # %%
        cuts = np.zeros((nx, 1))

        cuts[0] = 4
        cuts[1] = 4

        ordering_dimension = [1, 0]

        ns = 1
        for i in range(nx):
            ns *= (cuts[i] + 1)
        ns = int(ns[0])
        print(ns)

        # %%
        lis, _ = create_lis(cuts, ordering_dimension, dim=0, lis=[], last_idx=0)

        # %%
        print(lis)

        # %%
        opt_x = struct_symSX([
            entry('x_min', shape=nx, repeat=[N + 1, ns]),
            entry('x_max', shape=nx, repeat=[N + 1, ns]),
            entry('u', shape=nu, repeat=[N, ns])
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
                res = [xs]
            return res

        def depth(l):
            if isinstance(l, list):
                return 1 + max(depth(item) for item in l) if l else 1
            else:
                return 0

        # Functions to get the index from a vector containing the index of each cut in the respective dimension
        def from_count_get_s(count, cuts):
            assert len(cuts) == len(count)
            s = 0
            remainder = ns
            for l in ordering_dimension:
                remainder /= (cuts[l] + 1)
                s += remainder * count[l]
            return int(s)

        # And the oter way around
        def from_s_get_count(idx, cuts):
            count = np.zeros((nx, 1))
            remainder = ns
            rest = idx
            for l in ordering_dimension:
                if cuts[l] > 0:
                    remainder /= (cuts[l] + 1)
                    count[l] = rest // remainder
                    rest -= remainder * count[l]
            return count

        # Recursive function to set up the equality constraints defining the cuts

        def constraint_function(l, ord_dim, opt_x, h, lbg, ubg, i):
            for k in range(len(l)):
                idx = flatten(l[k])
                dim = ord_dim[-depth(l)]
                for s in idx:
                    if s == idx[0] and k == 0:
                        h.append(opt_x['x_min', i, s, dim] - opt_x['x_min', i, 0, dim])
                        lbg.append(0)
                        ubg.append(0)
                    else:
                        h.append(opt_x['x_min', i, s, dim] - opt_x['x_min', i, idx[0], dim])
                        # print(opt_x['alpha',i,s,d_min]-opt_x['alpha',i,idx[0],d_min])
                        lbg.append(0)
                        ubg.append(0)
                    if s == idx[-1] and k == len(l) - 1:  ###????
                        h.append(opt_x['x_max', i, s, dim] - opt_x['x_max', i, -1, dim])
                        # print(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,-1,d_max])
                        lbg.append(0)
                        ubg.append(0)
                    else:
                        h.append(opt_x['x_max', i, s, dim] - opt_x['x_max', i, idx[-1], dim])
                        # print(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,idx[-1],d_max])
                        lbg.append(0)
                        ubg.append(0)
                if k >= 1:
                    prev_last = flatten(l[k - 1])[-1]
                    h.append(opt_x['x_min', i, idx[0], dim] - opt_x['x_max', i, prev_last, dim])
                    # print(opt_x['alpha',i,idx[0],d_min]+opt_x['alpha',i,prev_last,d_max])
                    lbg.append(0)
                    ubg.append(0)
                if depth(l) > 1:
                    h, lbg, ubg = constraint_function(l[k], ord_dim, opt_x, h, lbg, ubg, i)

            return h, lbg, ubg

        # %%
        x_init_min = SX.sym('x_init_min', nx, 1)
        x_init_max = SX.sym('x_init_max', nx, 1)
        p_minus = SX.sym('p_minus', nd, 1)
        p_plus = SX.sym('p_plus', nd, 1)
        x_RCIS_plus = SX.sym('x_RCIS_plus', nx, 1)
        x_RCIS_minus = SX.sym('x_RCIS_minus', nx, 1)

        # %%
        x = SX.sym('x', nx, 1)
        u = SX.sym('u', nu, 1)
        u_bef = SX.sym('u_bef', nu, 1)
        x_ref = SX.sym('x_ref', nx, 1)
        stage_cost = (x - x_ref).T @ (x - x_ref)  + (u - u_bef).T @ (u - u_bef)
        stage_cost_fcn = Function('stage_cost', [x, u, u_bef, x_ref], [stage_cost])
        term_cost = (x - x_ref).T @ (x - x_ref)
        term_cost_fcn = Function('term_cost', [x, x_ref], [term_cost])

        # %%
        J = 0
        g = []  # constraint expression g
        lb_g = []  # lower bound for constraint expression g
        ub_g = []  # upper bound for constraint expression g

        # First u corresponds to feed directly after measuring x_init

        g.append(opt_x['x_max', 0, -1] - x_init_max)
        g.append(opt_x['x_min', 0, 0] - x_init_min)
        lb_g.append(np.zeros((2 * nx, 1)))
        ub_g.append(np.zeros((2 * nx, 1)))

        for i in range(N):
            # objective
            for s in range(ns):
                if i == 0:
                    J += stage_cost_fcn(opt_x['x_max', i, s], opt_x['u', i, s], u_bef, x_ref)
                    J += stage_cost_fcn(opt_x['x_min', i, s], opt_x['u', i, s], u_bef, x_ref)
                else:
                    J += stage_cost_fcn(opt_x['x_max', i, s], opt_x['u', i, s], opt_x['u', i - 1, s], x_ref)
                    J += stage_cost_fcn(opt_x['x_min', i, s], opt_x['u', i, s], opt_x['u', i - 1, s], x_ref)

            # equality constraints (system equation)

            for s in range(ns):
                x_next_max = system(opt_x['x_max', i, s], opt_x['u', i, s], p_max)
                x_next_min = system(opt_x['x_min', i, s], opt_x['u', i, s], p_min)

                g.append(-x_next_max + opt_x['x_max', i + 1, -1])
                g.append(x_next_min - opt_x['x_min', i + 1, 0])
                lb_g.append(np.zeros((2 * nx, 1)))
                ub_g.append(inf * np.ones((2 * nx, 1)))

        for s in range(ns):
            J += term_cost_fcn(opt_x['x_max', -1, s], x_ref)
            J += term_cost_fcn(opt_x['x_min', -1, s], x_ref)

        # Cutting
        for i in range(0, N + 1):
            g, lb_g, ub_g = constraint_function(lis, ordering_dimension, opt_x, g, lb_g, ub_g, i)
            for s in range(ns):
                g.append(opt_x['x_max', i, s] - opt_x['x_min', i, 0])
                g.append(opt_x['x_max', i, -1] - opt_x['x_min', i, s])
                g.append(opt_x['x_min', i, s] - opt_x['x_min', i, 0])
                g.append(opt_x['x_max', i, -1] - opt_x['x_max', i, s])
                lb_g.append(np.zeros((4 * nx, 1)))
                ub_g.append(np.ones((4 * nx, 1)) * inf)

        # Introduce RCIS at the end of the prediction horizon!
        g.append(x_RCIS_plus - opt_x['x_max', -1, -1])
        g.append(opt_x['x_min', -1, 0] - x_RCIS_minus)
        lb_g.append(np.zeros((2 * nx, 1)))
        ub_g.append(inf * np.ones((2 * nx, 1)))
        self.opt_x= opt_x
        # Concatenate constraints
        g = vertcat(*g)
        self.lb_g = vertcat(*lb_g)
        self.ub_g = vertcat(*ub_g)
        self.u0=np.array([[0],[0]])
        self.prob = {'f': J, 'x': opt_x.cat, 'g': g,
                'p': vertcat(x_init_min, x_init_max, x_ref, u_bef, p_plus, p_minus, x_RCIS_plus, x_RCIS_minus)}

        self.rob_mpc_solver = nlpsol('solver', 'ipopt', self.prob,
                                {'ipopt.max_iter': 4000, 'ipopt.linear_solver': 'mumps', 'ipopt.ma86_u': 1e-6,
                                 'ipopt.print_level': 2, 'ipopt.sb': 'yes', 'print_time': 0,
                                 'ipopt.ma57_automatic_scaling': 'yes', 'ipopt.ma57_pre_alloc': 10,
                                 'ipopt.ma27_meminc_factor': 100, 'ipopt.ma27_pivtol': 1e-4,
                                 'ipopt.ma27_la_init_factor': 100,
                                 'ipopt.warm_start_init_point': 'yes'})  # ,'ipopt.hessian_approximation':'limited-memory'})
    def make_step(self, x0):
        # Solve the NLP
        nx=2
        x_ref=np.array([[5],[1]])
        lb_x = 0 * np.ones((nx, 1))
        ub_x = 10 * np.ones((nx, 1))
        # input constraints
        lb_u = np.array([[-10], [-5]])  # -10*np.ones((nu,1))
        ub_u = np.array([[10], [5]])  # 10*np.ones(

        sol = self.rob_mpc_solver(x0=self.opt_x(0), lbx=lb_x, ubx=ub_x, lbg=self.lb_g, ubg=self.ub_g,
                             p=vertcat(x0, x_ref, self.u0, 0.3, 0.2,np.array([[-4.49345e-09], [-4.62047e-09]]) , np.array([[10], [6.43288]])))

        return sol.full()['x', 0:2]