# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 20:33:41 2023

@author: halvorak
"""
import casadi as ca
import numpy as np
import scipy.stats
import os
import shutil

def model(dt, integrator_casadi = "cvodes", opts_integrator = {}):
    

    #deterministic parameters - to use for the MPC formulation
    mu_m_d  = 0.02
    K_m_d	  = 0.05
    K_i_d	  = 5.0
    v_par_d = 0.004
    Y_p_d	  = 1.2
    Y_x_d = 0.5
    S_in_d = 200.
    
    #states
    X = ca.MX.sym("X", 1)
    S = ca.MX.sym("S", 1)
    P = ca.MX.sym("P", 1)
    V = ca.MX.sym("V", 1)
    x_var = ca.vertcat(X, S, P, V)
    
    #inputs
    u = ca.MX.sym("u", 1)
    
    #substrate inhibition
    mu_d = mu_m_d*S/(K_m_d + S + (S**2/K_i_d))
    
    #ODE - deterministic system (MPC)
    Xdot_d = mu_d*X - (u/V)*X
    Sdot_d = -mu_d*X/Y_x_d - v_par_d*X/Y_p_d + (u/V)*(S_in_d - S)
    Pdot_d = v_par_d*X - (u/V)*P
    Vdot_d = u
    xdot_d = ca.vertcat(Xdot_d, Sdot_d, Pdot_d, Vdot_d)
    
    #cost function.
    L = -P
    #formulation for orthogonal collocation
    F = ca.Function("F", [x_var, u], [xdot_d, L], ["x_var", "u"], ["xdot", "L"])
    
    
    #formulate deterministic ODE problem (MPC)
    opts_d = {}
    ode_d = {"x": x_var, "p": u, "ode": xdot_d*dt, "quad": L}
    integrator_d = ca.integrator("integrator_d", integrator_casadi, ode_d, opts_integrator)
    # integrator_d = ca.integrator("integrator_d", integrator_casadi, ode_d, opts_d)
    
    
    #we need a real/stochastic system. Assume parametric uncertainty
    # symbolic parameters - to use for the (stochastic) system
    mu_m  = ca.MX.sym("mu_m", 1)
    K_m	  = ca.MX.sym("K_m", 1)
    K_i	  = ca.MX.sym("K_i", 1)
    v_par = ca.MX.sym("v_par", 1)
    Y_p	  = ca.MX.sym("Y_p", 1)
    Y_x = ca.MX.sym("Y_x", 1)
    S_in = ca.MX.sym("S_in", 1)
    par = ca.vertcat(mu_m, K_m, K_i, v_par, Y_p, Y_x, S_in)

    #substrate inhibition - stochastic system    
    mu = mu_m*S/(K_m + S + (S**2/K_i))
    
    #ODE - stochastic system
    Xdot = mu*X - (u/V)*X
    Sdot = -mu*X/Y_x - v_par*X/Y_p + (u/V)*(S_in - S)
    Pdot = v_par*X - (u/V)*P
    Vdot = u
    xdot = ca.vertcat(Xdot, Sdot, Pdot, Vdot)
    
    #formulate stochastic ODE problem
    opts = {}
    ode = {"x": x_var, "p": ca.vertcat(par, u), "ode": xdot*dt, "quad": L}
    integrator = ca.integrator("integrator", integrator_casadi, ode, opts_integrator)
    
    return F, integrator_d, integrator, x_var, u, par

def model_mc(dt):
    
    # Assume parametric uncertainty
    # symbolic parameters - to use for the (stochastic) system
    mu_m  = ca.MX.sym("mu_m", 1)
    K_m	  = ca.MX.sym("K_m", 1)
    K_i	  = ca.MX.sym("K_i", 1)
    v_par = ca.MX.sym("v_par", 1)
    Y_p	  = ca.MX.sym("Y_p", 1)
    Y_x = ca.MX.sym("Y_x", 1)
    S_in = ca.MX.sym("S_in", 1)
    par = ca.vertcat(mu_m, K_m, K_i, v_par, Y_p, Y_x, S_in)
    
    #states
    X = ca.MX.sym("X", 1)
    S = ca.MX.sym("S", 1)
    P = ca.MX.sym("P", 1)
    V = ca.MX.sym("V", 1)
    x_var = ca.vertcat(X, S, P, V)
    
    #inputs
    u = ca.MX.sym("u", 1)
    

    #substrate inhibition - stochastic system    
    mu = mu_m*S/(K_m + S + (S**2/K_i))
    
    #ODE - stochastic system
    Xdot = mu*X - (u/V)*X
    Sdot = -mu*X/Y_x - v_par*X/Y_p + (u/V)*(S_in - S)
    Pdot = v_par*X - (u/V)*P
    Vdot = u
    xdot = ca.vertcat(Xdot, Sdot, Pdot, Vdot)
  
    #cost function. Add possibly soft constraints here
    L = -P
    
    #formulation for orthogonal collocation
    F = ca.Function("F", [x_var, u, par], [xdot, L], ["x_var", "u", "par"], ["xdot", "L"])
    
    
    #formulate stochastic ODE problem
    opts = {}
    ode = {"x": x_var, "p": ca.vertcat(par, u), "ode": xdot*dt, "quad": L}
    integrator = ca.integrator("integrator", "cvodes", ode, opts)
    
    return F, integrator, x_var, u, par

def model_scenario_tree(dt):
    #Make a model for the scenario tree MPC (5 certain parameters, 2 uncertain ones) and one model for the real system (7 uncertain/symbolic parameters)
    
    
    #deterministic parameters - to use for the MPC formulation
    mu_m_d  = 0.02
    K_m_d	  = 0.05
    K_i_d	  = 5.0
    v_par_d = 0.004
    Y_p_d	  = 1.2
    Y_x_st = ca.MX.sym("Y_x_st", 1)
    S_in_st = ca.MX.sym("S_in_st",1 )
    par_st = ca.vertcat(Y_x_st, S_in_st)
    
    #states
    X = ca.MX.sym("X", 1)
    S = ca.MX.sym("S", 1)
    P = ca.MX.sym("P", 1)
    V = ca.MX.sym("V", 1)
    x_var = ca.vertcat(X, S, P, V)
    
    #inputs
    u = ca.MX.sym("u", 1)
    
    #substrate inhibition
    mu_d = mu_m_d*S/(K_m_d + S + (S**2/K_i_d))
    
    #ODE - deterministic system (MPC)
    Xdot_d = mu_d*X - (u/V)*X
    Sdot_d = -mu_d*X/Y_x_st - v_par_d*X/Y_p_d + (u/V)*(S_in_st - S)
    Pdot_d = v_par_d*X - (u/V)*P
    Vdot_d = u
    xdot_d = ca.vertcat(Xdot_d, Sdot_d, Pdot_d, Vdot_d)
    
    #cost function. Add possibly soft constraints here
    L = -P
    
    #formulate deterministic ODE problem (MPC)
    opts_d = {}
    ode_d = {"x": x_var, "p": ca.vertcat(par_st, u), "ode": xdot_d*dt, "quad": L}
    integrator_d = ca.integrator("integrator_d", "cvodes", ode_d, opts_d)
    
    #formulation for orthogonal collocation
    F = ca.Function("F", [x_var, u, par_st], [xdot_d, L], ["x_var", "u", "par_st"], ["xdot", "L"])
    
    #we need a real/stochastic system. Assume parametric uncertainty
    # symbolic parameters - to use for the (stochastic) system
    mu_m  = ca.MX.sym("mu_m", 1)
    K_m	  = ca.MX.sym("K_m", 1)
    K_i	  = ca.MX.sym("K_i", 1)
    v_par = ca.MX.sym("v_par", 1)
    Y_p	  = ca.MX.sym("Y_p", 1)
    Y_x = ca.MX.sym("Y_x", 1)
    S_in = ca.MX.sym("S_in", 1)
    par = ca.vertcat(mu_m, K_m, K_i, v_par, Y_p, Y_x, S_in)

    #substrate inhibition - stochastic system    
    mu = mu_m*S/(K_m + S + (S**2/K_i))
    
    #ODE - stochastic system
    Xdot = mu*X - (u/V)*X
    Sdot = -mu*X/Y_x - v_par*X/Y_p + (u/V)*(S_in - S)
    Pdot = v_par*X - (u/V)*P
    Vdot = u
    xdot = ca.vertcat(Xdot, Sdot, Pdot, Vdot)
    
    #formulate stochastic ODE problem
    opts = {}
    ode = {"x": x_var, "p": ca.vertcat(par, u), "ode": xdot*dt, "quad": L}
    integrator = ca.integrator("integrator", "cvodes", ode, opts)
    
    return F, integrator_d, integrator, x_var, u, par, par_st

def get_parameters(std_dev_prct = .05):
    assert std_dev_prct < 1, "wrong input"
    # Certain parameters
    mu_m  = 0.02
    K_m	  = 0.05
    K_i	  = 5.0
    v_par = 0.004
    Y_p	  = 1.2
    Y_x = 0.5
    S_in = 200.
    par_nom = [mu_m, K_m, K_i, v_par, Y_p, Y_x, S_in]
    par_names = ["mu_m", "K_m", "K_i", "v_par", "Y_p", "Y_x", "S_in"]
    # par_fx = {key: val for key, val in zip(par_names, par_nom)}
    
    par_dist = {key: scipy.stats.norm(loc = val, scale = std_dev_prct*val) 
                for key, val in zip(par_names, par_nom)}
    return par_dist


def scen_param(par_a, par_b, w_a, w_b):
    
    scens = ca.vertcat([])  # param combinations
    scens_count = par_a.shape[0] * par_b.shape[0]
    weights = []
    for i in range(par_a.shape[0]):
        for j in range(par_b.shape[0]):
            scens = ca.vertcat(scens, ca.horzcat(par_a[i], par_b[j]))
            weights.append(w_a[i]*w_b[j])
    weights = np.array(weights)
    assert np.isclose(weights.sum(), 1), f"{weights.sum()=}, it should be close to 1"
    return scens, scens_count, weights

def delete_files_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
def has_abc_nmpc_converged(backoff_old, backoff_new, epsilon = 1e-6):
    if np.abs(backoff_new - backoff_old) < epsilon:
        return True
    else:
        return False
    
def resample_until_lb_fulfilled(dist_dict, size = 1, lb = -np.inf):
    assert isinstance(dist_dict, dict), "dist_dict must be a dict"
    assert isinstance(size, int), "size must be an int"
    assert isinstance(lb, (int, float)), "lb must be an int or float"
    ps = np.vstack([dist.rvs(size = size) for dist in dist_dict.values()])
    while (ps < lb).any().any(): #resample
        r, c = np.where(ps < lb)
        for ri, ci in zip(r, c):
            key = list(dist_dict.keys())[ri]
            ps[ri, ci] = dist_dict[key].rvs()
    assert ps.shape == (len(dist_dict), size), "output size not as expected"
    return ps
        
    
    
    
    
    
    
    