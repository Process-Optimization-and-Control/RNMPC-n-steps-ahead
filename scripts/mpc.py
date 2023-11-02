# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:36:45 2023

@author: halvorak
"""

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

"""
Contains "MPC-functions" which returns solver objects. Called from main-file. 

NB: some functions have headers describing input/outputs etc. These may be outdated.

"""

def collocation_points(d = 3):
    # Degree of interpolating polynomial
    
    # Get collocation points
    tau_root = np.append(0, ca.collocation_points(d, "radau"))
    
    # Coefficients of the collocation equation
    C = np.zeros((d+1,d+1))
    
    # Coefficients of the continuity equation
    D = np.zeros(d+1)
    
    # Coefficients of the quadrature function
    B = np.zeros(d+1)
    
    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
    
        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)
    
        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])
    
        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)
    return B, C, D
B, C, D = collocation_points(d=3)


def nlp_orthogonal_collocation_soft_constraints(F, dim_x, dim_u, dim_e, N, dt, d = 3, rho_sc = 1., opts_solver = {}):
    # opts_solver = {"print_time": 0, "ipopt": {"print_level": 0, "acceptable_tol": 1e-6, "linear_solver": "mumps"}}
    
    B, C, D = collocation_points(d = d) #d is number of collocation points
    
    #Time
    t0 = 0.
    
    x_lb = np.zeros(dim_x) #lower bounds for the states
    x_lb[1] = -.01 # subtrate
    
    #initial lifted/augmented states, symbolic variables
    xak = ca.MX.sym("xa_0", dim_x)
    xa = ca.vertcat(xak)
    
    #slack variables
    assert dim_e == (dim_x + 1), f"MPC is configured to have {1 + dim_x=} soft constraint, dim_e is {dim_e}" 
    ea = []
    backoff = ca.MX.sym("backoff", 1)
    
    #constraints on input change
    u_prev = ca.MX.sym("u_prev_0", dim_u)
    par = ca.vertcat(u_prev, backoff)
    
    #augmented input, symbolic variables
    ua = []
    
    #Initial value of cost function
    J = 0
    
    # inequality constraints
    g = []
    lbg = []
    ubg = []
    
    #inequality constraints - soft constraints
    g_sc = []
    lbg_sc = []
    ubg_sc = []
    
    # for obtaining the states given the solution
    x_trajectory = ca.vertcat(xak)
    u_trajectory = []
    
    #weight on input change
    R_du = np.array([[1.]])
    for i in range(N):
        #new control input
        uk = ca.MX.sym("u_" + str(i), dim_u)
        ua = ca.vertcat(ua, uk)
        
        #input change
        du = uk - u_prev
        
        #include in the u_trajectory
        u_trajectory = ca.vertcat(u_trajectory, uk)
        
        # State and slack variables at collocation points
        x_c = []
        # e_c = []
        for j in range(d):
            x_ij = ca.MX.sym('x_'+str(i)+'_'+str(j), dim_x)
            # e_ij = ca.MX.sym('e_'+str(i)+'_'+str(j), dim_e)
            x_c.append(x_ij)
            # e_c.append(e_ij)
        xa = ca.vertcat(xa, *x_c)
        # ea = ca.vertcat(ea, *e_c)
        
        # Loop over collocation points
        xak_end = D[0]*xak
        for j in range(1,d+1):
           # Expression for the state derivative at the collocation point
           xp = C[0,j]*xak
           for r in range(d): xp = xp + C[r+1,j]*x_c[r]
    
           # Append collocation equations
           t_current = t0 + i + j
           
           fj, qj = F(x_c[j-1], uk)
           g = ca.vertcat(g, dt*fj - xp)
           lbg = ca.vertcat(lbg, np.zeros(dim_x))
           ubg = ca.vertcat(ubg, np.zeros(dim_x))
    
           # Add contribution to the end state
           xak_end = xak_end + D[j]*x_c[j-1] 
           
           # Add contribution to quadrature function, without the soft constraint
           J = J + B[j]*qj*dt + du.T @ R_du @ du
        
        
        
        # New NLP variable for state at end of interval with its inquality bounds
        xak = ca.MX.sym('xa_' + str(i+1), dim_x)
        xa = ca.vertcat(xa, xak)
    
        #include in the x_trajectory
        x_trajectory = ca.vertcat(x_trajectory, xak)
        
        # Add equality constraint ("shooting constraint")
        g = ca.vertcat(g, xak_end - xak)
        lbg = ca.vertcat(lbg, np.zeros(dim_x))
        ubg = ca.vertcat(ubg, np.zeros(dim_x))
        
        #soft constraint only at the end of the (collocation) interval (not at each collocation point)
        ek = ca.MX.sym("e_" + str(i+1), dim_e)
        ea = ca.vertcat(ea, ek)
        g_sc = ca.vertcat(g_sc, (xak[0] - (3.7 - backoff)) - ek[0]) #biomass constraint
        g_sc = ca.vertcat(g_sc, (-(xak - x_lb) - ek[1:(1+dim_x)])) #non-negativity constraint on the states
        lbg_sc = ca.vertcat(lbg_sc, -np.inf*np.ones(dim_e))
        ubg_sc = ca.vertcat(ubg_sc, np.zeros(dim_e))
        J += (rho_sc*np.ones((1, dim_e))) @ ek
        
        #reset u_prev
        u_prev = uk*1
        
    #concatenate the decision variables
    w = ca.vertcat(xa, ua, ea)
    
    #concatenate nonlinear constraints
    g = ca.vertcat(g, g_sc)
    lbg = ca.vertcat(lbg, lbg_sc)
    ubg = ca.vertcat(ubg, ubg_sc)
    
    #define the NLP
    nlp = {"x": w, "f": J, "g": g, "p": par}
    S = ca.nlpsol("S", "ipopt", nlp, opts_solver)
    
    #define function to get back the trajectory given the lifted decision variables
    xa_mat = x_trajectory.reshape((dim_x, 1 + N))
    ua_mat = ua.reshape((dim_u, N))
    # ea_mat = ea.reshape((dim_e, N*d))
    ea_mat = ea.reshape((dim_e, N))
    trajectories = ca.Function("trajectories", [w], [xa_mat, ua_mat, ea_mat], ["w"], ["x_traj", "u_traj", "e_traj"])
    # trajectories = ca.Function("trajectories", [w], [xa_mat, ua_mat, ea], ["w"], ["x_traj", "u_traj", "ea"])
    
    #function to set initial guess
    w0_func = lambda x_guess, u_guess, e_guess: np.hstack((np.tile(x_guess, 1 + N*(d+1)), np.tile(u_guess, N), np.zeros(ea.shape[0])))

    return S, trajectories, w0_func, lbg, ubg, xa, ua, ea 


def nlp_monte_carlo_oc_soft_constraints(F, dist_samples, dist_mean, dim_x, dim_u, dim_e, N, dt, d = 3, par_sym = [], rho_sc = 1., opts_solver = {}):
    
    #dist_samples: samples from the disturbances/uncertain parameters which will be considered here
    N_mc, dim_d = dist_samples.shape
    assert dist_samples.ndim == 2, "Wrong dimension of dist_samples"
    assert N_mc > dim_d, f"dist_samples should be (N_mc, dim_d) = dist_samples.shape and N_mc > dim_d. Now we have dist_samples.shape = {dist_samples.shape}. Transpose the matrix?"
    assert dist_mean.ndim == 1, "dist_mean should be 1 dimensional np.array"
    assert dist_mean.shape[0] == dim_d, "Dimension mismatch between dist_samples and dist_mean"
    
    B, C, D = collocation_points(d = d) #d is number of collocation points
    
    #Time
    t0 = 0.
    
    #lower bounds for the states
    x_lb = np.zeros(dim_x) 
    x_lb[1] = -.01 # subtrate
    
    
    #constraints on input change
    u_prev = ca.MX.sym("u_prev_0", dim_u)
    par = ca.vertcat(u_prev)
    
    #augmented input, symbolic variables
    ua = []
    
    #initial lifted/augmented states, symbolic variables
    xak = ca.MX.sym("xa_0", dim_x)
    xa = ca.vertcat(xak)
    
    #slack variables
    assert dim_e == (dim_x + 1), f"MPC is configured to have {dim_x + 1=} soft constraint, dim_e is {dim_e}" 
    ea = []
    
    #Initial value of cost function
    J = 0
    
    # inequality constraints
    g = []
    lbg = []
    ubg = []
    
    # inequality constraints - soft constraints
    g_sc = []
    lbg_sc = []
    ubg_sc = []
    
    # for obtaining the states given the solution
    x_trajectory = ca.vertcat(xak)
    u_trajectory = []
    e_trajectory = []
    e_mc_trajectory = []
    x_mc_trajectory = []
    
    #weight on input change
    R_du = np.array([[1.]])
    for i in range(N):
        #new control input
        uk = ca.MX.sym("u_" + str(i), dim_u)
        ua = ca.vertcat(ua, uk)
        
        #input change
        du = uk - u_prev
        
        #include in the u_trajectory
        u_trajectory = ca.vertcat(u_trajectory, uk)
        
        if i == 0: #Monte Caro constraints - add g(x1_mck, u0) <= 0
            x1_mc = []
            e1_mc = []
            g_mc = []
            lbg_mc = []
            ubg_mc = []
            
            g1_mc_sc = []
            lbg1_mc_sc = []
            ubg1_mc_sc = []
            for n in range(N_mc):
                dist_val = dist_samples[n, :] #disturbance value to consider
                
                # State and slack variables at collocation points
                x_c = []
                # e_c = []
                for j in range(d):
                    x_ijn = ca.MX.sym('x_'+str(i)+'_'+str(j) + "_" + str(n), dim_x)
                    # e_ijn = ca.MX.sym('e_'+str(i)+'_'+str(j) + "_" + str(n), dim_e)
                    x_c.append(x_ijn)
                    # e_c.append(e_ijn)
                    
                x1_mc = ca.vertcat(x1_mc, *x_c)
                # e1_mc = ca.vertcat(e1_mc, *e_c)
                
                # Loop over collocation points
                xak_end = D[0]*xak
                for j in range(1,d+1):
                   # Expression for the state derivative at the collocation point
                   xp = C[0,j]*xak
                   for r in range(d): xp = xp + C[r+1,j]*x_c[r]
            
                   # Append collocation equations
                   t_current = t0 + i + j
                   
                   fj, qj = F(x_c[j-1],uk, dist_val)
                   g_mc = ca.vertcat(g_mc, dt*fj - xp)
                   lbg_mc = ca.vertcat(lbg_mc, np.zeros(dim_x))
                   ubg_mc = ca.vertcat(ubg_mc, np.zeros(dim_x))
            
                   # Add contribution to the end state
                   xak_end = xak_end + D[j]*x_c[j-1]
                   
                #Add soft constraints only on the end of the MC interval
                e1_mc_n = ca.MX.sym("e1_mc_" + str(n), dim_e) #slack variable
                e1_mc = ca.vertcat(e1_mc, e1_mc_n)
                g1_mc_sc = ca.vertcat(g1_mc_sc, (xak_end[0] - 3.7) - e1_mc_n[0]) #biomass constraint
                g1_mc_sc = ca.vertcat(g1_mc_sc, -(xak_end - x_lb) - e1_mc_n[1:(1+dim_x)]) #non-negativity constraint
                lbg1_mc_sc = ca.vertcat(lbg1_mc_sc, -np.inf*np.ones(dim_e))
                ubg1_mc_sc = ca.vertcat(ubg1_mc_sc, np.zeros(dim_e))
                
                #Add soft constraint contribution to cost function
                # J = J + (rho_sc*np.ones((1, dim_e))) @ e1_mc_n
                J = J + rho_sc*e1_mc_n[0]
                J = J + (rho_sc*np.ones((1, dim_e-1))) @ e1_mc_n[1:]
                
                x_mc_trajectory = ca.vertcat(x_mc_trajectory, xak_end)
                e_mc_trajectory = ca.vertcat(e_mc_trajectory, e1_mc_n)
        
        #Normal MPC for all time steps, including the first time step
        
        # State and slack variables at collocation points
        x_c = []
        for j in range(d):
            x_ij = ca.MX.sym('x_'+str(i)+'_'+str(j), dim_x)
            x_c.append(x_ij)
        xa = ca.vertcat(xa, *x_c)
        
        # Loop over collocation points
        xak_end = D[0]*xak
        for j in range(1,d+1):
           # Expression for the state derivative at the collocation point
           xp = C[0,j]*xak
           for r in range(d): xp = xp + C[r+1,j]*x_c[r]
    
           # Append collocation equations
           t_current = t0 + i + j
           
           fj, qj = F(x_c[j-1],uk, dist_mean)
           g = ca.vertcat(g, dt*fj - xp)
           lbg = ca.vertcat(lbg, np.zeros(dim_x))
           ubg = ca.vertcat(ubg, np.zeros(dim_x))
    
           # Add contribution to the end state
           xak_end = xak_end + D[j]*x_c[j-1]
           
    
           # Add contribution to quadrature function
           J = J + B[j]*qj*dt + du.T @ R_du @ du #+ rho_sc*e_c[j-1]
        
        # New NLP variable for state at end of interval with its inquality bounds
        xak = ca.MX.sym('xa_' + str(i+1), dim_x)
        xa = ca.vertcat(xa, xak)
    
        #include in the x_trajectory
        x_trajectory = ca.vertcat(x_trajectory, xak)
        
        # Add equality constraint ("shooting constraint")
        g = ca.vertcat(g, xak_end - xak)
        lbg = ca.vertcat(lbg, np.zeros(dim_x))
        ubg = ca.vertcat(ubg, np.zeros(dim_x))
        
        # Add soft constraints at the end of the collocation interval
        ek = ca.MX.sym("e_" + str(i+1), dim_e)
        ea = ca.vertcat(ea, ek)
        e_trajectory = ca.vertcat(e_trajectory, ek)
        g_sc = ca.vertcat(g_sc, (xak[0] - 3.7) - ek[0]) #biomass constraint
        g_sc = ca.vertcat(g_sc, (-(xak - x_lb) - ek[1:(1+dim_x)])) #non-negativity constraint on the states
        lbg_sc = ca.vertcat(lbg_sc, -np.inf*np.ones(dim_e))
        ubg_sc = ca.vertcat(ubg_sc, np.zeros(dim_e))
        J = J + (rho_sc*np.ones((1, dim_e))) @ ek

        #reset u_prev
        u_prev = uk*1
        
    #concatenate the decision variables
    w = ca.vertcat(xa, ua, x1_mc, ea, e1_mc)
    g = ca.vertcat(g, g_mc, g_sc, g1_mc_sc)
    lbg = ca.vertcat(lbg, lbg_mc, lbg_sc, lbg1_mc_sc)
    ubg = ca.vertcat(ubg, ubg_mc, ubg_sc, ubg1_mc_sc)
    
    #define the NLP
    nlp = {"x": w, "f": J, "g": g, "p": par}
    S = ca.nlpsol("S", "ipopt", nlp, opts_solver)
    
    #define function to get back the trajectory given the lifted decision variables
    xa_mat = x_trajectory.reshape((dim_x, 1 + N))
    ua_mat = ua.reshape((dim_u, N))
    x_mc_mat = x_mc_trajectory.reshape((dim_x, N_mc))
    ea_mat = ea.reshape((dim_e, N))
    e_mc_mat = e_mc_trajectory.reshape((dim_e, N_mc))
    trajectories = ca.Function("trajectories", [w], [xa_mat, ua_mat, x_mc_mat, ea_mat, e_mc_mat], ["w"], ["x_traj", "u_traj", "x_mc_traj", "e_traj", "e_mc_traj"])
    

    
    #function to set initial guess
    w0_func = lambda x_guess, u_guess: np.hstack((np.tile(x_guess, 1 + N*(d+1)), np.tile(u_guess, N), np.tile(x_guess, N_mc*(d)), np.zeros(ea.shape[0] + e1_mc.shape[0])))


    return S, trajectories, w0_func, lbg, ubg, xa, ua, x1_mc, ea, e1_mc



def nlp_orthogonal_collocation_scenario_tree_soft_constraints(F, dim_x, dim_u, dim_e, N, dt, par_scen, d = 3, rho_sc = 1., weights_scen = None, opts_solver = {}):
    """
    Sets up the NLP for scenario tree MPC

    Parameters
    ----------
    F : TYPE ca.Function
        DESCRIPTION. Function(F:(x_var,u,par_st)->(xdot,L) MXFunction). Returns xdot and cost function value
    dim_x : TYPE int
        DESCRIPTION. Number of state variables
    dim_u : TYPE int
        DESCRIPTION. Number of manipulated variables
    dim_e : TYPE int
        DESCRIPTION. Number of slack variables or the number of soft constraints
    N : TYPE int
        DESCRIPTION. Prediction horizon
    dt : TYPE float
        DESCRIPTION. Time step
    par_scen : TYPE np.array((dim_scen, dim_par_st))
        DESCRIPTION. Array of scenarios to consider
    d : TYPE, optional int
        DESCRIPTION. The default is 3. Degree of polynomial

    Returns
    -------
    S : TYPE ca.Function
        DESCRIPTION. Solver, NLP
    trajectories : TYPE not implemented
        DESCRIPTION.
    w0_func : TYPE not implemented
        DESCRIPTION.
    lbg : TYPE ca.DM
        DESCRIPTION. Lower bounds for nonlinear constraints
    ubg : TYPE ca.DM
        DESCRIPTION. Upper bounds for nonlinear constraints
    xa : TYPE ca.MX
        DESCRIPTION. Augmented/lifted x-variables
    ua : TYPE ca.MX
        DESCRIPTION. Augmented/lifted u-variables
    w : TYPE ca.MX
        DESCRIPTION. Combination of xa, ua
    nxs : TYPE int
        DESCRIPTION. Number of x-variables in each scenario (not including the initial point)
    nus : TYPE int
        DESCRIPTION. Number of u-variables in each scenario (not including the initial point/non-anticipativity constraints)

    """
    
    #get number of scenarios
    n_scen = par_scen.shape[0]
    
    if weights_scen is None:
        weights_scen = np.array([1/n_scen for i in range(n_scen)])
    assert (np.isclose(sum(weights_scen), 1)) or (sum(weights_scen) == n_scen), f"{sum(weights_scen)=}, should be close to 1 or {n_scen}"
    
    B, C, D = collocation_points(d = d) #d is number of collocation points
    
    #Time
    t0 = 0.
    
    #lower bounds for the states
    x_lb = np.zeros(dim_x) 
    x_lb[1] = -.01 # subtrate
    
    #constraints on input change
    u_prev0 = ca.MX.sym("u_prev_0", dim_u)
    par = ca.vertcat(u_prev0) #solver takes this as input
    
    #augmented input, symbolic variables
    ua = []
    u0 = ca.MX.sym("u_0", dim_u)
    ua = ca.vertcat(ua, u0)
    
    #initial lifted/augmented states, symbolic variables
    xa0 = ca.MX.sym("xa_0", dim_x)
    xa = ca.vertcat(xa0)
    
    #slack variables
    assert dim_e == (dim_x + 1), f"MPC is configured to have {dim_x + 1=} soft constraint, dim_e is {dim_e}" 
    ea = []
    
    #Initial value of cost function
    J = 0
    
    # inequality constraints
    g = []
    lbg = []
    ubg = []
    
    #inequality constraints - soft constraints
    g_sc = []
    lbg_sc = []
    ubg_sc = []
    
    # for obtaining the states given the solution - not implemented
    # x_trajectory = ca.vertcat(xa0)
    x_trajectory = []
    u_trajectory = []
    e_trajectory = []
    
    # #to help set initial guess for w0 later
    x0_traj = []
    u0_traj = []
    e0_traj = []
    
    #weight on input change
    R_du = np.array([[1.]])
    
    #all decision variables
    w = ca.vertcat(xa0, u0)
    # print(f"{w.shape=}")
    
    #build the scenario tree
    for s in range(n_scen):
        par_scenario = par_scen[s,:]
        assert par_scenario.shape[0] == 2
        assert par_scenario.ndim == 1, "not a numpy object"
        
        Js = 0 #cost for this scenario
        
        #starting values for each scenario. Variables are defined before the for-loop. This is kind of non-anticipativity constraints (I do not make a u0 for each scenario and constrain them to be equal. Instead, I make one single u0)
        xaks = xa0  
        uks = u0
        u_prev = u_prev0
        
        #make for each scenario, but don't include start point
        xas = []
        uas = []
        gs = []
        lbgs = []
        ubgs = []
        eas = [] #slack variables, soft constraint
        gs_sc = [] #inequality constraints, slack variables
        lbgs_sc = []
        ubgs_sc = []
        
        xs_traj = []
        us_traj = []
        es_traj = []
        
        for i in range(N):
            if not i==0:
                #new control input
                uks = ca.MX.sym("u_" + str(i) + "_" + str(s), dim_u)
                uas = ca.vertcat(uas, uks)
            #input change
            du = uks - u_prev
            
            # State variables at collocation points
            x_c = []
            for j in range(d):
                x_ijs = ca.MX.sym('x_'+str(i)+'_'+str(j) + "_" + str(s), dim_x)
                x_c.append(x_ijs)
            xas = ca.vertcat(xas, *x_c)
            
            # Loop over collocation points
            xak_end = D[0]*xaks
            for j in range(1,d+1):
               # Expression for the state derivative at the collocation point
               xp = C[0,j]*xaks
               for r in range(d): xp = xp + C[r+1,j]*x_c[r]
        
               # Append collocation equations
               t_current = t0 + i + j
               
               fj, qj = F(x_c[j-1],uks, par_scenario)
               g = ca.vertcat(g, dt*fj - xp)
               lbg = ca.vertcat(lbg, np.zeros(dim_x))
               ubg = ca.vertcat(ubg, np.zeros(dim_x))
        
               # Add contribution to the end state
               xak_end = xak_end + D[j]*x_c[j-1];
        
               # Add contribution to quadrature function
               Js = Js + B[j]*qj*dt + du.T @ R_du @ du
            
            # New NLP variable for state at end of interval with its inquality bounds
            xaks = ca.MX.sym('xa_' + str(i+1) + "_" + str(s), dim_x)
            xas = ca.vertcat(xas, xaks)
        
            
            # Add equality constraint ("shooting constraint")
            gs = ca.vertcat(gs, xak_end - xaks)
            lbgs = ca.vertcat(lbgs, np.zeros(dim_x))
            ubgs = ca.vertcat(ubgs, np.zeros(dim_x))
           
            # Add soft constraints at the end of the collocation interval
            eks = ca.MX.sym("e_" + str(i+1) + "_" + str(s), dim_e)
            eas = ca.vertcat(eas, eks)
            gs_sc = ca.vertcat(gs_sc, (xaks[0] - 3.7) - eks[0]) #biomass constraint
            gs_sc = ca.vertcat(gs_sc, -(xaks - x_lb) - eks[1:(1+dim_x)]) #non-negativity constraint on the states
            lbgs_sc = ca.vertcat(lbgs_sc, -np.inf*np.ones(dim_e))
            ubgs_sc = ca.vertcat(ubgs_sc, np.zeros(dim_e))
            Js = Js + (rho_sc*np.ones((1, dim_e))) @ eks
            
            #trajectories for plotting
            xs_traj = ca.vertcat(xs_traj, xaks)
            us_traj = ca.vertcat(us_traj, uks)
            es_traj = ca.vertcat(es_traj, eks)
            
            
            #reset u_prev
            u_prev = uks*1
        
        # Js = Js/n_scen
        
        #concatenate the "scenario variables"
        ws = ca.vertcat(xas, uas, eas)
        w = ca.vertcat(w, ws)
        
        
        #update total cost function and nonlinear constraints
        J += Js*weights_scen[s]
        g = ca.vertcat(g, gs, gs_sc)
        lbg = ca.vertcat(lbg, lbgs, lbgs_sc)
        ubg = ca.vertcat(ubg, ubgs, ubgs_sc)
        
        #trajectories for plotting
        xs_mat = xs_traj.reshape((dim_x, N))
        us_mat = us_traj.reshape((dim_u, N))
        es_mat = es_traj.reshape((dim_e, N))
        x_trajectory = ca.vertcat(x_trajectory, xs_mat)
        u_trajectory = ca.vertcat(u_trajectory, us_mat)
        e_trajectory = ca.vertcat(e_trajectory, es_mat)
        
        #trajectories for setting initial guesses
        assert (xas.shape[0] % dim_x) == 0, "Sth wrong, this should be an integer"
        assert (uas.shape[0] % dim_u) == 0, "Sth wrong, this should be an integer"
        assert (eas.shape[0] % dim_e) == 0, "Sth wrong, this should be an integer"
        nxs = int(xas.shape[0]/dim_x) #number of times x is repeated in each scenario
        nus = int(uas.shape[0]/dim_u)
        nes = int(eas.shape[0]/dim_e)

        x0s = xas.reshape((dim_x, nxs))
        u0s = uas.reshape((dim_u, nus))
        e0s = eas.reshape((dim_e, nes))
        x0_traj = ca.vertcat(x0_traj, x0s)
        u0_traj = ca.vertcat(u0_traj, u0s)
        e0_traj = ca.vertcat(e0_traj, e0s)
        
        
        
        
    #figure out how many variables there are of each kind for the scenario. Used later to define initial guess and lower/upper bounds
    
    #define the NLP
    nlp = {"x": w, "f": J, "g": g, "p": par}
    S = ca.nlpsol("S", "ipopt", nlp, opts_solver)
    
    #These are not implemented
    trajectories = ca.Function("trajectories", [w], [x_trajectory, u_trajectory, e_trajectory])

    
    w2x0traj = ca.Function("w2x0", [w], [x0_traj])
    w0_func = None
    
    return S, trajectories, w0_func, lbg, ubg, xa, ua, w, nxs, nus, nes, w2x0traj

    
def get_lbw_ubw_orthogonal_collocation(x_current, lbx, ubx, lbu, ubu, n_horizon, d, dim_e = 0):
    lbx_a = np.tile(lbx, n_horizon*(d+1))
    ubx_a = np.tile(ubx, n_horizon*(d+1))
    lbu_a = np.tile(lbu, n_horizon)
    ubu_a = np.tile(ubu, n_horizon)
    
    # lbw_a = np.hstack((x_current, lbx_a, u_current, lbu_a))
    # ubw_a = np.hstack((x_current, ubx_a, u_current, ubu_a))
    if dim_e > 0:
        lbe_a = np.tile(np.zeros(dim_e), n_horizon)
        ube_a = np.tile(np.inf*np.ones(dim_e), n_horizon)
        lbw_a = np.hstack((x_current, lbx_a, lbu_a, lbe_a))
        ubw_a = np.hstack((x_current, ubx_a, ubu_a, ube_a))
    else:
        lbw_a = np.hstack((x_current, lbx_a, lbu_a))
        ubw_a = np.hstack((x_current, ubx_a, ubu_a))
    
    return lbw_a, ubw_a

def get_lbw_ubw_mc_mpc_sc(x_current, lbx, ubx, lbu, ubu, lbe, ube, n_horizon, d, N_mc):
    lbx_a = np.tile(lbx, n_horizon*(d+1))
    ubx_a = np.tile(ubx, n_horizon*(d+1))
    lbu_a = np.tile(lbu, n_horizon)
    ubu_a = np.tile(ubu, n_horizon)
    lbx_mc = np.tile(lbx, N_mc*(d))
    ubx_mc = np.tile(ubx, N_mc*(d))
    
    lbe_a = np.tile(lbe, n_horizon)
    ube_a = np.tile(ube, n_horizon)
    lbe_mc = np.tile(lbe, N_mc)
    ube_mc = np.tile(ube, N_mc)
    
    lbw_a = np.hstack((x_current, lbx_a, lbu_a, lbx_mc, lbe_a, lbe_mc))
    ubw_a = np.hstack((x_current, ubx_a, ubu_a, ubx_mc, ube_a, ube_mc))
    
    return lbw_a, ubw_a

def set_w_scenario_tree_sc(x_val, u_val, e_val, nxs, nus, nes, n_scen):
    """
    Sets x_val and u_val in correct order with respect to w in scenario tree MPC formulation

    Parameters
    ----------
    x_val : TYPE np.array(dim_x,)
        DESCRIPTION. x-values to set
    u_val : TYPE np.array(dim_u,)
        DESCRIPTION. u-values to set
    e_val : TYPE np.array(dim_e,)
        DESCRIPTION. e-values (slack variables) to set
    nxs : TYPE int
        DESCRIPTION. Number of x-values in each scenario (not including initial point)
    nus : TYPE int
        DESCRIPTION. Number of u-values in each scenario (not including initial point)
    n_scen : TYPE int
        DESCRIPTION. Total number of scenarios

    Returns
    -------
    w_val : TYPE np.array(dim_w,)
        DESCRIPTION. w-values in correct order (x-val and u-val stacked together correctly)

    """
    
    #initial guess, or lower/upper bounds
    xs_val = np.tile(x_val, nxs)
    us_val = np.tile(u_val, nus)
    es_val = np.tile(e_val, nes)
    ws_val = np.hstack((xs_val, us_val, es_val))
    w_val = np.tile(ws_val, n_scen)
    w_val = np.hstack((x_val, u_val, w_val)) 
    return w_val
