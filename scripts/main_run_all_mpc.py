# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:20:40 2023

@author: halvorak
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 20:34:41 2023

@author: halvorak
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
# import matplotlib.lines
import pandas as pd
import pathlib
import os
import casadi as ca
import copy
import time
import scipy.stats.qmc
import seaborn as sns
import pickle

import utils
import mpc

"""
main file which runs all the simulations. There are parameter/disturbance sequences for each of the 100 simulations are saved in a separate file in the directory "dir_par" (a variable name). "sim_range = range(100)" (default) gives numbers which acceses each file containing disturbance sequence. Each controller in the list "mpc_implemented" runs on all simulations. The results are saved in files, and can be plotted by using the files plot_results.py or plot_results_compare_solvers.py.

NB: MS-NMPC behaved "strange" for the simulation number in "sim_range_msnmpc". We changed our initial guesses for the optimization variables for the ms-nmpc for that simulation number.

NBB: We have also supported parallel MC simulations (set n_threads > 1) and LHS (set use_LHS=True), although these results are not included in the paper.

NBBB: The code may not be very user-friendly to read (I did not have time to clean up)
"""

X = 0
S = 1
P = 2
V = 3

font = {'size': 12}
matplotlib.rc('font', **font)
#%%Input
sim_range = range(1) #simulations to run
N_sim = len(sim_range) #number of times we repeat the simulation
sim_range_msnmpc = [47] #simulations where initial guess is w0_method="w_opt_clipping" in ms-nmpc

use_LHS = False #if False, random sampling is done
N_mc_abc = int(1e3) #number of MC simulations at each time step in the ABC-MPC
N_mc_mcms = int(1e2) #number of MC simulations at each time step in the MCSS-MPC
n_threads = int(1) #If 1, evaluate MC sim in serial. Higher number ==> run in parallell with specified number of threads
rho_sc = 1e5 #weight for the slack variables in the cost function, only for soft constraints
n_horizon = 25 # prediction horizon for all nmpc methods
integrator_ca = "cvodes"
par_samples = "rvs"
# par_samples = "st"
backoff_iter_max = 1 #only used for abc-nmpc - also known as i_max in the paper

mpc_implemented = ["nmpc", "abc-nmpc", "mcms-nmpc", "ms-nmpc"]
# mpc_implemented = [mpc_implemented[1]] #select a specific mpc to run
w0_implemented = ["w_opt_clipping", "w_opt_copy"]
w0_method = w0_implemented[-1]
opts_solver = {"print_time": 0, "ipopt": {"print_level": 0, "acceptable_tol": 1e-6, "linear_solver": "mumps"}}
# opts_solver = {"print_time": 0, "ipopt": {"print_level": 0, "acceptable_tol": 1e-6, "linear_solver": "ma57"}}
ipopt_not_converged = []

for mpc_type in mpc_implemented:
    print(f"Running {mpc_type=}")
    if mpc_type == "mcms-nmpc":
        N_mc = N_mc_mcms
    else:
        N_mc = N_mc_abc
    #%% Def directories
    project_dir = pathlib.Path(__file__).parent.parent
    dir_data = os.path.join(project_dir, "data")
    dir_plots = os.path.join(project_dir, "plots")
    dir_par = os.path.join(dir_data, "par_samples", par_samples)
    
    dir_res = os.path.join(dir_data, "res")
    dir_res_rvs = os.path.join(dir_res, "rvs")
    dir_res_st = os.path.join(dir_res, "st")
    
    if par_samples == "rvs":
        dir_res_mpc = os.path.join(dir_res_rvs, mpc_type)
    elif par_samples == "st":
        dir_res_mpc = os.path.join(dir_res_st, mpc_type)
    if not os.path.exists(dir_res_mpc):
        os.mkdir(dir_res_mpc)
    utils.delete_files_in_folder(dir_res_mpc) #clean directory from previous results
    
    #%% Time
    t = np.arange(150)
    dim_t = t.shape[0]
    
    dt = t[1]-t[0]
    
    #%% Casadi func & integrator
    if (integrator_ca == "cvodes") or (integrator_ca == "idas"):
        opts_integrator = {"abstol": 1e-7, 
                           "reltol": 1e-6,
                           "linear_solver": "csparse",
                           "max_num_steps": int(1e5)}
    elif integrator_ca == "rk":
        opts_integrator = {"number_of_finite_elements": 20}
    elif integrator_ca == "collocation":
        opts_integrator = {"collocation_scheme": "radau",
                           "interpolation_order": 3,
                           "number_of_finite_elements": 1}
    else:
        raise ValueError(f"{integrator_ca=} is not a valid integration method")
        
    
    F, _, integrator_sys, x_var, u, par_sym = utils.model(dt, integrator_casadi = integrator_ca, opts_integrator= opts_integrator)
    
    #create a map function which can evaluate the IVP in parallell.
    F_int = ca.Function("F_int", [x_var, u, par_sym], [integrator_sys(x0 = x_var, p = ca.vertcat(par_sym, u))["xf"]])
    if n_threads == 1:
        int_map = F_int.map(N_mc) #serial evaluation
    elif n_threads > 1:
        int_map = F_int.map(N_mc, "thread", n_threads) #parallell evaluation
    else:
        raise ValueError(f"Wrong input, {n_threads=} and it must be a positive integer")
    #%% Dimensions
    dim_x = x_var.shape[0]
    dim_u = u.shape[0]
    dim_par = par_sym.shape[0]
    
    #%% Initial conditions and parameters
    x0 = np.array([1., .5, 0.0, 120])
    assert x0.shape[0] == dim_x, f"Dimension mismatch between casadi model and x0. Have from casadi dim_x = {dim_x} while x0 has shape {x0.shape[0]}"
    u_start = np.array([1e-10])
    u_guess = np.ones(dim_u)*u_start
    std_dev_prct = float(np.load(os.path.join(dir_par, "std_dev_prct.npy")))
    par_dist = utils.get_parameters(std_dev_prct)
    par_mean = [dist.mean() for dist in par_dist.values()]
    par_names = list(par_dist.keys())
    #%%  constraints
    x_min = np.ones(dim_x)*(-np.inf)
    x_max = np.ones(dim_x)*np.inf
    
    x_min_true = np.zeros(dim_x)
    x_min_true[1] = -.01 #substrate
    x_max_true = x_max.copy()
    x_max_true[X] = 3.7 #for plotting
    
    u_min = np.array([.0])
    u_max = np.array([.2])
    
    #%% MPC definition
    
    d = 3 #number of collocation points
    dim_e = 1 + dim_x #number of soft constraints
    if (mpc_type == "nmpc" or mpc_type == "abc-nmpc"):
        S, trajectories, w0_func, lbg, ubg, xa, ua, ea = mpc.nlp_orthogonal_collocation_soft_constraints(F, dim_x, dim_u, dim_e, n_horizon, dt, d = d, rho_sc = rho_sc, opts_solver = opts_solver)
        
        w0 = w0_func(x0, u_guess, 0.) #initial guess for the MPC
        
    elif mpc_type == "ms-nmpc":
        #scenarios to consider
        yx_st = par_dist["Y_x"].mean() + np.array([-2*par_dist["Y_x"].std(), 0., 2*par_dist["Y_x"].std()])
        sin_st = par_dist["S_in"].mean() + np.array([-2*par_dist["S_in"].std(), 0., 2*par_dist["S_in"].std()])
        w_st_method = "uniform"
        if w_st_method == "pdf":#weights based on the pdf-value
            w_yx_st = par_dist["Y_x"].pdf(yx_st)/sum(par_dist["Y_x"].pdf(yx_st)) 
            w_sin_st = par_dist["S_in"].pdf(sin_st)/sum(par_dist["S_in"].pdf(sin_st))
        elif w_st_method == "cdf":
            w_yx_st = [.025, .95, .025]
            w_sin_st = [.025, .95, .025]
        elif w_st_method == "uniform":
            w_yx_st = [1/3, 1/3, 1/3]
            w_sin_st = [1/3, 1/3, 1/3]
        elif w_st_method == "unity":
            w_yx_st = [1., 1., 1.]
            w_sin_st = [1., 1., 1.]
        else:
            raise ValueError(f"{w_st_method=} is not implemented")
        print(f"{w_st_method=}")
         
        par_scen, n_scen, weights_scen = utils.scen_param(yx_st, sin_st, w_yx_st, w_sin_st)
    
        par_scen = np.array(par_scen) #need it to be a numpy array in the mpc-setup
        
        F_ms_nmpc, integrator_ms, _, x_var, u, par_sym, par_st = utils.model_scenario_tree(dt)
        
        (S, trajectories, w0_func, 
         lbg, ubg, xa, ua, w, nxs, nus, nes, w2x0traj) = mpc.nlp_orthogonal_collocation_scenario_tree_soft_constraints(F_ms_nmpc, dim_x, dim_u, dim_e, n_horizon, dt, par_scen, d = d, rho_sc = rho_sc, weights_scen = weights_scen, opts_solver = opts_solver) #make the NLP
        
        #get lower and upper bounds for the scenario tree mpc. The first dim_x values should be updated in the for-loop, the rest are constant.
        e_min = np.zeros(dim_e)
        e_max = np.ones(dim_e)*(np.inf)
        lbw_nom = mpc.set_w_scenario_tree_sc(x_min, u_min, e_min, nxs, nus, nes, n_scen)
        ubw_nom = mpc.set_w_scenario_tree_sc(x_max, u_max, e_max, nxs, nus, nes, n_scen)
    
        
    elif mpc_type == "mcms-nmpc":
        F_mcms, _, x_var, u, par_sym = utils.model_mc(dt)
        #Random samples from the distribution
        # dist_samples = np.array([dist.rvs(size = N_mc) for key, dist in par_dist.items()]).T
        dist_samples = utils.resample_until_lb_fulfilled(par_dist, size = N_mc, lb = 0.).T #ensure (dist_samples>0).all()
    
        print("Constructing the NLP..")
        e_min = np.zeros(dim_e)
        e_max = np.ones(dim_e)*(np.inf)
        
        S, trajectories, w0_func, lbg, ubg, xa, ua, x1_mc, ea, e1_mc = mpc.nlp_monte_carlo_oc_soft_constraints(F_mcms, dist_samples, np.array(par_mean), dim_x, dim_u, dim_e, n_horizon, dt, d = d, rho_sc = rho_sc, opts_solver = opts_solver)

        lbw_nom, ubw_nom = mpc.get_lbw_ubw_mc_mpc_sc(x0, x_min, x_max, u_min, u_max, e_min, e_max, n_horizon, d, N_mc)
        dim_opt_var = xa.shape[0] + ua.shape[0] + x1_mc.shape[0] + ea.shape[0] + e1_mc.shape[0]
        print(f"NLP is made. Number of optimization variables are {dim_opt_var}, the number of MC var in the NLP is {x1_mc.shape[0]} (={x1_mc.shape[0]/dim_opt_var*100: .2f}%), number of slack variables are {e1_mc.shape[0] + ea.shape[0]} (={(e1_mc.shape[0] + ea.shape[0])/dim_opt_var*100: .2f}%)")
    
        w0 = w0_func(x0, u_guess) #initial guess for the MPC
        
        
    else:
        raise ValueError("Wrong input")
    #%% MPC sampling function
    if mpc_type == "abc-nmpc":
        sampler = scipy.stats.qmc.LatinHypercube(d = dim_par)
        samples_cdf = sampler.random(N_mc).T
        ps = np.vstack([dist.ppf(si_cdf) for dist, si_cdf in zip(par_dist.values(), samples_cdf)])
    
    for ns in sim_range:
        #%%Init matrices
        x_true = np.zeros((dim_x, dim_t))
        x_true[:, 0] = x0
        u_used = np.zeros((dim_u, dim_t))
        u_used[:, 0] = u_start
        t_iter_nlp = np.zeros(dim_t) #keep track of the required time to solve the nlp for each time-step
        t_iter_nlp[0] = np.nan
        
        df_par = pd.read_pickle(os.path.join(dir_par, "df_par_sim_" + str(ns) + ".pkl"))
        
        if True: #can be deleted for speed
            x_pred = np.zeros((dim_t, dim_x, n_horizon + 1)) #includes "initial" point
            u_pred = np.zeros((dim_t, dim_u, n_horizon))
            cost_func = np.zeros((dim_t))
            
            #Monte carlo matrix used for calculating the back-off
            x_mc = np.zeros((dim_t, dim_x, N_mc))
            x_max_mc_hist = np.zeros((dim_x, dim_t))
            x_max_mc_hist[:, :2] = np.nan #the first two values are not used
        
        if (mpc_type == "nmpc" or mpc_type == "abc-nmpc"):
            lbw_nom, ubw_nom = mpc.get_lbw_ubw_orthogonal_collocation(x_true[:, 0], x_min, x_max, u_min, u_max, n_horizon, d, dim_e = dim_e)
        #%%Run sim
        w_opt = np.nan
        ts = time.time()
        
        for i in range(1, dim_t):
            if i == 3:
                print(f"{ns=} and {w0_method=}")
            
            par_sample_system = df_par.iloc[i,:].to_numpy()
            
            #solve the system
            sol_int = integrator_sys(x0 = x_true[:, i-1], p = np.hstack((par_sample_system, u_used[:, i-1])))
            x_true[:, i] = np.array(sol_int["xf"]).flatten()
            x_true[:, i] = np.maximum(1e-8, x_true[:, i]) #ensure positive x, avoid integration error
            
            #solve the mpc
            t_sol_start = time.time()
            #%% abc-nmpc 
            if mpc_type == "abc-nmpc":
                #get initial guess
                if i == 1:
                    w0 = w0_func(x_true[:, i], u_used[:, i-1], 0.)
                    
                    #initial box constraints
                    lbw = lbw_nom.copy()
                    ubw = ubw_nom.copy()
                    
                else:
                    w0 = w_opt.copy()
                    w0[:dim_x] = x_true[:, i]
                
                
                backoff_sys = 0. #always assume zero back-off initially
                
                lbw[:dim_x] = x_true[:, i]
                ubw[:dim_x] = x_true[:, i]
                
                converged_abc_nmpc = False
                backoff_iter = 0
                
                while not converged_abc_nmpc:
                    
                
                    sol = S(x0 = w0, lbg = lbg, ubg = ubg, lbx = lbw, ubx = ubw, p = np.hstack((u_used[:, i-1], backoff_sys)))
                    if not S.stats()["success"]:
                        dict_fail = {"mpc_type": mpc_type,
                                     "ti": i,
                                     "ns": ns,
                                     "S_stats": copy.deepcopy(S.stats())}
                        ipopt_not_converged.append(dict_fail)
                                                              
                        print(f"{S.stats()['success']=}, {S.stats()['return_status']=} with x0 = {x_true[:,i]} at time {i}, ns={ns+1}/{N_sim} for {mpc_type=}, hopefully it recovers..")
                    
                    #extract solution and convert to numpy
                    w_opt = np.array(sol["x"]).flatten()
                    x_opt, u_opt, e_opt = trajectories(w_opt)
                    u_opt = np.array(u_opt) #predicted control trajectory
                    
                    #control input based on current state value and nominal constraints
                    uk = u_opt[:, 0]
                    
                    if backoff_iter >= backoff_iter_max:
                        converged_abc_nmpc = True
                        # print(f"{backoff_iter=} (maximum value) at {i=}")
                        break
                    
                    #sample N_mc random parameters and run Monte Carlo simulations on the system
                    while True: #if integrator fails, repeat with new samples until success
                        iter_ua = 0
                        try:
                            if use_LHS:
                                samples_cdf = sampler.random(N_mc).T
                                ps = np.vstack([dist.ppf(si_cdf) for dist, si_cdf in zip(par_dist.values(), samples_cdf)])
                            else: #random sampling
                                ps = utils.resample_until_lb_fulfilled(par_dist, size = N_mc, lb = 0.) #ensure (ps>0).all()
                                
                            #run MC prediction
                            if True: #run with map construct (parallell is possible here)
                                x_mc[i, :, :] = int_map(ca.repmat(x_true[:, i], 1, N_mc),
                                                         ca.repmat(uk, 1, N_mc),
                                                         ps)
                            else: #serial, normal way
                                for ni in range(N_mc):
                                    sol_mc = integrator_sys(x0 = x_true[:, i], p = np.hstack((ps[ni, :], uk)))
                                    x_mc[i, :, ni] = np.array(sol_mc["xf"]).flatten()
                            break #integrator worked, break out of the while loop
                        except RuntimeError as err:
                            print("Run time error in UA. Trying with a different sample")
                            if iter_ua >= 10:
                                raise err(f"{iter_ua=} and still we did not converge")
                            iter_ua += 1
                            
                    #calculate the required back-off
                    x_max_mc = np.max(x_mc[i, :, :], axis = 1) #max value of each state for all MC predictions.
                    x_max_backoff = np.maximum(0., x_max_mc - x_max_true) #element wise maximum (0 is broadcasted to correct shape)
                    # x_max_backoff = np.maximum(backoff_sys, x_max_mc - x_max_true) #element wise maximum
                    backoff_calc = backoff_sys + x_max_backoff[0] #we only have constraints on the zeroth variable for this system
                    
                    #check for convergence
                    converged_abc_nmpc = utils.has_abc_nmpc_converged(backoff_sys, backoff_calc, epsilon = 1e-6)
                    # print(f"{i=}, {backoff_iter=}, {backoff_sys=}, {x_max_backoff[0]=}, {backoff_calc=}, {converged_abc_nmpc=}")
                    backoff_sys = backoff_calc
                        
                    backoff_iter += 1
                    
                #save the new (soft) limits
                x_max_mpc = x_max_true - x_max_backoff
                x_max_mc_hist[:, i] = x_max_mpc
                
                x_opt = np.array(x_opt) #predicted state trajectory, including initial state at position x_opt[:, 0]
                u_opt = np.array(u_opt) #predicted control trajectory
                uk = u_opt[:, 0]
                    
                #select new control input
                u_used[:, i] = uk
                
                #save predicted values
                x_pred[i, :, :] = x_opt
                u_pred[i, :, :] = u_opt
                #save solution variables
                cost_func[i] = float(sol["f"])
                
    
            #%%nmpc    
            elif mpc_type == "nmpc": #normal MPC
                #get initial guess
                if i == 1:
                    w0 = w0_func(x_true[:, i], u_used[:, i-1], 0.)
                    
                    #initial box constraints
                    lbw = lbw_nom.copy()
                    ubw = ubw_nom.copy()
                    
                else:
                    w0 = w_opt.copy()
                    w0[:dim_x] = x_true[:, i]
                    
                backoff_sys = 0. #always have zero back-off for normal MPC
                
                #update box constraints - the first dim_x are equality constraints for the new initial conditions
                lbw[:dim_x] = x_true[:, i]
                ubw[:dim_x] = x_true[:, i]
                
                
                #solve the nlp
                sol = S(x0 = w0, lbg = lbg, ubg = ubg, lbx = lbw, ubx = ubw, p = np.hstack((u_used[:, i-1], backoff_sys)))
                
                if not S.stats()["success"]:
                    dict_fail = {"mpc_type": mpc_type,
                                 "ti": i,
                                 "ns": ns,
                                 "S_stats": copy.deepcopy(S.stats())}
                    ipopt_not_converged.append(dict_fail)
                    print(f"{S.stats()['success']=}, {S.stats()['return_status']=} with x0 = {x_true[:,i]} at time {i}, ns={ns+1}/{N_sim} for {mpc_type=}, hopefully it recovers..")
                
                #extract solution and convert to numpy
                w_opt = np.array(sol["x"]).flatten()
                x_opt, u_opt, e_opt = trajectories(w_opt)
                x_opt = np.array(x_opt) #predicted state trajectory, including initial state at position x_opt[:, 0]
                u_opt = np.array(u_opt) #predicted control trajectory
                
                #select new control input
                u_used[:, i] = u_opt[:, 0]
                
                #save predicted values
                x_pred[i, :, :] = x_opt
                u_pred[i, :, :] = u_opt
                
                #save solution variables
                cost_func[i] = float(sol["f"])
            
            #%%ms-nmpc
            elif mpc_type == "ms-nmpc":
                #get initial guess
                if i == 1:
                    w0 = mpc.set_w_scenario_tree_sc(x0, u_start, np.zeros(dim_e), nxs, nus, nes, n_scen)
                    
                    #initial box constraints
                    lbw = lbw_nom.copy()
                    ubw = ubw_nom.copy()
                    
                else:
                    if ns in sim_range_msnmpc:
                        w0_method = w0_implemented[0] #clipping
                    else:
                        w0_method = w0_implemented[-1] #w_opt_copy
                    
                    if w0_method=="w_opt_clipping":
                        #good initial guess - i) set slack variables e to zero ii) biomass at maxium 3.7 iii) substrate to be minimum 0 and iv) u0 to be away from extreme values (especially u_max) in the initial guess for the NLP.
                        
                        e0_traj = np.zeros(e_opt_traj2.shape)
                        x0_traj_ca = np.array(w2x0traj(w_opt)) #casadis format
                        x0_traj = x0_traj_ca.reshape((n_scen, dim_x, nxs))
                        u0_traj = np.zeros(u_opt_traj2.shape)
                        for nsi in range(n_scen):
                            x0_traj[nsi, 0, :] = np.minimum(x_max_true[0]-1e-3, x0_traj[4, 0, :]) #nominal trajectory for biomass
                            x0_traj[nsi, 1, :] = np.maximum(1e-10, x0_traj[nsi, 1, :]) #substrate xlb
                            u0_traj[nsi,0,:] = np.clip(u_opt_traj2[nsi,0,:], 1e-3, 4e-3)
                        
                        #build w0 in similar manner as w when we created the solver S in mpc.nlp_orthogonal_collocation_scenario_tree_soft_constraints()
                        u0 = u_opt.flatten()
                        w0 = np.hstack([x_true[:,i].flatten(order="F"), u0])
                        for nsi in range(n_scen):
                            w0 = np.hstack((w0, x0_traj[nsi,:,:].flatten(order="F"), u0_traj[nsi,:,1:].flatten(order="F"), e0_traj[nsi,:,:].flatten(order="F")))

                    elif w0_method == "w_opt_copy":
                        w0 = w_opt.copy()
                        w0[:dim_x] = x_true[:, i]
                    else:
                        raise KeyError(f"{w0_method=} is not implemented. Valid methods are {w0_implemented}")
                #update box constraints - the first dim_x are equality constraints for the new initial conditions
                lbw[:dim_x] = x_true[:, i]
                ubw[:dim_x] = x_true[:, i]
                
                #solve the nlp
                sol = S(x0 = w0, lbg = lbg, ubg = ubg, lbx = lbw, ubx = ubw, 
                        p = u_used[:, i-1] #used to calculate du for the first time step
                        )
                
                if not S.stats()["success"]:
                    dict_fail = {"mpc_type": mpc_type,
                                 "ti": i,
                                 "ns": ns,
                                 "S_stats": copy.deepcopy(S.stats())}
                    ipopt_not_converged.append(dict_fail)
                    print(f"{S.stats()['success']=}, {S.stats()['return_status']=} with x0 = {x_true[:,i]} at time {i}, ns={ns+1}/{N_sim} for {mpc_type=}, hopefully it recovers..")
                    # raise ValueError("Solver not successful")
                #extract solution and convert to numpy
                w_opt = np.array(sol["x"]).flatten()
                u_opt = w_opt[dim_x: dim_x + dim_u]
                
                # x_opt_traj, u_opt_traj, e_opt_traj = trajectories(w_opt)
                x_opt_traj, u_opt_traj, e_opt_traj = trajectories(w_opt)
                x_opt_traj = np.array(x_opt_traj)
                u_opt_traj = np.array(u_opt_traj)
                e_opt_traj = np.array(e_opt_traj)
                
                x_opt_traj2 = x_opt_traj.reshape((n_scen, dim_x, n_horizon))
                u_opt_traj2 = u_opt_traj.reshape((n_scen, dim_u, n_horizon))
                e_opt_traj2 = e_opt_traj.reshape((n_scen, dim_e, n_horizon))
                
                cost_func[i] = float(sol["f"])
                
                #select new control input
                u_used[:, i] = u_opt
            #%%mcms-nmpc
            elif mpc_type == "mcms-nmpc":
                #get initial guess
                if i == 1:
                    w0 = w0_func(x_true[:, i], u_used[:, i-1])
                    
                    #initial box constraints
                    lbw = lbw_nom.copy()
                    ubw = ubw_nom.copy()
                    
                else:
                    w0 = w_opt.copy()
                    w0[:dim_x] = x_true[:, i]
                    w0[-(e1_mc.shape[0] + ea.shape[0]):] = .0
                
                #update box constraints - the first dim_x are equality constraints for the new initial conditions
                lbw[:dim_x] = x_true[:, i]
                ubw[:dim_x] = x_true[:, i]
    
                #solve the nlp
                sol = S(x0 = w0, lbg = lbg, ubg = ubg, lbx = lbw, ubx = ubw, p = u_used[:, i-1])
                if not S.stats()["success"]:
                    dict_fail = {"mpc_type": mpc_type,
                                 "ti": i,
                                 "ns": ns,
                                 "S_stats": copy.deepcopy(S.stats())}
                    ipopt_not_converged.append(dict_fail)
                    print(f"{S.stats()['success']=}, {S.stats()['return_status']=} with x0 = {x_true[:,i]} at time {i}, ns={ns+1}/{N_sim} for {mpc_type=}, hopefully it recovers..")
                
                #extract solution and convert to numpy
                w_opt = np.array(sol["x"]).flatten()
                x_opt, u_opt, x_mc_opt, e_opt, e_mc_opt = trajectories(w_opt)
                e_opt = np.array(e_opt)
                e_mc_opt = np.array(e_mc_opt)
                x_opt = np.array(x_opt) #predicted state trajectory, including initial state at position x_opt[:, 0]
                u_opt = np.array(u_opt) #predicted control trajectory
                x_mc_opt = np.array(x_mc_opt) #predicted Monte Carlo trajectory
                
                #select new control input
                u_used[:, i] = u_opt[:, 0]
                
                #save predicted values
                x_pred[i, :, :] = x_opt
                u_pred[i, :, :] = u_opt
                x_mc[i, :, :] = x_mc_opt
                
                #save solution variables
                cost_func[i] = float(sol["f"])
                
            else:
                raise IndexError("Not implemented yet")

            t_iter_nlp[i] = time.time() - t_sol_start
            if ((i%30)==0):
                print(f"Iter {ns+1}/{N_sim} and timestep {i}/{dim_t}. {x_true[:2,i]=}")
        #%%Save results
        sim_time = time.time() - ts
        print(f"Iter {ns+1}/{N_sim} completed, {sim_time=:.1f}s = {sim_time/60:.1f} min and {x_true[1,-1]= :.2f}")
        
        df_true = pd.DataFrame(data = x_true.T, columns = [x_var[j].name() for j in range(dim_x)])
        df_true["u"] = u_used.T
        df_true["sim_time"] = sim_time #add simulation time
        df_true["t_iter_nlp"] = t_iter_nlp #time to solve the NLP for each time step in the simulation
        df_true["N_mc"] = N_mc 
        df_true["LHS"] = use_LHS 
        df_true["n_horizon"] = n_horizon
        df_true["w0_method"] = w0_method
        df_true["linear_solver"] = opts_solver["ipopt"]["linear_solver"]
        df_true["rho_sc"] = rho_sc
        
        
        #check if  we have an ok solution
        if x_true[0,-1] <=3.65:
            dict_fail = {"mpc_type": mpc_type,
                         "ti": i,
                         "ns": ns,
                         "S_stats": copy.deepcopy(S.stats()),
                         "x_true": x_true[:,-1]}
            ipopt_not_converged.append(dict_fail)
        
        #save df_true
        df_true.to_pickle(os.path.join(dir_res_mpc, f"df_true_{ns}.pkl"))
        
        print(f"Number of constraint violations over the whole time range: {(x_true > x_max_true.reshape(-1,1)).sum(axis = 1)} and {x_true[:,-1]=}")
        if (x_true > x_max_true.reshape(-1,1)).sum(axis = 1)[0] > 30:
            print(f"Many constraint violations here. Have {x_true[:, -1]=}")

#%% save ipopt_flags
dir_ipopt_flags = os.path.join(dir_res, "rvs", "ipopt_flags")
utils.delete_files_in_folder(dir_ipopt_flags) #clean directory from previous results
if not os.path.exists(dir_ipopt_flags):
    os.mkdir(dir_ipopt_flags)
for di in range(len(ipopt_not_converged)):
    fpath = os.path.join(dir_ipopt_flags, f"ipopt_not_converged_{di}.pkl")
    with open(fpath, "wb") as f:
        pickle.dump(ipopt_not_converged[di], f)

