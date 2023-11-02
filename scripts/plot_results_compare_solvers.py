# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:06:08 2023

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
import seaborn as sns

import utils
import mpc

"""
This file plots/compares the results obtained with MA57 and MUMPS as linear solvers in IPOPT.

"""


font = {'size': 16}
matplotlib.rc('font', **font)

mpc_implemented = ["nmpc", "abc-nmpc", "mcms-nmpc", "ms-nmpc"]
par_samples = "rvs"

#%% Def directories
project_dir = pathlib.Path(__file__).parent.parent
dir_data = os.path.join(project_dir, "data")
dir_plots = os.path.join(project_dir, "plots")
dir_par = os.path.join(dir_data, "par_samples", par_samples)

res_name = "res"
res_name1 = "res - 20231004 - mumps w itertime" #output used in the paper
res_name2 = "res - 20231004 - HSL w itertime" #output used in the paper

res_names = [res_name1, res_name2]
solver_name = ["ma57", "mumps"]
df_res_all_list = []
for res_name, lin_solver in zip(res_names, solver_name):
    dir_res = os.path.join(dir_data, res_name, par_samples)
    
    #%% Read data
    X_max = 3.7
    dir_nmpc = {}
    N_sim = {}
    df_nmpc = {}
    df_res = {}
    std_dev_prct = np.load(os.path.join(dir_par, "std_dev_prct.npy"))
    
    for mpc_type in mpc_implemented:
        dir_nmpc[mpc_type] = os.path.join(dir_res, mpc_type)
        N_sim[mpc_type] = len([name for name in os.listdir(dir_nmpc[mpc_type]) if os.path.isfile(os.path.join(dir_nmpc[mpc_type], name))])
        df_nmpc[mpc_type] = [[] for ns in range(N_sim[mpc_type])]
        df_res[mpc_type] = pd.DataFrame(data = 0., index = np.arange(N_sim[mpc_type]), columns = [r"$\sum_k P_k$", r"$\sum_k (\Delta u_k)^2$", r"$\sum_k J_{e,k}$", "num_cv", "X_cv", "sim_time", "S_in", "X_f", "linear_solver"])
        
        for ns in range(N_sim[mpc_type]):
            df_nmpc[mpc_type][ns] = pd.read_pickle(os.path.join(dir_nmpc[mpc_type], f"df_true_{ns}.pkl")) #full history
            
            #compute "interesting" results
            df_res[mpc_type].loc[ns, r"$\sum_k P_k$"] = df_nmpc[mpc_type][ns]["P"].sum()
            df_res[mpc_type].loc[ns, r"$\sum_k (\Delta u_k)^2$"] = (df_nmpc[mpc_type][ns]["u"].diff()[1:]**2).sum()
            df_res[mpc_type].loc[ns, r"$\sum_k J_{e,k}$"] = -df_res[mpc_type].loc[ns, r"$\sum_k P_k$"] + df_res[mpc_type].loc[ns, r"$\sum_k (\Delta u_k)^2$"]
            df_res[mpc_type].loc[ns, "num_cv"] = (df_nmpc[mpc_type][ns]["X"] > 3.7).sum()
            df_res[mpc_type].loc[ns, "X_cv"] = np.trapz(np.maximum(0, df_nmpc[mpc_type][ns]["X"] - 3.7), dx = 1.)
            df_res[mpc_type].loc[ns, "sim_time"] = df_nmpc[mpc_type][ns]["sim_time"].iloc[-1]
            df_res[mpc_type].loc[ns, "X_f"] = df_nmpc[mpc_type][ns]["X"].iloc[-1]
            df_res[mpc_type].loc[ns, "linear_solver"] = lin_solver
            
    try:
        N_mc = df_nmpc[mpc_type][-1]["N_mc"][-1]
        use_LHS = df_nmpc[mpc_type][-1]["N_mc"][-1]
    except:
        N_mc = 100
        use_LHS = False
        print("Rerun simulations to get these things")
 

    df_list_all3 = []
    for mpc_type in mpc_implemented:
    
        
        df3 = df_res[mpc_type].copy()
        df3["mpc"] = mpc_type
        df_list_all3.append(df3)
    df_res_all = pd.concat(df_list_all3, ignore_index = True)
    df_res_all_list.append(df_res_all)
df_res_all = pd.concat(df_res_all_list, ignore_index=True)


#%%Swarm plot
for i in range(df_res_all.shape[0]): #convert to upper case
    df_res_all.loc[i, "mpc"] = df_res_all.loc[i, "mpc"].upper()

key_xcv = r"$\int_{t_0}^{t_f} max(0,X(t)-3.7)dt$"
df_res_all = df_res_all.rename(columns = {"X_cv": key_xcv})

fig_strip, ax_strip = plt.subplots(1,1, layout = "constrained")    
sns.stripplot(df_res_all, x = "mpc", y = key_xcv, ax = ax_strip, hue = "linear_solver")
ax_strip.set_xlabel(None)



for i in range(df_res_all.shape[0]): #convert to lower case
    df_res_all.loc[i, "mpc"] = df_res_all.loc[i, "mpc"].lower()

#%%Print results

for mpc_type in mpc_implemented:
    print(f"\nmean({mpc_type}[J]/ms-nmpc[J]): " + "{0}".format((df_res[mpc_type][r'$\sum_k J_{e,k}$']/df_res['ms-nmpc'][r'$\sum_k J_{e,k}$']).mean()))      
    print(f"median({mpc_type}[J]/ms-nmpc[J]): " + "{0}".format((df_res[mpc_type][r'$\sum_k J_{e,k}$']/df_res['ms-nmpc'][r'$\sum_k J_{e,k}$']).median()))      

print("\n")
print(f"Constr. viol, abc-nmpc < ms-nmpc: {(df_res['abc-nmpc']['X_cv'] < df_res['ms-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc == ms-nmpc: {(df_res['abc-nmpc']['X_cv'] == df_res['ms-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc > ms-nmpc: {(df_res['abc-nmpc']['X_cv'] > df_res['ms-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}\n") 
                    
print(f"Constr. viol, mcms-nmpc < ms-nmpc: {(df_res['mcms-nmpc']['X_cv'] < df_res['ms-nmpc']['X_cv']).sum()}/{df_res['mcms-nmpc'].shape[0]}")                     
print(f"Constr. viol, mcms-nmpc == ms-nmpc: {(df_res['mcms-nmpc']['X_cv'] == df_res['ms-nmpc']['X_cv']).sum()}/{df_res['mcms-nmpc'].shape[0]}")                     
print(f"Constr. viol, mcms-nmpc > ms-nmpc: {(df_res['mcms-nmpc']['X_cv'] > df_res['ms-nmpc']['X_cv']).sum()}/{df_res['mcms-nmpc'].shape[0]}\n")  
                    
print(f"Constr. viol, abc-nmpc < mcms-nmpc: {(df_res['abc-nmpc']['X_cv'] < df_res['mcms-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc == mcms-nmpc: {(df_res['abc-nmpc']['X_cv'] == df_res['mcms-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc > mcms-nmpc: {(df_res['abc-nmpc']['X_cv'] > df_res['mcms-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}\n") 

print(f"Constr. viol, abc-nmpc < nmpc: {(df_res['abc-nmpc']['X_cv'] < df_res['nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc == nmpc: {(df_res['abc-nmpc']['X_cv'] == df_res['nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc > nmpc: {(df_res['abc-nmpc']['X_cv'] > df_res['nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}\n") 

print(f"Constr. viol, mcms-nmpc < nmpc: {(df_res['mcms-nmpc']['X_cv'] < df_res['nmpc']['X_cv']).sum()}/{df_res['mcms-nmpc'].shape[0]}")                     
print(f"Constr. viol, mcms-nmpc == nmpc: {(df_res['mcms-nmpc']['X_cv'] == df_res['nmpc']['X_cv']).sum()}/{df_res['mcms-nmpc'].shape[0]}")                     
print(f"Constr. viol, mcms-nmpc > nmpc: {(df_res['mcms-nmpc']['X_cv'] > df_res['nmpc']['X_cv']).sum()}/{df_res['mcms-nmpc'].shape[0]}\n") 


print(f"Constr. viol, ms-nmpc < nmpc: {(df_res['ms-nmpc']['X_cv'] < df_res['nmpc']['X_cv']).sum()}/{df_res['ms-nmpc'].shape[0]}")                     
print(f"Constr. viol, ms-nmpc == nmpc: {(df_res['ms-nmpc']['X_cv'] == df_res['nmpc']['X_cv']).sum()}/{df_res['ms-nmpc'].shape[0]}")                     
print(f"Constr. viol, ms-nmpc > nmpc: {(df_res['ms-nmpc']['X_cv'] > df_res['nmpc']['X_cv']).sum()}/{df_res['ms-nmpc'].shape[0]}\n") 

idx_abc = df_res['abc-nmpc'][df_res['abc-nmpc']['X_cv'] > df_res['ms-nmpc']['X_cv']]
idx_mcms = df_res['mcms-nmpc'][df_res['mcms-nmpc']['X_cv'] > df_res['ms-nmpc']['X_cv']]
idx_ms = df_res['ms-nmpc'][df_res['ms-nmpc']['X_cv'] > df_res['mcms-nmpc']['X_cv']]
idx_nmpc = df_res['nmpc'][df_res['nmpc']['X_cv'] == df_res['ms-nmpc']['X_cv']]["num_cv"]
idx_nmpc2 = df_res['nmpc'][df_res['nmpc']['X_cv'] <= df_res['ms-nmpc']['X_cv']]
idx_nmpc3 = df_res['nmpc'][df_res['nmpc']['X_cv'] <= df_res['mcms-nmpc']['X_cv']]
idx_nmpc4 = df_res['nmpc'][df_res['nmpc']['X_cv'] <= df_res['abc-nmpc']['X_cv']]

print(f"Sim where num constr. viol abc-nmpc > ms-nmpc: {list(idx_abc.index)}")                   
print(f"Sim where num constr. viol mcms-nmpc > ms-nmpc: {list(idx_mcms.index)}")                   
print(f"Sim where num constr. viol nmpc == ms-nmpc and num cv:\n{idx_nmpc}")                   
                           
print(f"Sim with 0 constraint viol, nmpc: {(df_res['nmpc']['X_cv'] == 0).sum()}")
print(f"Sim with 0 constraint viol, abc-nmpc: {(df_res['abc-nmpc']['X_cv'] == 0).sum()}")
print(f"Sim with 0 constraint viol, mcms-nmpc: {(df_res['mcms-nmpc']['X_cv'] == 0).sum()}")
print(f"Sim with 0 constraint viol, ms-nmpc: {(df_res['ms-nmpc']['X_cv'] == 0).sum()}")


#%% Simulation time
for i in range(df_res_all.shape[0]): #convert to upper case
    df_res_all.loc[i, "mpc"] = df_res_all.loc[i, "mpc"].upper()
    
fig_time, ax_time = plt.subplots(1, 1, layout = "constrained")
ax_time = sns.stripplot(data = df_res_all[df_res_all["mpc"]!="MCMS-NMPC"], y = "sim_time", x = "mpc", ax = ax_time, hue = "linear_solver")
ax_time.set_ylabel("Simulation time [s]")
ax_time.set_xlabel(None)


for i in range(df_res_all.shape[0]): #convert to lower case
    df_res_all.loc[i,"mpc"] = df_res_all.loc[i,"mpc"].lower()
#%%Print time
for lin_sol in solver_name:
    for mpc_type in mpc_implemented:
        print(f'{mpc_type} & {lin_sol=}, simulation time [s]: {df_res_all[(df_res_all["mpc"] == mpc_type) & (df_res_all["linear_solver"] == lin_sol)]["sim_time"].mean(): .2f} ({df_res_all[(df_res_all["mpc"] == mpc_type) & (df_res_all["linear_solver"] == lin_sol)]["sim_time"].std(): .2f})')


    
