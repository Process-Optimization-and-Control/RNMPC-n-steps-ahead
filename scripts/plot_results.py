# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:06:08 2023

@author: halvorak
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pathlib
import os

import seaborn as sns

"""
This file plots and prints results from the simulation. It reads data/files from the directory "dir_res" (a variable name). If you want to change the directory to plot, change "res_name" and "dir_res" will be updated.

"""

font = {'size': 16}
matplotlib.rc('font', **font)

mpc_implemented = ["nmpc", "abc-nmpc", "mcss-nmpc", "ms-nmpc"]
par_samples = "rvs"


#%% Def directories
project_dir = pathlib.Path(__file__).parent.parent
dir_data = os.path.join(project_dir, "data")
dir_plots = os.path.join(project_dir, "plots")
dir_par = os.path.join(dir_data, "par_samples", par_samples)

res_name = "res" # output from the last simulation
res_name = "res - 20231004 - mumps w itertime" #output used in the paper
res_name = "res - 20231004 - HSL w itertime" #output used in the paper
dir_res = os.path.join(dir_data, res_name, par_samples)

#%% Read data
X_max = 3.7
dir_nmpc = {}
N_sim = {}
df_nmpc = {}
df_res = {}
iter_time_ipopt = pd.DataFrame(index = mpc_implemented, data = np.nan, columns = ["min", "max", "mean", "std", "median"])
std_dev_prct = np.load(os.path.join(dir_par, "std_dev_prct.npy"))

for mpc_type in mpc_implemented:
    if mpc_type == "mcss-nmpc":
        dir_nmpc[mpc_type] = os.path.join(dir_res, "mcms-nmpc")
    else:
        dir_nmpc[mpc_type] = os.path.join(dir_res, mpc_type)
    N_sim[mpc_type] = len([name for name in os.listdir(dir_nmpc[mpc_type]) if os.path.isfile(os.path.join(dir_nmpc[mpc_type], name))])
    df_nmpc[mpc_type] = [[] for ns in range(N_sim[mpc_type])]
    df_res[mpc_type] = pd.DataFrame(data = 0., index = np.arange(N_sim[mpc_type]), columns = [r"$\sum_k P_k$", r"$\sum_k (\Delta u_k)^2$", r"$\sum_k J_{e,k}$", "num_cv", "X_cv", "sim_time", "S_in", "X_f"])
        
    for ns in range(N_sim[mpc_type]):
        df_nmpc[mpc_type][ns] = pd.read_pickle(os.path.join(dir_nmpc[mpc_type], f"df_true_{ns}.pkl")) #full history
            
        df_nmpc[mpc_type][ns]["Time"] = df_nmpc[mpc_type][ns].index
        df_nmpc[mpc_type][ns]["Ns"] = ns
        
        
        #compute "interesting" results
        df_res[mpc_type].loc[ns, r"$\sum_k P_k$"] = df_nmpc[mpc_type][ns]["P"].sum()
        df_res[mpc_type].loc[ns, r"$\sum_k (\Delta u_k)^2$"] = (df_nmpc[mpc_type][ns]["u"].diff()[1:]**2).sum()
        df_res[mpc_type].loc[ns, r"$\sum_k J_{e,k}$"] = -df_res[mpc_type].loc[ns, r"$\sum_k P_k$"] + df_res[mpc_type].loc[ns, r"$\sum_k (\Delta u_k)^2$"]
        df_res[mpc_type].loc[ns, "num_cv"] = (df_nmpc[mpc_type][ns]["X"] > 3.7).sum()
        df_res[mpc_type].loc[ns, "X_cv"] = np.trapz(np.maximum(0, df_nmpc[mpc_type][ns]["X"] - 3.7), dx = 1.)
        df_res[mpc_type].loc[ns, "sim_time"] = df_nmpc[mpc_type][ns]["sim_time"].iloc[-1]
        df_res[mpc_type].loc[ns, "X_f"] = df_nmpc[mpc_type][ns]["X"].iloc[-1]
    
    t_iter_ipopt_all = np.hstack([df_nmpc[mpc_type][s]["t_iter_nlp"].dropna().to_numpy() for s in range(N_sim[mpc_type])])
    iter_time_ipopt.loc[mpc_type, "min"] = t_iter_ipopt_all.min()
    iter_time_ipopt.loc[mpc_type, "max"] = t_iter_ipopt_all.max()
    iter_time_ipopt.loc[mpc_type, "mean"] = t_iter_ipopt_all.mean()
    iter_time_ipopt.loc[mpc_type, "std"] = t_iter_ipopt_all.std()
    iter_time_ipopt.loc[mpc_type, "median"] = np.median(t_iter_ipopt_all)
try:
    N_mc = df_nmpc[mpc_type][-1]["N_mc"][-1]
    use_LHS = df_nmpc[mpc_type][-1]["N_mc"][-1]
except:
    N_mc = 100
    use_LHS = False
    print("Rerun simulations to get these things")
 
#read parameters
df_par = []
for ns in range(list(N_sim.values())[0]):
    df_par.append(pd.read_pickle(os.path.join(dir_par, "df_par_sim_" + f"{ns}" + ".pkl")))
    for mpc_type in mpc_implemented:
        df_res[mpc_type]["S_in"].iloc[ns] = df_par[ns]["S_in"].iloc[0]

#multicolumn index dataframe
reform = {(outerKey, innerKey): values for outerKey, innerDict in df_res.items() for innerKey, values in innerDict.items()}
df_res_multi = pd.DataFrame(reform)

df_list_all2 = []
df_list_all3 = []
for mpc_type in mpc_implemented:
    df2 = df_res[mpc_type].copy()
    df_list_all2.append(df2)
    
    df3 = df_res[mpc_type].copy()
    df3["mpc"] = mpc_type
    df_list_all3.append(df3)
df_res_multi2 = pd.concat(df_list_all2, keys = mpc_implemented)
df_res_all = pd.concat(df_list_all3, ignore_index = True)

#%% Read ipopt_flags
dir_ipopt_flags = os.path.join(dir_res, "ipopt_flags")
ipopt_flags = []
x_true_div_ms = []
for filename in os.listdir(dir_ipopt_flags):
    fpath = os.path.join(dir_ipopt_flags, filename)
    ipopt_nc = pd.read_pickle(fpath) #not converged
    try:
        # print(f"{ipopt_nc['mpc_type']=}, {ipopt_nc['ns']=}, {ipopt_nc['ti']=}, {ipopt_nc['x_true']=}")
        if ((ipopt_nc['mpc_type'] == "ms-nmpc") and ("x_true" in ipopt_nc)):
            # print("here")
            x_true_div_ms.append(ipopt_nc["ns"])
    except KeyError:
        print("\n\n")
        print(f"{ipopt_nc['mpc_type']=}, {ipopt_nc['ns']=}, {ipopt_nc['ti']=}")
        print("\n\n")
        
    ipopt_flags.append(ipopt_nc)
x_true_div_ms = np.array(x_true_div_ms)
x_true_div_ms.sort()

#%%Swarm plot
for i in range(df_res_all.shape[0]): #convert to upper case
    df_res_all.loc[i, "mpc"] = df_res_all.loc[i, "mpc"].upper()

key_xcv = r"$\int_{t_0}^{t_f} max(0,X(t)-3.7)dt$"
df_res_all = df_res_all.rename(columns = {"X_cv": key_xcv})

order_mpc_plot = ["ABC-NMPC", "MCSS-NMPC", "MS-NMPC", "NMPC"]
fig_strip, ax_strip = plt.subplots(1,1, layout = "constrained", figsize = (6.9,4.8))    
sns.stripplot(df_res_all, x = "mpc", y = key_xcv, ax = ax_strip, order = order_mpc_plot)
ax_strip.set_xlabel(None)

for i in range(df_res_all.shape[0]): #convert to lower case
    df_res_all.loc[i, "mpc"] = df_res_all.loc[i, "mpc"].lower()
    
#%% Suspicious simulations
#X_true too low
x_too_low = [] #biomass concentration too low - sth strange with the trajectory
cv_too_high = {} #constraint violation too high
for mpc_type in mpc_implemented:
    for si in range(len(df_nmpc[mpc_type])):
        if df_nmpc[mpc_type][si].loc[149,"X"] < 3.63:
            try:
                x_too_low.append(dict(mpc=mpc_type, sim=si, w0_method = df_nmpc[mpc_type][si].loc[0,"w0_method"]))
            except KeyError:
                x_too_low.append(dict(mpc=mpc_type, sim=si))
                
        
        cv_too_high[mpc_type] = list(df_res[mpc_type][df_res[mpc_type]["X_cv"]>0.2].index)
        
            
sim_range = [d["sim"] for d in x_too_low]
try_clipping = [d["sim"] for d in x_too_low if d["w0_method"]=="w_opt_copy"]
# try_clipping =  [x for x in ]
#%% Plot
fig_res, ax_res = plt.subplots(len(df_res[mpc_type].columns), 1, sharex = True, layout = "constrained")
for i in range(len(ax_res)):
    for mpc_type in mpc_implemented:
        if ((mpc_type == "abc-nmpc") or (mpc_type == "mcss-nmpc")):
            # label = mpc_type + f", N_MC = {df_nmpc[mpc_type][-1]['N_mc'][-1]}"
            label = mpc_type + r", $N_{MC}$ = " + "{0}".format(df_nmpc[mpc_type][-1]['N_mc'].iloc[-1])
        else:
            label = mpc_type
        ax_res[i].plot(df_res[mpc_type].index, df_res[mpc_type].iloc[:, i], label = label)
    ax_res[i].set_ylabel(df_res[mpc_type].columns[i])
    ax_res[i].legend()
ax_res[-1].set_xlabel("MC simulation number")
fig_res.suptitle(r"$\theta$ " + f"sampled {par_samples}, {N_mc=}, {use_LHS=}, {std_dev_prct=}")

#%% Plot X_final
fig_xf, ax_xf = plt.subplots(1,1,layout = "constrained")
for mpc_type in mpc_implemented:
    ax_xf = df_res[mpc_type].plot(y = "X_f", label = mpc_type, ax = ax_xf)
idx_rerun_ms = df_res["ms-nmpc"][df_res["ms-nmpc"]["X_f"] < 3.6].index
df_res["ms-nmpc"].loc[idx_rerun_ms, ["sim_time", "X_f", "X_cv"]]
#%% X_cv vs S_in
df_res_all = df_res_all.rename(columns = {"S_in": r"$S_{in}$ [g/L]"})
ax_sin = sns.jointplot(data = df_res_all, x = r"$S_{in}$ [g/L]", y = key_xcv, hue = "mpc", height = 7)
handles, labels = ax_sin.ax_joint.get_legend_handles_labels()
labels = [label.upper() for label in labels]
ax_sin.ax_joint.legend(handles=handles, labels=labels)

ax_sin.ax_marg_y.remove()

axins = ax_sin.ax_joint.inset_axes([0.2, 0.5, 0.35, 0.3])
axins= sns.scatterplot(data = df_res_all, x = r"$S_{in}$ [g/L]", y = key_xcv, hue = "mpc", legend = False, ax = axins)

x1, x2, y1, y2 = 160, 260, 0., 0.02 #subregion of axins image
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_xlabel(None)
axins.set_ylabel(None)

ax_sin.ax_joint.indicate_inset_zoom(axins, edgecolor="black")

sns.move_legend(ax_sin.ax_joint, "upper right")
# sns.scatterplot(data = df_res_all, x = "S_in", y = key_xcv, hue = "mpc", ax = ax_sin)
df_res_all = df_res_all.rename(columns = {r"$S_{in}$ [g/L]": "S_in"})


#%%Print results


for mpc_type in mpc_implemented:
    print(f"\nmean({mpc_type}[J]/ms-nmpc[J]): " + "{0}".format((df_res[mpc_type][r'$\sum_k J_{e,k}$']/df_res['ms-nmpc'][r'$\sum_k J_{e,k}$']).mean()))      
    print(f"median({mpc_type}[J]/ms-nmpc[J]): " + "{0}".format((df_res[mpc_type][r'$\sum_k J_{e,k}$']/df_res['ms-nmpc'][r'$\sum_k J_{e,k}$']).median()))      

print("\n")
print(f"Constr. viol, abc-nmpc < ms-nmpc: {(df_res['abc-nmpc']['X_cv'] < df_res['ms-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc == ms-nmpc: {(df_res['abc-nmpc']['X_cv'] == df_res['ms-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc > ms-nmpc: {(df_res['abc-nmpc']['X_cv'] > df_res['ms-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}\n") 
                    
print(f"Constr. viol, mcss-nmpc < ms-nmpc: {(df_res['mcss-nmpc']['X_cv'] < df_res['ms-nmpc']['X_cv']).sum()}/{df_res['mcss-nmpc'].shape[0]}")                     
print(f"Constr. viol, mcss-nmpc == ms-nmpc: {(df_res['mcss-nmpc']['X_cv'] == df_res['ms-nmpc']['X_cv']).sum()}/{df_res['mcss-nmpc'].shape[0]}")                     
print(f"Constr. viol, mcss-nmpc > ms-nmpc: {(df_res['mcss-nmpc']['X_cv'] > df_res['ms-nmpc']['X_cv']).sum()}/{df_res['mcss-nmpc'].shape[0]}\n")  
                    
print(f"Constr. viol, abc-nmpc < mcss-nmpc: {(df_res['abc-nmpc']['X_cv'] < df_res['mcss-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc == mcss-nmpc: {(df_res['abc-nmpc']['X_cv'] == df_res['mcss-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc > mcss-nmpc: {(df_res['abc-nmpc']['X_cv'] > df_res['mcss-nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}\n") 

print(f"Constr. viol, abc-nmpc < nmpc: {(df_res['abc-nmpc']['X_cv'] < df_res['nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc == nmpc: {(df_res['abc-nmpc']['X_cv'] == df_res['nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}")                     
print(f"Constr. viol, abc-nmpc > nmpc: {(df_res['abc-nmpc']['X_cv'] > df_res['nmpc']['X_cv']).sum()}/{df_res['abc-nmpc'].shape[0]}\n") 

print(f"Constr. viol, mcss-nmpc < nmpc: {(df_res['mcss-nmpc']['X_cv'] < df_res['nmpc']['X_cv']).sum()}/{df_res['mcss-nmpc'].shape[0]}")                     
print(f"Constr. viol, mcss-nmpc == nmpc: {(df_res['mcss-nmpc']['X_cv'] == df_res['nmpc']['X_cv']).sum()}/{df_res['mcss-nmpc'].shape[0]}")                     
print(f"Constr. viol, mcss-nmpc > nmpc: {(df_res['mcss-nmpc']['X_cv'] > df_res['nmpc']['X_cv']).sum()}/{df_res['mcss-nmpc'].shape[0]}\n") 


print(f"Constr. viol, ms-nmpc < nmpc: {(df_res['ms-nmpc']['X_cv'] < df_res['nmpc']['X_cv']).sum()}/{df_res['ms-nmpc'].shape[0]}")                     
print(f"Constr. viol, ms-nmpc == nmpc: {(df_res['ms-nmpc']['X_cv'] == df_res['nmpc']['X_cv']).sum()}/{df_res['ms-nmpc'].shape[0]}")                     
print(f"Constr. viol, ms-nmpc > nmpc: {(df_res['ms-nmpc']['X_cv'] > df_res['nmpc']['X_cv']).sum()}/{df_res['ms-nmpc'].shape[0]}\n") 

idx_abc = df_res['abc-nmpc'][df_res['abc-nmpc']['X_cv'] > df_res['ms-nmpc']['X_cv']]
idx_mcss = df_res['mcss-nmpc'][df_res['mcss-nmpc']['X_cv'] > df_res['ms-nmpc']['X_cv']]
idx_ms = df_res['ms-nmpc'][df_res['ms-nmpc']['X_cv'] > df_res['mcss-nmpc']['X_cv']]
idx_nmpc = df_res['nmpc'][df_res['nmpc']['X_cv'] == df_res['ms-nmpc']['X_cv']]["num_cv"]
idx_nmpc2 = df_res['nmpc'][df_res['nmpc']['X_cv'] <= df_res['ms-nmpc']['X_cv']]
idx_nmpc3 = df_res['nmpc'][df_res['nmpc']['X_cv'] <= df_res['mcss-nmpc']['X_cv']]
idx_nmpc4 = df_res['nmpc'][df_res['nmpc']['X_cv'] <= df_res['abc-nmpc']['X_cv']]

print(f"Sim where num constr. viol abc-nmpc > ms-nmpc: {list(idx_abc.index)}")                   
print(f"Sim where num constr. viol mcss-nmpc > ms-nmpc: {list(idx_mcss.index)}")                   
print(f"Sim where num constr. viol nmpc == ms-nmpc and num cv:\n{idx_nmpc}")           

for mpc_type in mpc_implemented:
    print(f"Sim with zero constraint violation, {mpc_type}: {(df_res[mpc_type]['X_cv'] == 0).sum()}")                     
      


#%% Plot last state trajectory
iter_plt = 22 #which simulation to plot
plot_values = ["X", "S", "P", "V", "u"]
uom = ["[g/L]", "[g/L]", "[g/L]", "[L]", "[L/h]"]
figsize = (9,6)
fig_x, ax_x = plt.subplots(len(plot_values), 1, sharex = True, layout = "constrained", figsize = figsize)

for mpc_type in mpc_implemented:
    # ax_x = df_nmpc[mpc_type][-1].loc[:, ["X", "S", "P", "V"]].plot(ax = ax_x, label = mpc_type, subplots = True, kind = "line")
    for i in range(ax_x.shape[0]):
        if i == ax_x.shape[0]-1: #step plot for u
            ax_x[i].step(df_nmpc[mpc_type][iter_plt].index, df_nmpc[mpc_type][iter_plt].loc[:, plot_values].iloc[:, i], label = mpc_type.upper())
        else: # continuous plot for the states
            ax_x[i].plot(df_nmpc[mpc_type][iter_plt].index, df_nmpc[mpc_type][iter_plt].loc[:, plot_values].iloc[:, i], label = mpc_type.upper())
        ax_x[i].set_ylabel(plot_values[i] + " " + uom[i])
        ax_x[i].get_yaxis().set_label_coords(-0.1,0.5)
xlim = ax_x[0].get_xlim()
ax_x[0].plot(list(xlim), [3.7, 3.7], 'r--', label = "Constraint")
ax_x[-1].plot(list(xlim), [0., 0.], 'r--', label = "Constraint")
# ax_x[-1].plot(list(xlim), [0.2, 0.2], 'r--', label = "Constraint")
ax_x[0].set_xlim(xlim)
ax_x[-1].set_xlim(xlim)
ax_x[-1].set_xlabel("Time [h]")
ax_x[0].legend(ncol = 3)


fig_cv, ax_cv = plt.subplots(2, 1, sharex = True, layout = "constrained", figsize = figsize)
if iter_plt == 90:
    time_to_plt = np.arange(77,90)
    
else:
    time_to_plt = np.arange(70,150)

for mpc_type in mpc_implemented:
    ax_cv[0].plot(df_nmpc[mpc_type][iter_plt].index[time_to_plt], df_nmpc[mpc_type][iter_plt].loc[time_to_plt,"X"], label = mpc_type.upper())
    ax_cv[1].step(df_nmpc[mpc_type][iter_plt].index[time_to_plt], df_nmpc[mpc_type][iter_plt].loc[time_to_plt,"u"], label = mpc_type.upper())
ax_cv[0].set_ylabel("X "+ uom[0])
xlim = ax_cv[0].get_xlim()
ax_cv[0].plot(list(xlim), [3.7, 3.7], 'r--', label = "Constraint")
ax_cv[-1].plot(list(xlim), [0., 0.], 'r--', label = "Constraint")
ax_cv[1].set_ylabel("u "+ uom[-1])
ax_cv[1].set_xlabel("Time [h] ")
ax_cv[0].legend(ncol = 3)
ax_cv[0].set_xlim((time_to_plt[0], xlim[-1]))
ax_cv[0].get_yaxis().set_label_coords(-0.1,0.5)
ax_cv[1].get_yaxis().set_label_coords(-0.1,0.5)

fig_cv2, ax_cv2 = plt.subplots(3, 1, sharex = True, layout = "constrained", figsize = figsize)
if iter_plt == 90:
    time_to_plt = np.arange(74,90)
    
else:
    time_to_plt = np.arange(70,150)

for mpc_type in mpc_implemented:
    ax_cv2[0].plot(df_nmpc[mpc_type][iter_plt].index[time_to_plt], df_nmpc[mpc_type][iter_plt].loc[time_to_plt,"X"], label = mpc_type.upper())
    ax_cv2[1].plot(df_nmpc[mpc_type][iter_plt].index[time_to_plt], df_nmpc[mpc_type][iter_plt].loc[time_to_plt,"S"], label = mpc_type.upper())
    ax_cv2[2].step(df_nmpc[mpc_type][iter_plt].index[time_to_plt], df_nmpc[mpc_type][iter_plt].loc[time_to_plt,"u"], label = mpc_type.upper())
ax_cv2[0].set_ylabel("X "+ uom[0])
xlim = ax_cv2[0].get_xlim()
ax_cv2[0].plot(list(xlim), [3.7, 3.7], 'r--', label = "Constraint")
ax_cv2[-1].plot(list(xlim), [0., 0.], 'r--', label = "Constraint")
ax_cv2[1].set_ylabel("S "+ uom[1])
ax_cv2[2].set_ylabel("u "+ uom[-1])
ax_cv2[2].set_xlabel("Time [h] ")
ax_cv2[0].legend(ncol = 3)
ax_cv2[0].set_xlim((time_to_plt[0], xlim[-1]))
ax_cv2[0].get_yaxis().set_label_coords(-0.1,0.5)
ax_cv2[1].get_yaxis().set_label_coords(-0.1,0.5)
ax_cv2[2].get_yaxis().set_label_coords(-0.1,0.5)



#%% S_in at each simulation
s_in_traj = np.array([df_par[ns]["S_in"].iloc[0] for ns in range(N_sim["nmpc"])])
s_in_traj = pd.DataFrame(data = s_in_traj, columns = [r"$S_{in}$"])
s_in_traj.index.name = "Simulation number"
s_in_traj.plot()
s_in_traj.sort_values(by = r"$S_{in}$").tail(15)

#%% Simulation time
# df_res_all = df_res_all.rename(columns = {"sim_time": "Simulation time [s]"})
fig_time, ax_time = plt.subplots(1, 1, layout = "constrained")
ax_time = sns.stripplot(data = df_res_all, y = "sim_time", x = "mpc", ax = ax_time)
ax_time.set_ylabel("Simulation time [s]")
# ax_time = sns.histplot(data = df_res_all, x = "sim_time", hue = "mpc", ax = ax_time, bins = 50)
# ax_time.set_xlabel("Simulation time [s]")

for mpc_type in mpc_implemented:
    print(f'{mpc_type}, simulation time [s]: {df_res_all[df_res_all["mpc"] == mpc_type]["sim_time"].mean(): .2f} ({df_res_all[df_res_all["mpc"] == mpc_type]["sim_time"].std(): .2f})')

print(f"\niter_time_ipopt=\n{iter_time_ipopt}") #statistics for time used to solve the nlp (actually, OCP for the abc-nmpc) at each time step
df_t = []
i = 0
fig_t_nlp, ax_t_nlp = plt.subplots(1,1, layout = "constrained")
s = 2
for mpc_type in mpc_implemented:
    t_nlp_iter = df_nmpc[mpc_type][0]["t_iter_nlp"].dropna()
    l = ax_t_nlp.scatter(t_nlp_iter.index, t_nlp_iter, label = mpc_type.upper(), s = s)
    for ns in range(1, N_sim[mpc_type]):
        t_nlp_iter = df_nmpc[mpc_type][ns]["t_iter_nlp"].dropna()
        ax_t_nlp.scatter(t_nlp_iter.index, t_nlp_iter, color = l.get_facecolor(), s = s)

    df_t.append(pd.concat(df_nmpc[mpc_type], axis = 0, ignore_index = True))
    df_t[i]["Method"] = mpc_type
    i += 1
    # df_
df_t = pd.concat(df_t, axis = 0, ignore_index = True)
df_t = df_t[["Time", "Ns", "Method", "t_iter_nlp", "X"]]
df_t = df_t.dropna()

if False:
    fig_sns, ax_sns = plt.subplots(1,1, layout = "constrained")
    # ax_sns = sns.lineplot(data= df_t, x = "Time", y = "t_iter_nlp", hue = "Method", ax = ax_sns, err_style = None)
    ax_sns = sns.lineplot(data= df_t, x = "Time", y = "t_iter_nlp", hue = "Method", ax = ax_sns, errorbar=("sd", 1))
    ax_sns = sns.lineplot(data= df_t, x = "Time", y = "t_iter_nlp", hue = "Method", ax = ax_sns, err_style = None, estimator = np.max, legend = False, linestyle = "dashed")
    ax_sns = sns.lineplot(data= df_t, x = "Time", y = "t_iter_nlp", hue = "Method", ax = ax_sns, err_style = None, estimator = np.min, legend = False, linestyle = "dashed")
    lines = ax_sns.get_legend().legend_handles

if False:
    fig_sns2, ax_sns2 = plt.subplots(1,1, layout = "constrained")
    ax_sns2 = sns.lineplot(data= df_t, x = "Time", y = "t_iter_nlp", hue = "Method", ax = ax_sns2, estimator = None, units = "Ns", linewidth = .4)

if True:
    fig_sns3, ax_sns3 = plt.subplots(2,1, layout = "constrained", sharex = True)
    ax_sns3[0] = sns.lineplot(data= df_t, x = "Time", y = "X", hue = "Method", ax = ax_sns3[0], estimator = None, units = "Ns", linewidth = .4, legend = False)
    ax_sns3[1] = sns.lineplot(data= df_t, x = "Time", y = "t_iter_nlp", hue = "Method", ax = ax_sns3[1], estimator = None, units = "Ns", linewidth = .4)
    
    xlim = ax_sns3[0].get_xlim()
    ax_sns3[0].plot(list(xlim), [3.7, 3.7], 'r--')
    ax_sns3[0].set_xlim(xlim)
    
    ax_sns3[0].set_ylabel(r"$X$ [g/L]")
    ax_sns3[1].set_ylabel(r"OCP iteration time [s]")
    ax_sns3[1].set_xlabel("Time [h]")
    #set legend to upper case and use mcss-nmpc
    lines = ax_sns3[1].get_legend().legend_handles
    for l in lines:
        l.set_label(l.get_label().upper())
    ax_sns3[1].legend(handles = lines) 
        

ax_t_nlp.set_xlabel("Time [h]")       
ax_t_nlp.set_ylabel("OCP iteration time [s]")    
ax_t_nlp.legend()   


#%% Cost function
for i in range(df_res_all.shape[0]): #convert to upper case
    df_res_all.loc[i, "mpc"] = df_res_all.loc[i, "mpc"].upper()

key_cost = r'$\sum_k J_{e,k}$'
df_res_all["Constraint violation"] = df_res_all["num_cv"] > 0.

fig_strip, ax_strip = plt.subplots(1,1, layout = "constrained", figsize = (6.9,4.8))    
sns.stripplot(df_res_all, x = "mpc", y = key_cost, ax = ax_strip, hue = "Constraint violation", order = order_mpc_plot)
ax_strip.set_xlabel(None)

ax_strip.set_ylabel(r'$\sum_k (-P_k + (\Delta u_k)^2)$')


for i in range(df_res_all.shape[0]): #convert to lower case
    df_res_all.loc[i,"mpc"] = df_res_all.loc[i,"mpc"].lower()

    
