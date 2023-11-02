# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:34:04 2023

@author: halvorak
"""

import numpy as np
import pandas as pd
import seaborn as sns
import os
import pathlib
import shutil


import utils



#%% Def directories
project_dir = pathlib.Path(__file__).parent.parent
dir_data = os.path.join(project_dir, "data")
dir_param_samples = os.path.join(dir_data, "par_samples")
if not os.path.exists(dir_param_samples):
    os.mkdir(dir_param_samples)
if not os.path.exists(dir_param_samples):
    os.mkdir(dir_param_samples)
    
dir_rvs = os.path.join(dir_param_samples, "rvs") #random sampling
dir_st = os.path.join(dir_param_samples, "st") #according to scenario tree


#create sampling functions
def sampling_random(key, dist, n_samples = 1):
    return dist.rvs(size = n_samples)
def sampling_mean(key, dist, n_samples = 1):
    return np.ones(n_samples)*dist.mean()
def sampling_for_scenario_tree(key, dist, n_samples = 1):
    #samples the discrete scenarioes used in the scenario tree MPC. This SHOULD not break the constraints for the scenario tree MPC!
    
    if ((key == "Y_x") or (key == "S_in")):
        idx = np.random.randint(0, 3, size = n_samples)
        allowable_values = dist.mean() + np.array([0, -2*dist.std(), 2*dist.std()])
        return allowable_values[idx]
    else:
        return sampling_mean(key, dist, n_samples = n_samples)

#%%Input to sim
    
N_sim = 100 #how many times we will repeat the simulation

sampling_fun = sampling_random
# sampling_fun = sampling_mean
# sampling_fun = sampling_for_scenario_tree

n_samples = 500 #just have to be larger than simulation time
S_in_const = True # whether S_in should be constant through one simulation (but changes each simulation)

if sampling_fun == sampling_random:
    if os.path.exists(dir_rvs):
        utils.delete_files_in_folder(dir_rvs)
    else:
        os.mkdir(dir_rvs)
    dir_used = dir_rvs
elif sampling_fun == sampling_for_scenario_tree:
    if os.path.exists(dir_st):
        utils.delete_files_in_folder(dir_st)
    else:
        os.mkdir(dir_st)
    dir_used = dir_st


#%% Create parameters
std_dev_prct=.20
par_dist = utils.get_parameters(std_dev_prct=std_dev_prct) #dict with distributions of the parametes
par_mean = [dist.mean() for dist in par_dist.values()]
par_names = list(par_dist.keys())

with open(os.path.join(dir_used, "std_dev_prct.npy"), "wb") as f:
    np.save(f, np.array([std_dev_prct]))
for Ni in range(N_sim):
    fpath = os.path.join(dir_used, "df_par_sim_" + str(Ni) + ".pkl")
    samples = np.array([sampling_fun(key, dist, n_samples = n_samples) for key, dist in par_dist.items()]).T
    df_par = pd.DataFrame(data = samples, columns = par_names)
    if S_in_const:
        df_par["S_in"] = df_par["S_in"].iloc[0]
    assert (df_par>0).all().all(), "Have negative values in sampled parameters"
    df_par.to_pickle(fpath)
 
    
if True:
    sns.pairplot(df_par)
