# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 13:39:43 2025

@author: likwa
"""
import os
import shutil


for p in range(0,64000,1000):
    filename = f'E:\\\Ising_model_results_daai\\temp\\J_min_median_max_mean_std_ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_{p}.pt'
    if os.path.exists(filename):
        path = f'E:\\Ising_model_results_daai\\even_more_temp\\{p}'
        if not os.path.exists(path):
            os.mkdir(path)
        
        # Move the file into the directory
        base_filename = os.path.basename(filename)
        shutil.move(filename, os.path.join(path, base_filename))
        print(f'Moved {filename} to {path}/')