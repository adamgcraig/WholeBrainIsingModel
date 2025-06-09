import os
import pandas
import numpy as np
import torch
import math
import isingmodellight
from isingmodellight import IsingModelLight
from scipy import stats
import time
import hcpdatautils as hcp
import nibabel as nib

code_start_time = time.time()

int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

file_dir = 'E:\\Ising_model_results_daai'
other_file_dir = 'D:\\Ising_model_results_daai'

training_subject_start=0
training_subject_end = 670

def depytorch(t:torch.Tensor):
    return t.detach().cpu().numpy()

def get_closest_match(values:torch.Tensor, target:float):
    return torch.argmin( torch.abs(values - target) )

def load_h_only():
    ising_model_file = os.path.join(file_dir, f'ising_model_{param_string}.pt')
    model = torch.load(ising_model_file, weights_only=False)
    print( 'h size', model.h.size() )
    print( 'J size', model.J.size() )
    return model.h.mean(dim=0)

def prepend_zero(array:torch.Tensor):
    return torch.cat(   (  torch.zeros( size=(1,), dtype=array.dtype, device=array.device ), array  ), dim=0   )

num_beta = 101
min_beta = 1e-10
max_beta = 0.05
num_threshold = 31
min_threshold = 0.0
max_threshold = 3.0
threshold = torch.linspace(start=min_threshold, end=3, steps=num_threshold, dtype=float_type, device=device)
blue_thresh = get_closest_match(values=threshold, target=0.0)
green_thresh = get_closest_match(values=threshold, target=1.0)
red_thresh = get_closest_match(values=threshold, target=2.4)
data_string = f'thresholds_{num_threshold}_min_{min_threshold:.3g}_max_{max_threshold:.3g}'
param_string = f'light_group_{data_string}_betas_{num_beta}_min_{min_beta:.3g}_max_{max_beta}_steps_1200_lr_0.01_beta_updates_8_param_updates_40000'

print(f'time {time.time()-code_start_time:.3f}, loading h values from model...')
h_mean = load_h_only()
print( f'time {time.time()-code_start_time:.3f}, loaded h values from model', h_mean.size() )

glasser_voxels_file = os.path.join('C:\\Users','agcraig','Documents','GitHub','glasser360','glasser360MNI.nii.gz')
glasser_voxels = nib.load(glasser_voxels_file)
print(glasser_voxels)

print(f'time {time.time()-code_start_time:.3f}, Atlas voxels...')
glasser_voxels_data = torch.from_numpy( glasser_voxels.get_fdata() ).to(device=device)
print( f'time {time.time()-code_start_time:.3f}, loaded Atlas voxels', glasser_voxels_data.size() )

selected_threshold = 1.0
threshold_index = torch.argmin( torch.abs(threshold - selected_threshold) )
selected_threshold = threshold[threshold_index]
print(f'selected threshold {selected_threshold:.3g} at index {threshold_index}')
selected_h = h_mean[threshold_index,:]
print( f'min absolute value of selected h values {torch.min( torch.abs(selected_h) )}' )
h_is_0 = selected_h == 0.0
print( f'number of exact 0s in selected h {torch.count_nonzero(h_is_0)}' )
selected_h[h_is_0] = -1.0e-5

# The Atlas lists the hemispheres in the opposite order from what we are using.
# They do right first, then left.
# We do left first, then right.
# Swap our order to match theirs here.
h_swapped = torch.zeros_like(selected_h)
num_nodes_half = selected_h.numel()//2
h_swapped[:num_nodes_half] = selected_h[num_nodes_half:]
h_swapped[num_nodes_half:] = selected_h[:num_nodes_half]

h_with_0 = prepend_zero(selected_h)
voxel_indices = glasser_voxels_data.int()
h_voxels = h_with_0[voxel_indices]
print( h_voxels.size() )
h_image = nib.Nifti1Image( dataobj=depytorch(h_voxels), affine=glasser_voxels.affine, header=glasser_voxels.header, extra=glasser_voxels.extra, file_map=glasser_voxels.file_map )
nib.save( h_image, os.path.join(file_dir, f'h_voxels_{param_string}_selected_thresh_{selected_threshold:.3g}.nii') )