import os
import torch
from scipy import stats
import time
import argparse
import hcpdatautils as hcp
import isingmodellight
from isingmodellight import IsingModelLight

# Put data from Pytorch Tensor pickle (.pt) files of brain region properties into tab-delimited tables.
coords_directory = 'D:\\HCP_data'
tensor_directory = 'E:\\Ising_model_results_daai'
table_directory = 'E:\\Ising_model_results_daai'

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
epsilon = 0.0

def get_closest_match(values:torch.Tensor, target:float):
    return torch.argmin( torch.abs(values - target) )

def write_node_file(file_name:str, coords:torch.Tensor, values_for_color:torch.Tensor=None, values_for_size:torch.Tensor=None, names:list=None):
    num_nodes = coords.size(dim=0)
    if type(values_for_color) == type(None):
        values_for_color = torch.zeros_like(coords[:,0])
    if type(values_for_size) == type(None):
        values_for_size = torch.ones_like(coords[:,0])
    if type(names) == type(None):
        names = ['-']*num_nodes
    node_lines = [ '\t'.join(['x', 'y', 'z', 'color', 'size', 'name'])+'\n' ] + [ '\t'.join([f'{x:.3g}', f'{y:.3g}', f'{z:.3g}', f'{c:.3g}', f'{s:.3g}', n])+'\n' for x, y, z, c, s, n in zip(coords[:,0], coords[:,1], coords[:,2], values_for_color, values_for_size, names) ]
    with open(file_name, 'w') as node_file:
        node_file.writelines(node_lines)

with torch.no_grad():
    region_names, region_coords = hcp.load_roi_info( directory_path=coords_directory, dtype=float_type, device=device )
    print( 'loaded coordinates, size', region_coords.size() )
    # Save region feature mean and SD.
    region_features_file = os.path.join(tensor_directory, 'node_features_all_as_is.pt')
    region_feature_names = ['thickness', 'myelination', 'curvature', 'sulcus_depth']
    num_region_features = len(region_feature_names)
    region_features = torch.load(f=region_features_file, weights_only=False)[:,:,:num_region_features]
    print(f'time {time.time()-code_start_time:.3f}, loaded {region_features_file}')
    region_features_std, region_features_mean = torch.std_mean(input=region_features, dim=0, keepdim=False)
    for feature_index in range(num_region_features):
        feature_name = region_feature_names[feature_index]
        feature_mean = region_features_mean[:,feature_index]
        feature_std = region_features_std[:,feature_index]
        feature_mean_file = os.path.join(table_directory, f'mean_std_{feature_name}.dlm')
        write_node_file(file_name=feature_mean_file, coords=region_coords, values_for_color=feature_mean, values_for_size=feature_std, names=region_names)
        print(f'time {time.time()-code_start_time:.3f}, saved {feature_mean_file}')
    # Save group h mean and SD over replicas.
    min_threshold = 0.0
    max_threshold = 3.0
    num_thresholds = 31
    thresholds = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device)
    threshold_index = get_closest_match(values=thresholds, target=1.0)
    group_ising_model_file_part = f'group_thresholds_{num_thresholds}_min_{min_threshold:.3g}_max_{max_threshold:.3g}_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000'
    group_model_file = os.path.join(tensor_directory, f'ising_model_light_{group_ising_model_file_part}.pt')
    group_h_std, group_h_mean = torch.std_mean( input=torch.load(f=group_model_file, weights_only=False).h[:,threshold_index,:], dim=0 )
    print(f'time {time.time()-code_start_time:.3f}, loaded {group_model_file}')
    group_h_file = os.path.join(table_directory, f'h_mean_std_{group_ising_model_file_part}.dlm')
    write_node_file(file_name=group_h_file, coords=region_coords, values_for_color=group_h_mean, values_for_size=group_h_std, names=region_names)
    # Save individual h mean and SD over subjects after taking the mean over replicas.
    individual_ising_model_file_part = 'group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000'
    individual_model_file = os.path.join(tensor_directory, f'light_{individual_ising_model_file_part}.pt')
    individual_h_std, individual_h_mean = torch.std_mean(  input=torch.mean( torch.load(individual_model_file, weights_only=False).h, dim=0 ), dim=0  )
    print(f'time {time.time()-code_start_time:.3f}, loaded {individual_model_file}')
    individual_h_file = os.path.join(table_directory, f'h_mean_std_{individual_ising_model_file_part}.dlm')
    write_node_file(file_name=individual_h_file, coords=region_coords, values_for_color=individual_h_mean, values_for_size=individual_h_std, names=region_names)
    # Save correlations between Ising model h and predicted h from structure along with p values.
    prediction_correlation_file_part = 'lstsq_corr_all_h_ising_model_light_group_threshold_1_betas_5_min_1e-10_max_0.01_beta_steps_1200_param_steps_1200_lr_0.01_beta_updates_68_param_updates_10000_individual_updates_45000'
    prediction_correlation_file = os.path.join(tensor_directory, f'{prediction_correlation_file_part}.pt')
    prediction_correlations = torch.load(f=prediction_correlation_file, weights_only=False)
    print(f'time {time.time()-code_start_time:.3f}, loaded {prediction_correlation_file}')
    p_value_file = os.path.join(tensor_directory, f'p_value_{prediction_correlation_file_part}_perms_1000000.pt')
    p_values = torch.load(f=p_value_file, weights_only=False)
    print(f'time {time.time()-code_start_time:.3f}, loaded {p_value_file}')
    prediction_correlation_with_p_file = os.path.join(table_directory, f'{prediction_correlation_file_part}.dlm')
    write_node_file(file_name=prediction_correlation_with_p_file, coords=region_coords, values_for_color=prediction_correlations, values_for_size=p_values, names=region_names)
print(f'time {time.time()-code_start_time:.3f}, done')