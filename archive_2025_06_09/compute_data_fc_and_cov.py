import os
import torch
import time
import argparse
import hcpdatautils as hcp
from isingutilsslow import IsingModel
from isingutilsslow import binarize_data_ts
from isingutilsslow import get_param_means_and_covs_slower
from isingutilsslow import get_fc

parser = argparse.ArgumentParser(description="Fit multiple Ising models to the concatenated fMRI data of all subjects.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='training', help="which list of subjects to use, either training, validation, testing, or all")
parser.add_argument("-t", "--threshold", type=str, default='0.1', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, or the string 'median'")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes to model")
parser.add_argument("-w", "--window_length", type=int, default=4800, help="number of time points between model parameter updates")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
threshold_str = args.threshold
if threshold_str == 'median':
    threshold = threshold_str
else:
    threshold = float(threshold_str)
print(f'threshold={threshold_str}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
window_length = args.window_length
print(f'window_length={window_length}')

def get_num_nan(mat:torch.Tensor):
    return torch.count_nonzero( torch.isnan(mat) )

def prep_ts_data(data_directory:str, data_subset:str, num_nodes:int, window_length:int):
    subject_ids = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_ids)
    print(f'Data subset {data_subset} has {num_subjects} subjects.')
    # Load, normalize, binarize, and flatten the fMRI time series data.
    data_ts = torch.zeros( (num_subjects, hcp.time_series_per_subject, hcp.num_time_points, num_nodes), dtype=float_type, device=device )
    print(f'preallocated space for each unique subject time series..., time {time.time() - code_start_time:.3f}')
    for subject_index in range(num_subjects):
        subject_id = subject_ids[subject_index]
        print(f'subject {subject_index} of {num_subjects}, ID {subject_id}')
        # We originally load a 4 x T/4 x N' Tensor with values over a continuous range.
        # N' is the original total number of nodes. Cut the dimensions down to N, the desired number of nodes.
        data_ts[subject_index,:,:,:] = hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device)[:,:,:num_nodes]
    print(f'loaded all time series, time {time.time() - code_start_time:.3f}')
    # Binarize each individual dimension of each individual time series separately in case of batch effects.
    # Then rearrange things: num_subjects x 4 x T/4 x num_nodes -> num_subjects*T x num_nodes -> num_nodes x num_subjects*T -> 1 x 1 x num_nodes x num_subjects*T 
    data_ts = binarize_data_ts(data_ts=data_ts, step_dim=-2, threshold=threshold).flatten(start_dim=0, end_dim=2).transpose(dim0=-2, dim1=-1).unsqueeze(dim=0).unsqueeze(dim=0)
    print(f'binarized time series with threshold {threshold_str} and rearranged dimensions, time {time.time() - code_start_time:.3f}')
    print( 'data_ts size', data_ts.size() )
    # Precompute means and covs to compare to those of the Ising model sim windows.
    # data_means is 1 x num_subjects x num_nodes x num_windows
    # data_covs is 1 x num_subjects x num_nodes x num_nodes x num_windows
    param_means, param_covs = get_param_means_and_covs_slower(data_ts=data_ts, window=window_length)
    data_means = param_means[:,:,:num_nodes,:]
    data_covs = param_covs[:,:,:num_nodes,:num_nodes,:]
    # Since the windows are of consistent size, we can take the means of the mean and covs over all of them to get a mean and cov over all time points,
    # except for the truncated ones.
    data_fc = get_fc( s_mean=data_means.mean(dim=-1), s_cov=data_covs.mean(dim=-1) )
    print(f'computed data FC, time {time.time() - code_start_time:.3f}')
    # Add a singleton batch dimension along which we can broadcast to individual replications.
    return param_means, param_covs, data_fc

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    file_name_suffix = f'group_{data_subset}_threshold_{threshold_str}_window_{window_length}.pt'
    param_means, param_covs, data_fc = prep_ts_data(data_directory=data_directory, data_subset=data_subset, num_nodes=num_nodes, window_length=window_length)
    param_means_file =  os.path.join(output_directory, f'data_node_and_edge_means_{file_name_suffix}')
    torch.save(obj=param_means, f=param_means_file)
    param_covs_file =  os.path.join(output_directory, f'data_node_and_edge_covs_{file_name_suffix}')
    torch.save(obj=param_covs, f=param_covs_file)
    data_fc_file =  os.path.join(output_directory, f'data_fc_{file_name_suffix}')
    torch.save(obj=data_fc, f=data_fc_file)
    print(f'done, time {time.time() - code_start_time:.3f}')