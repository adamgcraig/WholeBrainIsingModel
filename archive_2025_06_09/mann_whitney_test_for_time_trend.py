import os
import torch
import time
import argparse
import hcpdatautils as hcp
from isingutilsslow import binarize_data_ts

parser = argparse.ArgumentParser(description="Use a Mann-Whitney U-test to check whether -1 and +1 are evenly distributed along each time series.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='training', help="which list of subjects to use, either training, validation, testing, or all")
parser.add_argument("-t", "--threshold", type=str, default='0.1', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, or the string 'median'")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes to model")
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

def get_num_nan(mat:torch.Tensor):
    return torch.count_nonzero( torch.isnan(mat) )

def prep_ts_data(data_directory:str, data_subset:str, num_nodes:int):
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
    data_ts = binarize_data_ts(data_ts=data_ts, step_dim=-2, threshold=threshold).transpose(dim0=-2, dim1=-1)
    print(f'binarized time series with threshold {threshold_str} and rearranged dimensions, time {time.time() - code_start_time:.3f}')
    print( 'data_ts size', data_ts.size() )
    return data_ts

# See https://datatab.net/tutorial/mann-whitney-u-test
with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    file_name_suffix = f'mann_whitney_individual_data_ts_{data_subset}_threshold_{threshold_str}_nodes_{num_nodes}.pt'
    data_ts = prep_ts_data(data_directory=data_directory, data_subset=data_subset, num_nodes=num_nodes)
    is_neg = data_ts < 0
    is_pos = data_ts > 0
    n_neg = torch.count_nonzero(is_neg, dim=-1)
    n_pos = torch.count_nonzero(is_pos, dim=-1)
    U_max = n_neg * n_pos
    # We need to broadcast the ranks over three dimensions: time series of subject, subject, and node.
    ranks = torch.arange( data_ts.size(dim=-1), dtype=float_type, device=device ).unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
    R_neg = torch.sum( is_neg.float() * ranks, dim=-1 )
    R_pos = torch.sum( is_pos.float() * ranks, dim=-1 )
    U_neg = U_max + n_neg * (n_neg+1) / 2 - R_neg
    U_pos = U_max + n_pos * (n_pos+1) / 2 - R_pos
    U = torch.minimum(U_neg, U_pos)
    U_mean = U_max/2
    U_std = torch.sqrt(  ( (n_neg + n_pos + 1)*n_neg*n_pos )/12  )
    z = (U - U_mean)/U_std
    U_file =  os.path.join(output_directory, f'U_{file_name_suffix}')
    torch.save(obj=U, f=U_file)
    print(f'saved {U_file}')
    z_file =  os.path.join(output_directory, f'z_{file_name_suffix}')
    torch.save(obj=z, f=z_file)
    print(f'saved {z_file}')
    print(f'done, time {time.time() - code_start_time:.3f}')