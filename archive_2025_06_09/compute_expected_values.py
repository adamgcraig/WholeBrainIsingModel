import os
import torch
import time
import argparse
import hcpdatautils as hcp

parser = argparse.ArgumentParser(description="What unique states occur in the binarized time series? How many times does each occur?")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='training', help="the subset of subjects over which to search for unique states")
parser.add_argument("-t", "--threshold", type=str, default='median', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, or the string 'median', or the string 'none'.")
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
elif threshold_str == 'none':
    threshold = threshold_str
else:
    threshold = float(threshold_str)
print(f'threshold={threshold_str}')

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    data_ts_file = os.path.join(data_directory, f'data_ts_{data_subset}.pt')
    # Swap dimensions so that nodes are in the second-to-last dimension, time points in the last.
    num_reps = hcp.time_series_per_subject
    subject_list = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_list)
    num_time_points = hcp.num_time_points
    print(f'loaded all time series, time {time.time() - code_start_time:.3f}')
    max_num_states = num_subjects * num_reps * num_time_points
    unique_states = torch.zeros( (hcp.num_brain_areas, max_num_states), dtype=torch.bool, device=device )
    state_counts = torch.zeros( (max_num_states,), dtype=torch.int, device=device )
    num_unique_states = 0
    total_time_points = 0
    triu_indices = torch.triu_indices(row=hcp.num_brain_areas, col=hcp.num_brain_areas, offset=1, dtype=int_type, device=device)
    triu_row = triu_indices[0,:]
    triu_col = triu_indices[1,:]
    num_params = hcp.num_brain_areas + ( hcp.num_brain_areas * (hcp.num_brain_areas-1) )//2
    params_for_subject = torch.zeros( (num_subjects, num_params), dtype=float_type, device=device )
    for subject in range(num_subjects):
        subject_id = subject_list[subject]
        for rep in range(num_reps):
            rep_string = hcp.time_series_suffixes[rep]
            data_ts_file = hcp.get_time_series_file_path(directory_path=data_directory, subject_id=subject_id, time_series_suffix=rep_string)
            data_ts = hcp.load_matrix_from_binary(file_path=data_ts_file, dtype=float_type, device=device).transpose(dim0=0, dim1=1)
            num_nodes, num_time_points = data_ts.size()
            # print( 'data_ts size', data_ts.size() )
            data_ts = 2.0*( data_ts > torch.median(data_ts, dim=-1, keepdim=True).values ).float() - 1.0
            data_ts_mean = torch.mean(data_ts, dim=-1)
            data_ts_outer_product = data_ts[:,None,:] * data_ts[None,:,:]
            data_ts_outer_product_mean = torch.mean(data_ts_outer_product, dim=-1)
            data_ts_outer_product_mean_triu = data_ts_outer_product_mean[triu_row, triu_col]
            params_for_subject[subject,:] += torch.cat( (data_ts_mean, data_ts_outer_product_mean_triu), dim=0 )
            print( f'subject {subject} of {num_subjects}, ts {subject_id} {rep_string}, nonzero means {data_ts_mean.count_nonzero()} mean outer product mean {torch.mean(data_ts_outer_product_mean_triu):.3g}'  )
    params_for_subject /= num_reps
    params_for_group = torch.mean(params_for_subject, dim=0)
    print( f'num nonzero group means of regions {torch.count_nonzero(params_for_group[:num_nodes])}' )
    print( f'mean mean of products for group {torch.mean(params_for_group[num_nodes:])}' )
    params_for_subject_file = os.path.join(output_directory, f'observable_expected_values_individual_{data_subset}_threshold_{threshold_str}.pt')
    torch.save(obj=params_for_subject, f=params_for_subject_file)
    print(f'saved {params_for_subject_file}, time {time.time() - code_start_time:.3f}')
    params_for_group_file = os.path.join(output_directory, f'observable_expected_values_group_{data_subset}_threshold_{threshold_str}.pt')
    torch.save(obj=params_for_group, f=params_for_group_file)
    print(f'saved {params_for_group_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')