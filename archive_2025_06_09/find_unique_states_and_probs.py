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
    for subject in range(num_subjects):
        subject_id = subject_list[subject]
        for rep in range(num_reps):
            rep_string = hcp.time_series_suffixes[rep]
            data_ts_file = hcp.get_time_series_file_path(directory_path=data_directory, subject_id=subject_id, time_series_suffix=rep_string)
            data_ts = hcp.load_matrix_from_binary(file_path=data_ts_file, dtype=torch.float, device=device).transpose(dim0=0, dim1=1)
            data_ts = ( data_ts > torch.median(data_ts, dim=1, keepdim=True).values )
            for time_point in range(num_time_points):
                state = data_ts[:,time_point].unsqueeze(dim=1)
                is_match = torch.all(unique_states[:,:num_unique_states] == state, dim=0)
                total_time_points += 1
                if is_match.count_nonzero() == 0:
                    unique_states[:,num_unique_states] = data_ts[:,time_point]
                    state_counts[num_unique_states] = 1
                    num_unique_states += 1
                    print(f'subject {subject}, ts {rep}, step {time_point}, unique states {num_unique_states} out of time points {total_time_points}, time {time.time() - code_start_time:.3f}')
                else:
                    match_index = is_match.nonzero().item()
                    state_counts[match_index] += 1
    # Create a new Tensor that is just large enough to contain the number of unique states actually encountered.
    unique_states = unique_states[:,:num_unique_states].clone()
    state_counts = state_counts[:num_unique_states].clone()
    states_file = os.path.join(output_directory, f'unique_states_group_{data_subset}_threshold_{threshold_str}.pt')
    torch.save(unique_states, states_file)
    print(f'saved {states_file}, time {time.time() - code_start_time:.3f}')
    counts_file = os.path.join(output_directory, f'unique_state_counts_group_{data_subset}_threshold_{threshold_str}.pt')
    torch.save(state_counts, counts_file)
    print(f'saved {counts_file}, time {time.time() - code_start_time:.3f}')
    # Now do per-individual counts.
    for rep in range(num_reps):
        rep_string = hcp.time_series_suffixes[rep]
        for subject in range(num_subjects):
            subject_id = subject_list[subject]
            data_ts_file = hcp.get_time_series_file_path(directory_path=data_directory, subject_id=subject_id, time_series_suffix=rep_string)
            data_ts = hcp.load_matrix_from_binary(file_path=data_ts_file, dtype=torch.float, device=device).transpose(dim0=0, dim1=1)
            data_ts = 2*( data_ts > torch.median(data_ts, dim=1, keepdim=True).values ).int() - 1
            state_counts.zero_()
            for time_point in range(num_time_points):
                state = data_ts[:,time_point].unsqueeze(dim=1)
                match_index = torch.all(unique_states == state, dim=0).nonzero().item()
                state_counts[match_index] += 1
            counts_file = os.path.join(output_directory, f'unique_state_counts_group_{data_subset}_threshold_{threshold_str}_subject_{subject}_rep_{rep}.pt')
            torch.save(state_counts, counts_file)
            num_nonzero_states = state_counts.count_nonzero()
            print(f'saved {counts_file}, found {num_nonzero_states} of {num_unique_states} unique states, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')