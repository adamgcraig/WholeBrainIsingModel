import os
import torch
import time
import argparse

def get_avalanche_stats(data_ts:torch.Tensor, data_ts_mean:torch.Tensor, data_ts_std:torch.Tensor, threshold:float):
    # Flatten together the timeseries/subject and subject dimensions.
    binarized_ts = ( data_ts >= (data_ts_mean + threshold*data_ts_std) )
    num_ts, num_nodes, num_steps = binarized_ts.size()
    num_active = binarized_ts.count_nonzero(dim=-2)# number of nodes active at each time point
    has_any_active = num_active > 0
    is_transition = torch.not_equal(has_any_active[:,:-1], has_any_active[:,1:]) 
    avalanche_duration_counts = torch.zeros( size=(num_steps+1,), dtype=int_type, device=device )
    gap_duration_counts = torch.zeros_like(avalanche_duration_counts)
    all_durations = torch.arange(start=0, end=num_steps+1, dtype=int_type, device=device)
    max_size = num_steps*num_nodes
    avalanche_size_counts = torch.zeros( size=(max_size+1,), dtype=int_type, device=device )
    all_sizes = torch.arange(start=0, end=max_size+1, dtype=int_type, device=device)
    avalanche_unique_nodes_counts = torch.zeros( size=(num_nodes+1,), dtype=int_type, device=device )
    all_unique_nodes = torch.arange(start=0, end=num_nodes+1, dtype=int_type, device=device)
    for ts_index in range(num_ts):
        # The logical operations give us
        # is_avalanche_start that is true for an inactive point immediately before an active point.
        # is_avalanche_end that is true for an active point immediately before an inactive point.
        # When indexing a slice of time points in which an avalanche falls, we want
        # the first index to be an active time point immediately after an inactive time point
        # the last index to be an inactive time point immediately after an active time point.
        # The solution is to just add 1 to both.
        transitions = is_transition[ts_index,:].nonzero().flatten() + 1
        num_transitions = transitions.numel()
        # We need at least two transitions to have either a complete avalanche or a complete gap.
        # If we have neither, then we will not have anything with which to update the counts.
        if num_transitions >= 2:
            start_is_active = has_any_active[ts_index,0]
            if start_is_active:
                # If we start out active, then the first transition starts a gap
                # and ends an avalanche for which we have no start.
                gap_endpoints = transitions
                avalanche_endpoints = transitions[1:]
            else:
                # If we start out inactive, then the first tansition starts an avalanche
                # and ends a gap for which we have no start.
                avalanche_endpoints = transitions
                gap_endpoints = transitions[1:]
            # In both cases, for both avalanches and gaps,
            # we need to discard the last transition if it is a start without an end.
            if ( avalanche_endpoints.numel() % 2 ) != 0:
                avalanche_endpoints = avalanche_endpoints[:-1]
            if ( gap_endpoints.numel() % 2 ) != 0:
                gap_endpoints = gap_endpoints[:-1]
            # In each case, we now need to check that we actually have some avalanches or gaps.
            if avalanche_endpoints.numel() >= 2:
                avalanche_endpoints = avalanche_endpoints.reshape( (-1,2) )
                avalanche_starts = avalanche_endpoints[:,0]
                avalanche_ends = avalanche_endpoints[:,1]
                avalanche_durations = avalanche_ends - avalanche_starts
                # Now compute the size of each avalanche in terms of number of activated (region, time) cells and number of distinct nodes.
                num_avalanches = avalanche_starts.numel()
                avalanche_sizes = torch.zeros_like(avalanche_durations)
                avalanche_unique_nodes = torch.zeros_like(avalanche_durations)
                for avalanche_index in range(num_avalanches):
                    this_start = avalanche_starts[avalanche_index]
                    this_end = avalanche_ends[avalanche_index]
                    avalanche_sizes[avalanche_index] = num_active[ts_index,this_start:this_end].sum()
                    avalanche_unique_nodes[avalanche_index] = binarized_ts[ts_index,:,this_start:this_end].any(dim=-1).count_nonzero()
                # Add occurrences to our histogram counts.
                avalanche_duration_counts += torch.count_nonzero( all_durations[:,None] == avalanche_durations[None,:], dim=-1 )
                avalanche_size_counts += torch.count_nonzero( all_sizes[:,None] == avalanche_sizes[None,:], dim=-1 )
                avalanche_unique_nodes_counts += torch.count_nonzero( all_unique_nodes[:,None] == avalanche_unique_nodes[None,:], dim=-1 )
            if gap_endpoints.numel() >= 2:
                gap_endpoints = gap_endpoints.reshape( (-1,2) )
                gap_starts = gap_endpoints[:,0]
                gap_ends = gap_endpoints[:,1]
                gap_durations = gap_ends - gap_starts
                # Add occurrences to our histogram counts.
                gap_duration_counts += torch.count_nonzero( all_durations[:,None] == gap_durations[None,:], dim=-1 )
            print(f'threshold {threshold:.3g}, ts {ts_index}, gaps {gap_endpoints.numel()//2}, avalanches {avalanche_endpoints.numel()//2}')
    return avalanche_duration_counts, gap_duration_counts, avalanche_size_counts, avalanche_unique_nodes_counts

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Load and threshold unbinarized time series data. Make and save a Tensor counts such that counts[x] is the number of times x nodes flip in a single step.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-c", "--data_subset", type=str, default='all', help="'training', 'validation', 'testing', or 'all'")
    parser.add_argument("-d", "--file_name_fragment", type=str, default='as_is', help="part of the output file name between mean_state_[data_subset]_ or mean_state_product_[data_subset]_ and .pt")
    parser.add_argument("-f", "--min_threshold", type=float, default=-1, help="minimum threshold in std. dev.s")
    parser.add_argument("-g", "--max_threshold", type=float, default=5, help="minimum threshold in std. dev.s")
    parser.add_argument("-i", "--num_thresholds", type=int, default=100, help="number of thresholds to try")
    parser.add_argument("-j", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-k", "--training_subject_end", type=int, default=670, help="one past last training subject index")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_subset = args.data_subset
    print(f'data_subset={data_subset}')
    file_name_fragment = args.file_name_fragment
    print(f'file_name_fragment={file_name_fragment}')
    min_threshold = args.min_threshold
    print(f'min_threshold={min_threshold}')
    max_threshold = args.max_threshold
    print(f'max_threshold={max_threshold}')
    num_thresholds = args.num_thresholds
    print(f'num_thresholds={num_thresholds}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')

    data_ts_file = os.path.join(output_directory, f'data_ts_{data_subset}_{file_name_fragment}.pt')
    # Compute these stats using only the training data, since they inform the design of our ML models.
    # Flatten together scan and subject dims so we only need to keep track of one batch dim.
    data_ts = torch.flatten( torch.load(data_ts_file)[:,training_subject_start:training_subject_end,:,:], start_dim=0, end_dim=1 )
    num_ts, num_nodes, num_steps = data_ts.size()
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    num_nodes = data_ts.size(dim=-2)
    thresholds = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device)
    avalanche_sizes = torch.arange(start=0, end=num_nodes+1, step=1, dtype=float_type, device=device)# 0 through 360, inclusive
    num_sizes = avalanche_sizes.numel()
    avalanche_duration_counts = torch.zeros( size=(num_thresholds, num_steps+1), dtype=int_type, device=data_ts.device )
    gap_duration_counts = torch.zeros( size=(num_thresholds, num_steps+1), dtype=int_type, device=data_ts.device )
    avalanche_size_counts = torch.zeros( size=(num_thresholds, num_nodes*num_steps+1), dtype=int_type, device=data_ts.device )
    avalanche_unique_nodes_counts = torch.zeros( size=(num_thresholds, num_nodes+1), dtype=int_type, device=data_ts.device )
    for threshold_index in range(num_thresholds):
        threshold = thresholds[threshold_index]
        print(f'threshold {threshold_index+1} of {num_thresholds}: {threshold:.3g}')
        avalanche_duration_counts[threshold_index, :], gap_duration_counts[threshold_index, :], avalanche_size_counts[threshold_index, :], avalanche_unique_nodes_counts[threshold_index, :] = get_avalanche_stats(data_ts=data_ts, data_ts_mean=data_ts_mean, data_ts_std=data_ts_std, threshold=threshold)
    avalanche_duration_counts_file = os.path.join(output_directory, f'avalanche_duration_counts_{data_subset}_choices_{num_thresholds}.pt')
    torch.save(obj=avalanche_duration_counts, f=avalanche_duration_counts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {avalanche_duration_counts_file}')
    gap_duration_counts_file = os.path.join(output_directory, f'gap_duration_counts_{data_subset}_choices_{num_thresholds}.pt')
    torch.save(obj=gap_duration_counts, f=gap_duration_counts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {gap_duration_counts_file}')
    # To make the file size more reasonable, remove sizes beyond the largest one that actually occurs.
    # Use clone() so we do not save a view + all underlying data.
    last_avalanche_size = torch.any(avalanche_size_counts > 0, dim=0).nonzero().flatten()[-1]
    avalanche_size_counts = avalanche_size_counts[:,:last_avalanche_size+1].clone()
    avalanche_size_counts_file = os.path.join(output_directory, f'avalanche_size_counts_{data_subset}_choices_{num_thresholds}.pt')
    torch.save(obj=avalanche_size_counts, f=avalanche_size_counts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {avalanche_size_counts_file}')
    avalanche_unique_nodes_counts_file = os.path.join(output_directory, f'avalanche_unique_nodes_counts_{data_subset}_choices_{num_thresholds}.pt')
    torch.save(obj=avalanche_unique_nodes_counts, f=avalanche_unique_nodes_counts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {avalanche_unique_nodes_counts_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')