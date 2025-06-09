import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
    parser.add_argument("-a", "--data_directory", type=str, default='/data/agcraig', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='/data/agcraig', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_ts_file_fragment", type=str, default='data_ts_all_as_is', help="data time series pickle (.pt) file excluding path and file extension")
    parser.add_argument("-d", "--threshold", type=float, default=0, help="number of standard deviations above the mean at which to binarize each region time series, can be 0 or negative")
    parser.add_argument("-e", "--training_subject_start", type=int, default=0, help="index of first training subject")
    parser.add_argument("-f", "--training_subject_end", type=int, default=670, help="one after index of last training subject")

    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_ts_file_fragment = args.data_ts_file_fragment
    print(f'data_ts_file_fragment={data_ts_file_fragment}')
    threshold = args.threshold
    print(f'threshold={threshold}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')
    
    def get_individual_and_group_time_series(data_directory:str, data_ts_file_fragment:str, threshold:float, training_subject_start:int, training_subject_end:int):
        print('loading data time series')
        data_ts_file = os.path.join(data_directory, f'{data_ts_file_fragment}.pt')
        data_ts = torch.load(data_ts_file)[:,training_subject_start:training_subject_end,:,:].clone()
        print( f'time {time.time()-code_start_time:.3f}, selected training subjects, so data_ts now size', data_ts.size() )
        data_std, data_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
        data_ts = data_ts >= (data_mean + threshold*data_std)
        print(f'time {time.time()-code_start_time:.3f}, binarized data_ts at threshold mean + {threshold:.3g} std.dev.')
        group_data_ts = torch.permute( input=data_ts, dims=(1,0,3,2) ).flatten(start_dim=0, end_dim=-2)
        print( f'time {time.time()-code_start_time:.3f}, rearranged dimensions so that the node dimension is the last size and flattened together the others, now size', group_data_ts.size() )
        return data_ts, group_data_ts
    
    def compile_unique_state_list(group_data_ts:torch.Tensor):
        is_match = torch.all( group_data_ts.unsqueeze(dim=1) == group_data_ts.unsqueeze(dim=0), dim=-1 )
        print( f'time {time.time()-code_start_time:.3f}, compared all time points, is_match size ', is_match.size() )
        # If state i is the first match for state i, then no column prior to i in row i will be True.
        is_first = torch.count_nonzero( torch.tril(input=is_match, diagonal=-1) ) == 0
        print( f'time {time.time()-code_start_time:.3f}, found the first instance of each state, is_first size ', is_first.size() )
        data_states = group_data_ts[is_first,:].clone()
        print( f'time {time.time()-code_start_time:.3f}, selected out all unique states, data_states size ', data_states.size() )
        group_data_counts = torch.count_nonzero( input=is_match[is_first,:], dim=-1 )
        print( f'time {time.time()-code_start_time:.3f}, found counts for all unique states, group_data_counts size ', group_data_counts.size() )
        return data_states, group_data_counts
    
    def compile_unique_state_list_low_memory(group_data_ts:torch.Tensor):
        data_states = torch.zeros_like(group_data_ts)
        num_time_points = group_data_ts.size(dim=0)
        group_data_counts = torch.zeros( size=(num_time_points,), dtype=int_type, device=group_data_ts.device )
        num_unique = 0
        num_remaining = group_data_ts.size(dim=0)
        while num_remaining > 0:
            current_state = group_data_ts[0:1,:]
            is_match = torch.all(group_data_ts == current_state, dim=-1)
            data_states[num_unique:(num_unique+1),:] = current_state
            group_data_counts[num_unique] = torch.count_nonzero(is_match)
            group_data_ts = group_data_ts[torch.logical_not(is_match),:]
            num_unique += 1
            num_remaining = group_data_ts.size(dim=0)
        # Clone so that the underlying memory is the right size, not the larger max possible size.
        data_states = data_states[:num_unique,:].clone()
        print( f'time {time.time()-code_start_time:.3f}, selected out all unique states, data_states size ', data_states.size() )
        group_data_counts = group_data_counts[:num_unique].clone()
        print( f'time {time.time()-code_start_time:.3f}, found counts for all unique states, group_data_counts size ', group_data_counts.size() )
        return data_states, group_data_counts
    
    def get_individual_state_counts(data_ts:torch.Tensor, data_states:torch.Tensor):
        num_ts = data_ts.size(dim=0)
        num_subjects = data_ts.size(dim=1)
        num_states = data_states.size(dim=0)
        individual_data_counts = torch.zeros( size=(num_ts, num_subjects, num_states), dtype=int_type, device=device )
        for state in range(num_states):
            individual_data_counts[:,:,state] = torch.count_nonzero(  torch.all( data_ts == data_states[state,:].unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=-1), dim=-2 ), dim=-1  )
        print( f'time {time.time()-code_start_time:.3f}, found counts for all unique states for each individual time series, individual_data_counts size ', individual_data_counts.size() )
        return individual_data_counts
    
    data_ts, group_data_ts = get_individual_and_group_time_series(data_directory=data_directory, data_ts_file_fragment=data_ts_file_fragment, threshold=threshold, training_subject_start=training_subject_start, training_subject_end=training_subject_end)
    group_data_ts_file = os.path.join(output_directory, f'group_data_ts_mean_std_{threshold:.3g}.pt')
    torch.save(obj=group_data_ts, f=group_data_ts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {group_data_ts_file}')
    data_states, group_data_counts = compile_unique_state_list_low_memory(group_data_ts=group_data_ts)
    data_states_file = os.path.join(output_directory, f'data_states_mean_std_{threshold:.3g}.pt')
    torch.save(obj=data_states, f=data_states_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {data_states_file}')
    group_data_counts_file = os.path.join(output_directory, f'group_data_counts_mean_std_{threshold:.3g}.pt')
    torch.save(obj=group_data_counts, f=group_data_counts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {group_data_counts_file}')
    individual_data_counts = get_individual_state_counts(data_ts=data_ts, data_states=data_states)
    individual_data_counts_file = os.path.join(output_directory, f'individual_data_counts_mean_std_{threshold:.3g}.pt')
    torch.save(obj=individual_data_counts, f=individual_data_counts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {individual_data_counts_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')