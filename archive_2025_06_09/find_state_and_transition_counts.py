import os
import torch
import time
import argparse
import hcpdatautils as hcp

parser = argparse.ArgumentParser(description="Compile a set of the unique states from the time series data. Find their counts and transition counts.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='training', help="the subset of subjects over which to search for unique states")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    data_ts_file = os.path.join(output_directory, f'data_ts_gt_median_{data_subset}.pt')
    data_ts = torch.load(data_ts_file)
    num_subjects, num_reps, num_nodes, num_time_points = data_ts.size()
    max_unique_states = num_subjects * num_reps * num_time_points + 1# Leave a free space for the putative new state.
    unique_states = torch.zeros( size=(max_unique_states, num_nodes), dtype=torch.bool, device=device )
    unique_state_counts = torch.zeros( size=(max_unique_states,), dtype=int_type, device=device )
    num_unique_states = 0
    state_transition_counts = torch.zeros( size=(max_unique_states, num_nodes), dtype=float_type, device=device )
    flip_number_counts = torch.zeros( size=(num_nodes+1,), dtype=int_type, device=device )# Allow for numbers of flips from 0 through 360.
    print(f'loaded {data_ts_file}, time {time.time() - code_start_time:.3f}')
    for subject in range(num_subjects):
        for rep in range(num_reps):
            step = 0
            state = data_ts[subject, rep, :, step]
            # unique_states[num_unique_states,:] = state
            # is_match = torch.all( unique_states[:num_unique_states+1,:] == state.unsqueeze(dim=0), dim=1 )
            # state_index = torch.nonzero(is_match)[0,0]# Get the first matching index.
            # unique_state_counts[state_index] += 1
            # num_unique_states += (state_index == num_unique_states).int()
            is_match = torch.all( unique_states[:num_unique_states,:] == state.unsqueeze(dim=0), dim=1 )
            if torch.count_nonzero(is_match) == 0:
                state_index = num_unique_states
                unique_states[state_index,:] = state
                unique_state_counts[state_index] = 1
                num_unique_states += 1
            else:
                state_index = torch.nonzero(is_match).item()
                unique_state_counts[state_index] += 1
            previous_state = state
            previous_state_index = state_index
            for step in range(1,num_time_points):
                state = data_ts[subject, rep, :, step]
                # unique_states[num_unique_states,:] = state
                # is_match = torch.all( unique_states[:num_unique_states+1,:] == state.unsqueeze(dim=0), dim=1 )
                # state_index = torch.nonzero(is_match)[0,0]# Get the first matching index.
                # unique_state_counts[state_index] += 1
                # num_unique_states += (state_index == num_unique_states).int()
                is_match = torch.all( unique_states[:num_unique_states,:] == state.unsqueeze(dim=0), dim=1 )
                if torch.count_nonzero(is_match) == 0:
                    state_index = num_unique_states
                    unique_states[state_index,:] = state
                    unique_state_counts[state_index] = 1
                    num_unique_states += 1
                else:
                    state_index = torch.nonzero(is_match).item()
                    unique_state_counts[state_index] += 1
                is_flipped = state != previous_state
                num_flipped = torch.count_nonzero(is_flipped)
                state_transition_counts[previous_state_index,is_flipped] += 1/num_flipped
                flip_number_counts[num_flipped] += 1
                previous_state = state
                previous_state_index = state_index
        print( f'subject {subject} of {num_subjects}, time {time.time() - code_start_time:.3f}'  )
    file_suffix = f'{data_subset}.pt'
    unique_states_file = os.path.join(output_directory, f'unique_states_{file_suffix}')
    torch.save( obj=unique_states[:num_unique_states,:].clone(), f=unique_states_file )
    print(f'saved {unique_states_file}, time {time.time() - code_start_time:.3f}')
    unique_state_counts_file = os.path.join(output_directory, f'unique_state_counts_{file_suffix}')
    torch.save( obj=unique_state_counts[:num_unique_states].clone(), f=unique_state_counts_file )
    print(f'saved {unique_state_counts_file}, time {time.time() - code_start_time:.3f}')
    state_transition_counts_file = os.path.join(output_directory, f'state_transition_counts_{file_suffix}')
    torch.save( obj=state_transition_counts[:num_unique_states,:].clone(), f=state_transition_counts_file )
    print(f'saved {state_transition_counts_file}, time {time.time() - code_start_time:.3f}')
    flip_number_counts_file = os.path.join(output_directory, f'flip_number_counts_{file_suffix}')
    torch.save(obj=flip_number_counts, f=flip_number_counts_file)
    print(f'saved {flip_number_counts_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')