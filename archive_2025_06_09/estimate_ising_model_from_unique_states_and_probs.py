import math
import os
import torch
import time
import argparse
import hcpdatautils as hcp

parser = argparse.ArgumentParser(description="Estimate Ising model parameters from the set of unique states encountered and their probabilities.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-f", "--data_subset", type=str, default='training', help="the subset of subjects over which to search for unique states")
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

def binary_to_float_state(binary_state:torch.Tensor):
    return 2.0*binary_state.float() - 1.0

def extend_state(state:torch.Tensor):
    return torch.cat( ( state, (state[:,None] * state[None,:])[triu_row, triu_col] ), dim=0 )

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')

    states_file = os.path.join(output_directory, f'unique_states_group_{data_subset}_threshold_{threshold_str}.pt')
    unique_states = torch.load(states_file)
    print(f'loaded {states_file}, time {time.time() - code_start_time:.3f}')
    print( 'unique_states size', unique_states.size() )
    counts_file = os.path.join(output_directory, f'unique_state_counts_group_{data_subset}_threshold_{threshold_str}.pt')
    state_counts = torch.load(counts_file).float()
    data_probs = state_counts/torch.sum(state_counts)
    print(f'loaded {counts_file}, time {time.time() - code_start_time:.3f}')
    print( 'state_counts size', state_counts.size() )

    num_nodes, num_observed_states = unique_states.size()
    num_params = num_nodes + ( num_nodes*(num_nodes-1) )//2
    triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, dtype=torch.int, device=device)
    triu_row = triu_indices[0]
    triu_col = triu_indices[1]

    log_counts = torch.log(state_counts)
    log_counts_diff = torch.zeros( (num_params,), dtype=float_type, device=device )
    sign_imbalance = torch.zeros_like(log_counts_diff)
    for m in range(num_nodes):
        # True -> +1, False -> -1
        is_plus = unique_states[m,:]
        log_counts_diff[m] = torch.sum( log_counts * binary_to_float_state(is_plus) )
        sign_imbalance[m] = torch.logical_not(is_plus).count_nonzero() - is_plus.count_nonzero()
        # if raw_params[m] == 0:
        print(f'params {m}, , {log_counts_diff[m]:.3g}, {sign_imbalance[m]}')
    for index in range( triu_row.numel() ):
        m = num_nodes + index
        i = triu_row[index]
        j = triu_col[index]
        state = binary_to_float_state(unique_states[i,:])*binary_to_float_state(unique_states[j,:])
        is_plus = state > 0
        log_counts_diff[m] = torch.sum(log_counts * state)
        sign_imbalance[m] = torch.logical_not(is_plus).count_nonzero() - is_plus.count_nonzero()
        # if raw_params[m] == 0:
        print(f'params {m} ({i}, {j}), {log_counts_diff[m]:.3g}, {sign_imbalance[m]}')
        
    # To get the probabilities, we need to optimize two constants.
    # We could do a least squares regression over all points,
    # but, in practice, we find that the two most probable states are much more probable than the rest.
    # As such, making the model generate the right probabilities for these two states is an efficient way to make its behavior closer to that of the real system.
    log_data_probs = torch.log(data_probs)
    sorted_log_data_probs, sort_indices = torch.sort(input=log_data_probs, descending=True)
    log_prob_1 = sorted_log_data_probs[0]
    state1 = extend_state( binary_to_float_state(unique_states[ :, sort_indices[0] ]) )
    log_prob_2 = sorted_log_data_probs[1]
    state2 = extend_state( binary_to_float_state(unique_states[ :, sort_indices[1] ]) )
    counts_diff_times_state_1 = torch.sum(log_counts_diff * state1)
    sign_imbalance_times_state_1 = torch.sum(sign_imbalance * state1)
    counts_diff_times_state_2 = torch.sum(log_counts_diff * state2)
    sign_imbalance_times_state_2 = torch.sum(sign_imbalance * state2)
    denominator = (log_prob_1 * sign_imbalance_times_state_2 - log_prob_2 * sign_imbalance_times_state_1)
    log_total_minus_log_Q = (log_prob_1 * counts_diff_times_state_2 - log_prob_2 * counts_diff_times_state_1)/denominator
    scaling_factor = (counts_diff_times_state_1 * sign_imbalance_times_state_2 - counts_diff_times_state_2 * sign_imbalance_times_state_1)/denominator
    print(f'computed optimal log_total_minus_log_Q {log_total_minus_log_Q:.3g} and scaling_factor {scaling_factor:.3g}')
    params = (log_counts_diff - log_total_minus_log_Q*sign_imbalance)/scaling_factor
    params_file = os.path.join(output_directory, f'params_lstsq_limit_group_{data_subset}_threshold_{threshold_str}.pt')
    torch.save(obj=log_counts_diff, f=params_file)
    print(f'saved {params_file}, time {time.time() - code_start_time:.3f}')

    # Check that the probabilities for the observed states match up.
    model_probs = torch.zeros_like(data_probs)
    for state_index in range(num_observed_states):
        state = extend_state( binary_to_float_state(unique_states[:,state_index]) )
        entropy = torch.sum(state * params)
        model_probs[state_index] = torch.exp(entropy)
        # if model_probs[state_index].isinf():
        print(f'state {state_index}, model entropy {entropy:.3g}, data entropy {data_probs[state_index].log():.3g}, model prob {model_probs[state_index]:.3g}, data prob {data_probs[state_index]:.3g}')
    probs_mse = torch.mean( torch.square(model_probs - data_probs) )
    probs_kld = torch.sum( data_probs*torch.log(data_probs/model_probs) )
    abs_error = torch.abs(model_probs - data_probs)
    worst_diff = abs_error.max()
    best_diff = abs_error.min()
    data_max_prob = data_probs[ sort_indices[0] ]
    model_max_prob = model_probs[ sort_indices[0] ]
    data_second_prob = data_probs[ sort_indices[1] ]
    model_second_prob = model_probs[ sort_indices[1] ]
    print(f'computed model probabilities, MSE {probs_mse:.3g}, KL-divergence {probs_kld:.3g}, biggest difference {worst_diff:.3g}, smallest difference {best_diff:.3g}, biggest probability in data {data_max_prob:.3g}, in model {model_max_prob:.3g}, 2nd biggest data {data_second_prob:.3g}, in model {model_second_prob:.3g}, time {time.time() - code_start_time:.3f}')
    probs_pair = torch.stack( (data_probs, model_probs), dim=0 )
    probs_pair_file = os.path.join(output_directory, f'data_model_probability_pairs_lstsq_limit_group_{data_subset}_threshold_{threshold_str}.pt')
    torch.save(obj=probs_pair, f=probs_pair_file)
    print(f'saved {probs_pair_file}, time {time.time() - code_start_time:.3f}')

print(f'done, time {time.time() - code_start_time:.3f}')