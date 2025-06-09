import math
import os
import torch
import time
import argparse
import hcpdatautils as hcp

parser = argparse.ArgumentParser(description="Estimate independent model state probabilities from the set of unique states encountered and their probabilities.")
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
    return torch.cat(   (  state, torch.logical_not( torch.logical_xor(state[:,None], state[None,:]) )[triu_row, triu_col]  ), dim=0   )

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
    counts_total = torch.sum(state_counts)
    num_up = torch.zeros( (num_nodes,), dtype=torch.int, device=device )
    for m in range(num_nodes):
        # True -> +1, False -> -1
        num_up[m] = torch.sum(state_counts[ unique_states[m,:] ])
        # if raw_params[m] == 0:
        print(f'params {m}, , {num_up[m]:.3g}')
    prob_up = num_up.float()/counts_total
    prob_down = 1.0 - prob_up
    log_prob_up = torch.log(prob_up)
    log_prob_down = torch.log(prob_down)

    # Check that the probabilities for the observed states match up.
    log_data_probs = torch.log(data_probs)
    log_model_probs = torch.zeros_like(data_probs)
    for state_index in range(num_observed_states):
        state = unique_states[:,state_index]
        # If the observables are independent, then the probability of the state is just the product of the probabilities of the individual elements having the specified values.
        log_model_probs[state_index] = torch.sum(log_prob_up[state]) + torch.sum( log_prob_down[torch.logical_not(state)] )
        # if model_probs[state_index].isinf():
        print(f'state {state_index}, log model prob {log_model_probs[state_index]:.3g}, log data prob {log_data_probs[state_index]:.3g}')
    model_probs = torch.exp(log_model_probs)
    probs_mse = torch.mean( torch.square(model_probs - data_probs) )
    probs_kld = torch.sum( data_probs*(log_data_probs - log_model_probs) )
    abs_error = torch.abs(log_model_probs - data_probs)
    worst_diff = abs_error.max()
    best_diff = abs_error.min()
    print(f'computed model probabilities, MSE {probs_mse:.3g}, KL-divergence {probs_kld:.3g}, biggest difference {worst_diff:.3g}, smallest difference {best_diff:.3g}, time {time.time() - code_start_time:.3f}')
    probs_pair = torch.stack( (data_probs, log_model_probs), dim=0 )
    probs_pair_file = os.path.join(output_directory, f'data_model_probability_pairs_independent_group_{data_subset}_threshold_{threshold_str}.pt')
    torch.save(obj=probs_pair, f=probs_pair_file)
    print(f'saved {probs_pair_file}, time {time.time() - code_start_time:.3f}')

print(f'done, time {time.time() - code_start_time:.3f}')