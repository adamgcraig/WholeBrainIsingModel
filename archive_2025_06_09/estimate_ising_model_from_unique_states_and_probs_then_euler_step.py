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

    params = torch.zeros( (num_params,), dtype=float_type, device=device )
    log_counts = torch.log(state_counts)
    # Q is some arbitrary factor greater than 2^N that serves as our estimate of how many time points we would need to sample to observe every state at least once.
    num_steps = torch.sum(state_counts)
    log_total_minus_log_Q = torch.log(num_steps) - 2*num_nodes/math.log(2)
    # In the formula we derived, this scaling factor should be math.pow(2,num_nodes), but this leads to underflow error.
    # Since it is a fixed scaling factor we apply to all parameters equally, we can instead just try values until we get something reasonable.
    scaling_factor = num_observed_states * num_observed_states# math.pow(2,num_nodes)
    for m in range(num_nodes):
        # True -> +1, False -> -1
        is_plus = unique_states[m,:]
        is_minus = torch.logical_not(is_plus)
        num_plus = is_plus.count_nonzero()
        num_minus = is_minus.count_nonzero()
        # sign_imbalance = (num_minus - num_plus)*log_total_minus_log_Q
        unobserved_imbalance = 0
        params[m] = ( torch.sum(data_probs[is_plus]) - torch.sum(data_probs[is_minus]) )
        if params[m] == 0:
            print(f'params {m}, {params[m]:.3g}')
    for index in range( triu_row.numel() ):
        m = num_nodes + index
        i = triu_row[index]
        j = triu_col[index]
        # notxor(F,F) = T, notxor(F,T) = F, notxor(T,F) = F, notxor(T,T) = T
        #      -1*-1 = +1,      -1*+1 = -1,      +1*-1 = -1,      +1*+1 = +1
        is_minus = torch.logical_xor(unique_states[i,:], unique_states[j,:])
        is_plus = torch.logical_not(is_minus)
        num_plus = is_plus.count_nonzero()
        num_minus = is_minus.count_nonzero()
        # sign_imbalance = (num_minus - num_plus)*log_total_minus_log_Q
        params[m] = ( torch.sum(data_probs[is_plus]) - torch.sum(data_probs[is_minus]) )
        if params[m] == 0:
            print(f'params {m} ({i}, {j}), {params[m]:.3g}')
    print( 'params size', params.size() )
    params_file = os.path.join(output_directory, f'params_expected_value_group_{data_subset}_threshold_{threshold_str}.pt')
    torch.save(obj=params, f=params_file)
    print(f'saved {params_file}, time {time.time() - code_start_time:.3f}')

    # Check that the probabilities for the observed states match up.
    model_probs = torch.zeros_like(data_probs)
    state = torch.zeros_like(params)
    for state_index in range(num_observed_states):
        state[:num_nodes] = 2.0*unique_states[:,state_index].float() - 1.0
        cross_state = state[:num_nodes,None] * state[None,:num_nodes]
        state[num_nodes:] = cross_state[triu_row, triu_col]
        entropy = torch.sum(state * params)
        model_probs[state_index] = torch.exp(entropy)
        if model_probs[state_index].isinf():
            print(f'state {state_index}, model entropy {entropy:.3g}, data entropy {data_probs[state_index].log():.3g} model prob {model_probs[state_index]:.3g}, data prob {data_probs[state_index]:.3g}')
    model_probs /= torch.sum(model_probs)
    probs_mse = torch.mean( torch.square(model_probs - data_probs) )
    probs_kld = torch.sum( data_probs*torch.log(data_probs/model_probs) )
    abs_error = torch.abs(model_probs - data_probs)
    worst_diff = abs_error.max()
    best_diff = abs_error.min()
    print(f'computed model probabilities, MSE {probs_mse:.3g}, KL-divergence {probs_kld:.3g}, biggest difference {worst_diff:.3g}, smallest difference {best_diff:.3g}, time {time.time() - code_start_time:.3f}')
    probs_pair = torch.stack( (data_probs, model_probs), dim=0 )
    probs_pair_file = os.path.join(output_directory, f'data_expected_value_model_probability_pairs_group_{data_subset}_threshold_{threshold_str}.pt')
    torch.save(obj=probs_pair, f=probs_pair_file)
    print(f'saved {probs_pair_file}, time {time.time() - code_start_time:.3f}')

    # Simulate. Record the observed means of observables. Do an Euler step.
    num_epochs = 100000
    thresholds = torch.cumsum(data_probs)
    init_state_index = torch.nonzero( torch.rand( (1,), device=device ) >= threshold ).flatten()[0].item()
    state = torch.clone(unique_states[init_state_index])
    state_sum = torch.zeros_like(state)
    for epoch in range(num_epochs):
        state_sum.zero_()
        node_indices = torch.randint( (num_steps,), dtype=torch.int, device=device )
        for step in range(num_steps):
            next_state = state.clone()
print(f'done, time {time.time() - code_start_time:.3f}')