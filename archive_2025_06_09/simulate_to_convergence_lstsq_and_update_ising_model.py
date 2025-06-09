import math
import os
import torch
import time
import argparse

parser = argparse.ArgumentParser(description="Simulate an Ising model while tracking the expected values of its observables. Compare to expected values from data. Do an Euler step update. Repeat.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-f", "--data_subset", type=str, default='training', help="the subset of subjects over which to search for unique states")
parser.add_argument("-t", "--threshold", type=str, default='median', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, or the string 'median', or the string 'none'")
parser.add_argument("-s", "--sim_length", type=int, default=1000000, help="number of simulation steps between Euler updates")
parser.add_argument("-u", "--num_updates", type=int, default=1000, help="number of Euler updates to do")
parser.add_argument("-l", "--learning_rate", type=float, default=0.1, help="Euler step learning rate")
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
sim_length = args.sim_length
print(f'sim_length={sim_length}')
num_updates = args.num_updates
print(f'num_updates={num_updates}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    expected_values_file = os.path.join(output_directory, f'params_expected_value_group_{data_subset}_threshold_{threshold_str}.pt')
    expected_values = torch.load(expected_values_file)
    params_file = os.path.join(output_directory, f'params_lstsq_limit_group_{data_subset}_threshold_{threshold_str}.pt')
    params = torch.clone(expected_values)
    observed_values_sum = torch.zeros_like(expected_values)
    num_params = params.numel()
    # num_params = num_nodes + num_nodes*(num_nodes-1)//2
    # We can use the quadratic formula to get back an expression for num_nodes in terms of num_params.
    num_nodes = int(  ( math.sqrt(1 + 8*num_params) - 1 )/2  )
    triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, dtype=int_type, device=device)
    triu_row = triu_indices[0,:]
    triu_col = triu_indices[1,:]
    h = params[:num_nodes]
    J = torch.zeros( (num_nodes, num_nodes), dtype=float_type, device=device )
    J[triu_row,triu_col] = params[num_nodes:]
    J = J + J.transpose(dim0=0, dim1=1)
    J = J
    # All-down is the most frequently occurring state in the data.
    state = torch.full( size=(num_nodes,), fill_value=-1.0, dtype=float_type, device=device )
    state_sum = torch.zeros_like(state)
    cross_sum = state_sum[:,None] * state_sum[None,:]
    for sim in range(num_updates):
        node_choice = torch.randint( low=0, high=num_nodes, size=(sim_length,), dtype=int_type, device=device )
        threshold_choice = torch.rand( size=(sim_length,), dtype=float_type, device=device )
        state_sum.zero_()
        cross_sum.zero_()
        num_flips = 0
        for step in range(sim_length):
            node = node_choice[step]
            do_flip = threshold_choice[step] < torch.exp(  -2.0*( h[node] + torch.sum(J[node,:]*state) )*state[node]  )
            num_flips += do_flip.int()
            state[node] *= -2.0*do_flip.float() + 1.0
            state_sum += state
            cross_sum += state[:,None] * state[None,:]
        observed_values_sum[:num_nodes] = state_sum
        observed_values_sum[num_nodes:] = cross_sum[triu_row, triu_col]
        diffs = expected_values - observed_values_sum / sim_length
        params += learning_rate * diffs
        h = params[:num_nodes]
        J = torch.zeros( (num_nodes, num_nodes), dtype=float_type, device=device )
        J[triu_row,triu_col] = params[num_nodes:]
        J = J + J.transpose(dim0=0, dim1=1)
        rmse = torch.sqrt(  torch.mean( torch.square(diffs) )  )
        print(f'sim {sim+1}, flips {num_flips}, RMSE {rmse:.3g}, time {time.time() - code_start_time:.3f}')
    new_params_file = os.path.join(output_directory, f'params_simple_euler_from_expected_group_{data_subset}_threshold_{threshold_str}_updates_{num_updates}_sim_length_{sim_length}_learning_rate_{learning_rate:.3g}.pt')
    torch.save(obj=params, f=new_params_file)
    print(f'saved {new_params_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')