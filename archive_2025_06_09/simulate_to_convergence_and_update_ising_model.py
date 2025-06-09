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
parser.add_argument("-s", "--sim_length", type=int, default=10000, help="number of simulation steps between Euler updates")
parser.add_argument("-u", "--num_updates", type=int, default=1000, help="number of Euler updates to do")
parser.add_argument("-l", "--learning_rate", type=float, default=0.0, help="Euler step learning rate")
parser.add_argument("-b", "--beta", type=float, default=0.0043178, help="temperature factor that that scales the probability of energetically unfavorable flips, closer to 0 => more unfavorable flips")
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
beta = args.beta
print(f'beta={beta}')

# num_params = num_nodes + num_nodes*(num_nodes-1)//2
# We can use the quadratic formula to get back an expression for num_nodes in terms of num_params.
def num_params_to_num_nodes(num_params:int):
    return int(  ( math.sqrt(1 + 8*num_params) - 1 )/2  )

def metropolis_sim(params:torch.Tensor, state:torch.Tensor, sim_length:int, beta:float):
    num_params = params.numel()
    num_nodes = num_params_to_num_nodes(num_params)
    triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, dtype=int_type, device=device)
    triu_row = triu_indices[0,:]
    triu_col = triu_indices[1,:]
    h = params[:num_nodes]
    J = torch.zeros( (num_nodes, num_nodes), dtype=float_type, device=device )
    J[triu_row,triu_col] = params[num_nodes:]
    # print(J[:5,:5])
    J = J + J.transpose(dim0=0, dim1=1)
    # print(J[:5,:5])
    state_sum = torch.zeros_like(state)
    cross_sum = state_sum[:,None] * state_sum[None,:]
    num_flips = 0
    min_entropy_change = 10e10
    max_entropy_change = -10e10
    node_choice = torch.randint( low=0, high=num_nodes, size=(sim_length,), dtype=int_type, device=device )
    threshold_choice = torch.rand( size=(sim_length,), dtype=float_type, device=device )
    for step in range(sim_length):
        node = node_choice[step]
        entropy_change = -2.0*( h[node] + torch.sum(J[node,:]*state) )*state[node]*beta
        min_entropy_change = min(min_entropy_change, entropy_change)
        max_entropy_change = max(max_entropy_change, entropy_change)
        do_flip = threshold_choice[step] < torch.exp(entropy_change)
        num_flips += do_flip.int()
        state[node] *= -2.0*do_flip.float() + 1.0
        state_sum += state
        cross_sum += state[:,None] * state[None,:]
    observed_values_sum[:num_nodes] = state_sum
    observed_values_sum[num_nodes:] = cross_sum[triu_row,triu_col]
    observed_values = observed_values_sum/sim_length
    return state, observed_values, num_flips, min_entropy_change, max_entropy_change

def balanced_metropolis_sim(params:torch.Tensor, state:torch.Tensor, sim_length:int, beta:float):
    num_params = params.numel()
    num_nodes = num_params_to_num_nodes(num_params)
    triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, dtype=int_type, device=device)
    triu_row = triu_indices[0,:]
    triu_col = triu_indices[1,:]
    h = params[:num_nodes]
    J = torch.zeros( (num_nodes, num_nodes), dtype=float_type, device=device )
    J[triu_row,triu_col] = params[num_nodes:]
    # print(J[:5,:5])
    J = J + J.transpose(dim0=0, dim1=1)
    # print(J[:5,:5])
    state_sum = torch.zeros_like(state)
    cross_sum = state_sum[:,None] * state_sum[None,:]
    num_flips = 0
    min_entropy_change = 10e10
    max_entropy_change = -10e10
    for _ in range(sim_length):
        # Randomize the order of nodes so that the simulation does not have a bias in favor of the state of a particular node.
        node_choice = torch.randperm(n=num_nodes, dtype=int_type, device=device)
        threshold_choice = torch.rand( size=(num_nodes,), dtype=float_type, device=device )
        for node_index in range(num_nodes):
            node = node_choice[node_index]
            entropy_change = -2.0*( h[node] + torch.sum(J[node,:]*state) )*state[node]*beta
            min_entropy_change = min(min_entropy_change, entropy_change)
            max_entropy_change = max(max_entropy_change, entropy_change)
            do_flip = threshold_choice[node_index] < torch.exp(entropy_change)
            num_flips += do_flip.int()
            state[node] *= -2.0*do_flip.float() + 1.0
        # Only add in the new state after we have given each node a chance to flip.
        state_sum += state
        cross_sum += state[:,None] * state[None,:]
    observed_values_sum[:num_nodes] = state_sum
    observed_values_sum[num_nodes:] = cross_sum[triu_row,triu_col]
    observed_values = observed_values_sum/sim_length
    return state, observed_values, num_flips, min_entropy_change, max_entropy_change

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    expected_values_file = os.path.join(output_directory, f'observable_expected_values_group_{data_subset}_threshold_{threshold_str}.pt')
    expected_values = torch.load(expected_values_file)
    params = torch.clone(expected_values)
    observed_values_sum = torch.zeros_like(expected_values)
    num_params = params.numel()
    num_nodes = num_params_to_num_nodes(num_params)
    # Randomly select a starting state.
    state = 2.0*torch.randint( low=0, high=2, size=(num_nodes,), dtype=float_type, device=device ) - 1.0
    # print(state)
    # Do an initial sim to let the state settle into something typical of the system before we start sampling the observables for Euler updates.
    state, observed_values, num_flips, min_entropy_change, max_entropy_change = balanced_metropolis_sim(params=params, state=state, sim_length=sim_length, beta=beta)
    diffs = expected_values - observed_values
    rmse = torch.sqrt(  torch.mean( torch.square(diffs) )  )
    print(f'sim 0, flips {num_flips}, min entropy change {min_entropy_change:.3g}, max entropy change {max_entropy_change:.3g}, RMSE {rmse:.3g}, time {time.time() - code_start_time:.3f}')
    # Do a series of sims, after each of which we adjust the parameters based on the disparity between observed and expected mean values of observables.
    for sim in range(num_updates):
        # print(state)
        state, observed_values, num_flips, min_entropy_change, max_entropy_change = balanced_metropolis_sim(params=params, state=state, sim_length=sim_length, beta=beta)
        diffs = expected_values - observed_values
        rmse = torch.sqrt(  torch.mean( torch.square(diffs) )  )
        print(f'sim {sim+1}, flips {num_flips}, min entropy change {min_entropy_change:.3g}, max entropy change {max_entropy_change:.3g}, RMSE {rmse:.3g}, time {time.time() - code_start_time:.3f}')
        params += learning_rate * diffs
    new_params_file = os.path.join(output_directory, f'params_simple_euler_from_expected_group_{data_subset}_threshold_{threshold_str}_updates_{num_updates}_sim_length_{sim_length}_learning_rate_{learning_rate:.3g}.pt')
    torch.save(obj=params, f=new_params_file)
    print(f'saved {new_params_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')