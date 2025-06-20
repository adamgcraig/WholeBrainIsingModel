import math
import os
import torch
import time
import argparse

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-f", "--data_subset", type=str, default='training', help="the subset of subjects over which to search for unique states")
parser.add_argument("-t", "--threshold", type=str, default='median', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, or the string 'median', or the string 'none'")
parser.add_argument("-s", "--sim_length", type=int, default=1200, help="number of simulation steps between Euler updates")
parser.add_argument("-u", "--num_updates", type=int, default=10, help="number of Euler updates to do")
parser.add_argument("-p", "--num_parallel", type=int, default=5, help="number of parallel simulations to run")
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
num_parallel = args.num_parallel
print(f'num_parallel={num_parallel}')

# num_params = num_nodes + num_nodes*(num_nodes-1)//2
# We can use the quadratic formula to get back an expression for num_nodes in terms of num_params.
def num_params_to_num_nodes(num_params:int):
    return int(  ( math.sqrt(1 + 8*num_params) - 1 )/2  )

def diffs_to_rmse(diffs:torch.Tensor):
    return torch.sqrt(  torch.mean( torch.square(diffs), dim=-1 )  )

def balanced_metropolis_sim(params:torch.Tensor, state:torch.Tensor, sim_length:int, beta:torch.Tensor):
    num_parallel = state.size(dim=0)
    num_subjects, num_params = params.size()
    num_nodes = num_params_to_num_nodes(num_params)
    triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, dtype=int_type, device=device)
    triu_row = triu_indices[0,:]
    triu_col = triu_indices[1,:]
    h = params[:,:num_nodes].unsqueeze(dim=0)
    J = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    J[:,triu_row,triu_col] = params[:,num_nodes:]
    # print(J[:5,:5])
    J = J + J.transpose(dim0=-2, dim1=-1)
    # print(J[:5,:5])
    J = J.unsqueeze(dim=0)
    state_sum = torch.zeros( (num_parallel, num_subjects, num_nodes), dtype=float_type, device=device )
    cross_sum = torch.zeros( (num_parallel, num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    observed_values_sum = torch.zeros( (num_parallel, num_subjects, num_params), dtype=float_type, device=device )
    for _ in range(sim_length):
        # Randomize the order of nodes so that the simulation does not have a bias in favor of the state of a particular node.
        threshold_choice = torch.rand( size=(num_parallel, num_subjects, num_nodes), dtype=float_type, device=device )
        for node in torch.randperm(n=num_nodes, dtype=int_type, device=device):
            state[:,:,node] *= -2.0*(   threshold_choice[:,:,node] < torch.exp(  -2.0*( h[:,:,node] + torch.sum(J[:,:,node,:]*state, dim=-1) )*state[:,:,node]*beta  )   ).float() + 1.0
        # Only add in the new state after we have given each node a chance to flip.
        state_sum += state
        cross_sum += (state[:,:,:,None] * state[:,:,None,:])
    observed_values_sum[:,:,:num_nodes] = state_sum
    observed_values_sum[:,:,num_nodes:] = cross_sum[:,:,triu_row,triu_col]
    observed_values = observed_values_sum/sim_length
    return state, observed_values

def balanced_metropolis_sim_old(params:torch.Tensor, state:torch.Tensor, sim_length:int, beta:torch.Tensor):
    num_parallel = state.size(dim=0)
    num_subjects, num_params = params.size()
    num_nodes = num_params_to_num_nodes(num_params)
    triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, dtype=int_type, device=device)
    triu_row = triu_indices[0,:]
    triu_col = triu_indices[1,:]
    h = params[:,:num_nodes].unsqueeze(dim=0)
    J = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    J[:,triu_row,triu_col] = params[:,num_nodes:]
    # print(J[:5,:5])
    J = J + J.transpose(dim0=-2, dim1=-1)
    # print(J[:5,:5])
    J = J.unsqueeze(dim=0)
    state_sum = torch.zeros( (num_parallel, num_subjects, num_nodes), dtype=float_type, device=device )
    cross_sum = torch.zeros( (num_parallel, num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    observed_values_sum = torch.zeros( (num_parallel, num_subjects, num_params), dtype=float_type, device=device )
    beta = beta.unsqueeze(dim=1)# Try every beta with every subject by having the singleton dimension match up with the subject dimension. 
    for _ in range(sim_length):
        # Randomize the order of nodes so that the simulation does not have a bias in favor of the state of a particular node.
        threshold_choice = torch.rand( size=(num_parallel, num_subjects, num_nodes), dtype=float_type, device=device )
        for node in torch.randperm(n=num_nodes, dtype=int_type, device=device):
            state[:,:,node] *= -2.0*(   threshold_choice[:,:,node] < torch.exp(  -2.0*( h[:,:,node] + torch.sum(J[:,:,node,:]*state, dim=-1) )*state[:,:,node]*beta  )   ).float() + 1.0
        # Only add in the new state after we have given each node a chance to flip.
        state_sum += state
        cross_sum += (state[:,:,:,None] * state[:,:,None,:])
    observed_values_sum[:,:,:num_nodes] = state_sum
    observed_values_sum[:,:,num_nodes:] = cross_sum[:,:,triu_row,triu_col]
    observed_values = observed_values_sum/sim_length
    return state, observed_values

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    expected_values_file = os.path.join(output_directory, f'observable_expected_values_individual_{data_subset}_threshold_{threshold_str}.pt')
    expected_values = torch.load(expected_values_file)
    params = torch.clone(expected_values)
    num_subjects, num_params = params.size()
    num_nodes = num_params_to_num_nodes(num_params)
    # Randomly select a starting state.
    state = 2.0*torch.randint( low=0, high=2, size=(num_parallel, num_subjects, num_nodes), dtype=float_type, device=device ) - 1.0
    # print(state)
    # Do an initial sim to let the state settle into something typical of the system before we start sampling the observables for Euler updates.
    # state, observed_values = balanced_metropolis_sim(params=params, state=state, sim_length=sim_length, beta=beta)
    # diffs = expected_values - observed_values
    # rmse = diffs_to_rmse(diffs)
    # min_rmse = rmse.min()
    # mean_rmse = rmse.mean()
    # max_rmse = rmse.max()
    # print(f'sim 0, RMSE min {min_rmse:.3g}, mean {mean_rmse:.3g}, max {max_rmse:.3g}, time {time.time() - code_start_time:.3f}')
    # Do a series of sims, after each of which we adjust the parameters based on the disparity between observed and expected mean values of observables.
    expected_values = torch.unsqueeze(expected_values, dim=0)# Create a singleton dimension to match up with sims for the same subject at different beta.
    beta_starts = torch.full( size=(num_subjects,), fill_value=1.0/num_parallel, dtype=float_type, device=device )
    beta_ends = torch.full( size=(num_subjects,), fill_value=1.0, dtype=float_type, device=device )
    beta = torch.zeros( (num_parallel, num_subjects), dtype=float_type, device=device )
    best_betas = torch.zeros( (num_subjects,), dtype=float_type, device=device )
    best_observed_values = torch.zeros( (num_subjects, num_params), dtype=float_type, device=device )
    for sim in range(num_updates):
        for subject_index in range(num_subjects):
            beta[:,subject_index] = torch.linspace(start=beta_starts[subject_index], end=beta_ends[subject_index], steps=num_parallel, dtype=float_type, device=device)
        # print( 'testing beta values', beta.tolist() )
        state, observed_values = balanced_metropolis_sim(params=params, state=state, sim_length=sim_length, beta=beta)
        diffs = expected_values - observed_values
        rmse = diffs_to_rmse(diffs)
        # rmse_mean_over_subjects = rmse.mean(dim=-1)
        min_rmses, best_beta_indices = torch.min(rmse, dim=0)
        for subject_index in range(num_subjects):
            best_beta_index = best_beta_indices[subject_index]
            best_beta = beta[best_beta_index,subject_index]
            best_betas[subject_index] = best_beta
            best_observed_values[subject_index,:] = observed_values[best_beta_index,subject_index,:]
            if best_beta_index == 0:
                second_best_beta = beta[best_beta_index+1,subject_index]
                beta_start = max( best_beta - (second_best_beta - best_beta), 0.0 )
                beta_end = second_best_beta
            elif best_beta_index == (num_parallel-1):
                second_best_beta = beta[best_beta_index-1,subject_index]
                beta_start = second_best_beta
                beta_end = best_beta + (best_beta - second_best_beta)
            else:
                beta_start = beta[best_beta_index-1,subject_index]
                beta_end = beta[best_beta_index+1,subject_index]
            beta_starts[subject_index] = beta_start
            beta_ends[subject_index] = beta_end
        rmse_for_best_beta = rmse[best_beta_index,:]
        min_beta = best_betas.min()
        mean_beta = best_betas.mean()
        max_beta = best_betas.max()
        min_rmse = min_rmses.min()
        mean_rmse = min_rmses.mean()
        max_rmse = min_rmses.max()
        print(f'sim {sim+1}, best beta min {min_beta:.3g}, mean {mean_beta:.3g}, max {max_beta:.3g}, RMSE min {min_rmse:.3g}, mean {mean_rmse:.3g}, max {max_rmse:.3g}, time {time.time() - code_start_time:.3f}')
    file_suffix = f'beta_test_from_expected_individual_{data_subset}_threshold_{threshold_str}_parallel_{num_parallel}_updates_{num_updates}_sim_length_{sim_length}.pt'
    betas_file = os.path.join(output_directory, f'betas_{file_suffix}')
    torch.save(obj=best_betas, f=betas_file)
    print(f'saved {betas_file}, time {time.time() - code_start_time:.3f}')
    observed_values_file = os.path.join(output_directory, f'observed_values_{file_suffix}')
    torch.save(obj=best_observed_values, f=observed_values_file)
    print(f'saved {observed_values_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')