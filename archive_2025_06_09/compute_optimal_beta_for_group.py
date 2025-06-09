import math
import os
import torch
import time
import argparse
import isingmodel
from isingmodel import IsingModel

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-f", "--data_subset", type=str, default='training', help="the subset of subjects over which to search for unique states")
parser.add_argument("-s", "--sim_length", type=int, default=4800, help="number of simulation steps between Euler updates")
parser.add_argument("-u", "--num_updates", type=int, default=3, help="number of Euler updates to do")
parser.add_argument("-p", "--num_parallel", type=int, default=1000, help="number of parallel simulations to run")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
num_updates = args.num_updates
print(f'num_updates={num_updates}')
num_parallel = args.num_parallel
print(f'num_parallel={num_parallel}')

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    
    # Load the mean values and FC of the data.
    group_state_mean_file = os.path.join(output_directory, f'mean_state_group_{data_subset}.pt')
    group_state_mean = torch.load(group_state_mean_file)
    print(f'loaded {group_state_mean_file}, time {time.time() - code_start_time:.3f}')
    group_product_mean_file = os.path.join(output_directory, f'mean_state_product_group_{data_subset}.pt')
    group_product_mean = torch.load(group_product_mean_file)
    print(f'loaded {group_product_mean_file}, time {time.time() - code_start_time:.3f}')
    # Compute the FC of the data.
    group_fc_file = os.path.join(output_directory, f'fc_group_{data_subset}.pt')
    group_fc = torch.load(group_fc_file)
    print(f'loaded {group_fc_file}, time {time.time() - code_start_time:.3f}')

    # Initialize the Ising model using the means.
    num_nodes = group_state_mean.size(dim=-1)
    group_ising_model = IsingModel(batch_size=num_parallel, num_nodes=num_nodes, dtype=float_type, device=device)
    group_ising_model.init_with_target_means(target_mean=group_state_mean, target_product_mean=group_product_mean)

    # Choose a range of possible beta values.
    beta_start = 0.0
    beta_end = 1.0
    best_beta = 1.0
    for sim in range(num_updates):
        beta = torch.linspace(start=beta_start, end=beta_end, steps=num_parallel, dtype=float_type, device=device)
        # print( 'beta', beta.tolist() )
        # Add a singleton dimension at the end so that it knows to broadcast the same beta to every node in a model.
        group_ising_model.beta = beta.unsqueeze(dim=-1)
        # Run the Ising model, and find the sim FC.
        sim_group_fc = group_ising_model.simulate_and_record_fc(num_steps=sim_length)
        # Compare the sim FC to the data FC.
        fc_rmse = isingmodel.get_pairwise_rmse(mat1=group_fc, mat2=sim_group_fc)
        min_fc_rmse_index = fc_rmse.argmin()
        min_fc_rmse = fc_rmse[min_fc_rmse_index]
        best_beta = beta[min_fc_rmse_index]
        # print( 'RMSE', fc_rmse.tolist() )
        print(f'best beta {best_beta:.3g}')
        print(f'FC RMSE min {min_fc_rmse:.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}, time {time.time() - code_start_time:.3f}')
        fc_correlation = isingmodel.get_pairwise_correlation(mat1=group_fc, mat2=sim_group_fc)
        # print( 'correlation', fc_correlation.tolist() )
        print(f'FC correlation min {fc_correlation.min():.3g}, mean {fc_correlation.mean():.3g}, max {fc_correlation.max():.3g}, time {time.time() - code_start_time:.3f}')
        # Narrow down the range in which we will search for the best beta.
        if min_fc_rmse_index == 0:
            if best_beta == 0.0:
                # If the best value is 0,
                # search the interval between 0 and the next highest value.
                beta_start = 0.0
                beta_end = beta[min_fc_rmse_index+1]
            else:
                # If the best value is at the lower end of the interval and is non-0,
                # search the interval from 0 to the this value.
                beta_start = 0.0
                beta_end = best_beta
        elif min_fc_rmse_index == num_parallel-1:
            # If the best value is at the upper end of the interval,
            # search a new interval with the same width starting at this upper limit.
            beta_width = best_beta - beta_start
            beta_start = best_beta
            beta_end = best_beta + beta_width
        else:
            # If the best value is not at either end of the interval,
            # create a new interval centered on this value.
            beta_start = beta[min_fc_rmse_index-1]
            beta_end = beta[min_fc_rmse_index+1]
    
    file_suffix = f'group_{data_subset}_parallel_{num_parallel}_sims_{num_updates}_steps_{sim_length}.pt'
    # Set beta to the single best value from the last iteration, and save the model.
    group_ising_model.beta = best_beta.item()
    group_ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    torch.save(obj=group_ising_model, f=group_ising_model_file)
    # Run all the copies of the model again with the best beta to test reproducibility.
    print( f'testing with beta {group_ising_model.beta:.3g}' )
    # Run the Ising model, and find the sim FC.
    sim_group_fc = group_ising_model.simulate_and_record_fc(num_steps=sim_length)
    # Compare the sim FC to the data FC.
    fc_rmse = isingmodel.get_pairwise_rmse(mat1=group_fc, mat2=sim_group_fc)
    # print( 'RMSE', fc_rmse.tolist() )
    print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}, time {time.time() - code_start_time:.3f}')
    fc_correlation = isingmodel.get_pairwise_correlation(mat1=group_fc, mat2=sim_group_fc)
    # print( 'correlation', fc_correlation.tolist() )
    print(f'FC correlation min {fc_correlation.min():.3g}, mean {fc_correlation.mean():.3g}, max {fc_correlation.max():.3g}, time {time.time() - code_start_time:.3f}')
    # Save the simulation FC, FC RMSE, and FC correlation.
    print(f'saved {group_ising_model_file}, time {time.time() - code_start_time:.3f}')
    sim_group_fc_file = os.path.join(output_directory, f'sim_fc_{file_suffix}')
    torch.save(obj=sim_group_fc, f=sim_group_fc_file)
    print(f'saved {sim_group_fc_file}, time {time.time() - code_start_time:.3f}')
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{file_suffix}')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'saved {fc_rmse_file}, time {time.time() - code_start_time:.3f}')
    fc_correlation_file = os.path.join(output_directory, f'fc_correlation_{file_suffix}')
    torch.save(obj=fc_correlation, f=fc_correlation_file)
    print(f'saved {fc_correlation_file}, time {time.time() - code_start_time:.3f}')

print(f'done, time {time.time() - code_start_time:.3f}')