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
parser.add_argument("-f", "--data_subset", type=str, default='validation', help="the subset of subjects over which to search for unique states")
parser.add_argument("-m", "--model_type", type=str, default='group', help="group or individual")
parser.add_argument("-s", "--sim_length", type=int, default=4800, help="number of simulation steps between Euler updates")
parser.add_argument("-u", "--num_updates", type=int, default=5, help="number of Euler updates to do")
parser.add_argument("-p", "--num_parallel", type=int, default=1000, help="number of parallel simulations to run")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
model_type = args.model_type
print(f'model_type={model_type}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
num_updates = args.num_updates
print(f'num_updates={num_updates}')
num_parallel = args.num_parallel
print(f'num_parallel={num_parallel}')

def get_quarter_connections(triu_pairs:torch.Tensor, num_nodes:int):
    batch_size = triu_pairs.size(dim=0)
    triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=triu_pairs.device)
    square_mat = torch.zeros( size=(batch_size, num_nodes, num_nodes), dtype=triu_pairs.dtype, device=triu_pairs.device )
    square_mat[:,triu_rows,triu_cols] = triu_pairs
    square_mat[:,triu_cols,triu_rows] = triu_pairs
    half_nodes = num_nodes//2
    first_quad = square_mat[:,:half_nodes,:half_nodes]
    last_quad = square_mat[:,half_nodes:,half_nodes:]
    half_triu_rows, half_triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=half_nodes, device=triu_pairs.device)
    first_triu = first_quad[:,half_triu_rows,half_triu_cols]
    last_triu = last_quad[:,half_triu_rows,half_triu_cols]
    return torch.cat( tensors=(first_triu, last_triu), dim=0 )

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    
    # Load the mean values and FC of the data.
    file_suffix = f'{model_type}_{data_subset}.pt'
    mean_state_file = os.path.join(output_directory, f'mean_state_{file_suffix}')
    mean_state = torch.load(mean_state_file)
    num_subjects, num_nodes = mean_state.size()
    mean_state = torch.unflatten( input=mean_state, dim=-1, sizes=(2,num_nodes//2) ).flatten(start_dim=0, end_dim=1)
    mean_state = mean_state.repeat( (num_parallel, 1) )
    print(f'loaded {mean_state_file}, time {time.time() - code_start_time:.3f}')
    print( 'folded group_state_mean size', mean_state.size() )
    mean_state_product_file = os.path.join(output_directory, f'mean_state_product_{file_suffix}')
    mean_state_product = torch.load(mean_state_product_file)
    mean_state_product = get_quarter_connections(triu_pairs=mean_state_product, num_nodes=num_nodes)
    mean_state_product = mean_state_product.repeat( (num_parallel,1) )
    print(f'loaded {mean_state_product_file}, time {time.time() - code_start_time:.3f}')
    print( 'upper and lower quadrants of mean_state_product size', mean_state_product.size() )
    # Compute the FC of the data.
    data_fc_file = os.path.join(output_directory, f'fc_{file_suffix}')
    data_fc = torch.load(data_fc_file)
    data_fc = get_quarter_connections(triu_pairs=data_fc, num_nodes=num_nodes)
    data_fc = data_fc.repeat( (num_parallel,1) )
    print(f'loaded {data_fc_file}, time {time.time() - code_start_time:.3f}')
    print( 'upper and lower quadrants of data_fc size', data_fc.size() )

    # Initialize the Ising model using the means.
    batch_size, num_nodes = mean_state.size()
    model = IsingModel(batch_size=batch_size, num_nodes=num_nodes, dtype=float_type, device=device)
    model.init_with_target_means(target_mean=mean_state, target_product_mean=mean_state_product)

    # Choose a range of possible beta values.
    # Create a different set of beta values for each subject.
    # They start out with [0, 1] but get optimized independently.
    num_models = 2*num_subjects# one model for each subject half-brain
    beta_start = torch.zeros( size=(num_models,), dtype=float_type, device=device )
    beta_end = torch.ones( size=(num_models,), dtype=float_type, device=device )
    best_beta = beta_end.clone()
    beta = torch.zeros( size=(num_parallel, num_models), dtype=float_type, device=device )
    for sim in range(num_updates):
        for model_index in range(num_models):
            beta[:,model_index] = torch.linspace(start=beta_start[model_index], end=beta_end[model_index], steps=num_parallel, dtype=float_type, device=device)
        # print('beta')
        # print(beta)
        # print( 'beta', beta.tolist() )
        # Add a singleton dimension at the end so that it knows to broadcast the same beta to every node in a model.
        model.beta = beta.flatten().unsqueeze(dim=1)
        # Run the Ising model, and find the sim FC.
        sim_fc = model.simulate_and_record_fc(num_steps=sim_length)
        # Compare the sim FC to the data FC.
        fc_rmse = isingmodel.get_pairwise_rmse(mat1=data_fc, mat2=sim_fc)
        print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}, time {time.time() - code_start_time:.3f}')
        fc_correlation = isingmodel.get_pairwise_correlation(mat1=data_fc, mat2=sim_fc)
        print(f'FC correlation min {fc_correlation.min():.3g}, mean {fc_correlation.mean():.3g}, max {fc_correlation.max():.3g}, time {time.time() - code_start_time:.3f}')
        # Narrow down the range in which we will search for the best beta.
        min_fc_rmse, min_fc_rmse_index = fc_rmse.unflatten( dim=0, sizes=(num_parallel, num_models) ).min(dim=0)
        # print('min_fc_rmse')
        # print(min_fc_rmse)
        for model_index in range(num_models):
            min_fc_rmse_index_for_subject = min_fc_rmse_index[model_index]
            best_beta_for_subject = beta[min_fc_rmse_index_for_subject, model_index]
            best_beta[model_index] = best_beta_for_subject
            # Narrow down the range in which we will search for the best beta.
            if min_fc_rmse_index_for_subject == 0:
                if best_beta_for_subject == 0.0:
                    # If the best value is 0,
                    # search the interval between 0 and the next highest value.
                    beta_start[model_index] = 0.0
                    beta_end[model_index] = beta[min_fc_rmse_index_for_subject+1,model_index]
                else:
                    # If the best value is at the lower end of the interval and is non-0,
                    # search the interval from 0 to the this value.
                    beta_start[model_index] = 0.0
                    beta_end[model_index] = best_beta_for_subject
            elif min_fc_rmse_index_for_subject == num_parallel-1:
                # If the best value is at the upper end of the interval,
                # search a new interval with the same width starting at this upper limit.
                beta_width = best_beta_for_subject - beta_start[model_index]
                beta_start[model_index] = best_beta_for_subject
                beta_end[model_index] = best_beta_for_subject + beta_width
            else:
                # If the best value is not at either end of the interval,
                # create a new interval centered on this value.
                beta_start[model_index] = beta[min_fc_rmse_index_for_subject-1,model_index]
                beta_end[model_index] = beta[min_fc_rmse_index_for_subject+1,model_index]
        # print('best_beta')
        # print(best_beta)
    # Save the model, simulation FC, FC RMSE, and FC correlation.
    file_suffix = f'halfbrain_{model_type}_{data_subset}_parallel_{num_parallel}_sims_{num_updates}_steps_{sim_length}.pt'
    ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    torch.save(obj=model, f=ising_model_file)
    print(f'saved {ising_model_file}, time {time.time() - code_start_time:.3f}')
    sim_fc_file = os.path.join(output_directory, f'sim_fc_{file_suffix}')
    torch.save(obj=sim_fc, f=sim_fc_file)
    print(f'saved {sim_fc_file}, time {time.time() - code_start_time:.3f}')
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{file_suffix}')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'saved {fc_rmse_file}, time {time.time() - code_start_time:.3f}')
    fc_correlation_file = os.path.join(output_directory, f'fc_correlation_{file_suffix}')
    torch.save(obj=fc_correlation, f=fc_correlation_file)
    print(f'saved {fc_correlation_file}, time {time.time() - code_start_time:.3f}')

print(f'done, time {time.time() - code_start_time:.3f}')