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
parser.add_argument("-s", "--sim_length", type=int, default=1200, help="number of simulation steps between Euler updates")
parser.add_argument("-u", "--num_updates", type=int, default=5, help="number of Euler updates to do")
parser.add_argument("-p", "--num_parallel", type=int, default=3, help="number of parallel simulations to run")
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
    individual_state_mean_file = os.path.join(output_directory, f'mean_state_individual_{data_subset}.pt')
    individual_state_mean = torch.load(individual_state_mean_file)
    print(f'loaded {individual_state_mean_file}, time {time.time() - code_start_time:.3f}')
    individual_product_mean_file = os.path.join(output_directory, f'mean_state_product_individual_{data_subset}.pt')
    individual_product_mean = torch.load(individual_product_mean_file)
    print(f'loaded {individual_product_mean_file}, time {time.time() - code_start_time:.3f}')
    # Compute the FC of the data.
    individual_fc_file = os.path.join(output_directory, f'fc_individual_{data_subset}.pt')
    individual_fc = torch.load(individual_fc_file)
    print(f'loaded {individual_fc_file}, time {time.time() - code_start_time:.3f}')

    # Initialize the Ising model using the means.
    # Make a duplicate of each subject for each beta we want to test.
    # Then flatten together the two batch dimensions, subject and beta.
    num_subjects, num_nodes = individual_state_mean.size()
    total_num_models = num_subjects * num_parallel
    individual_ising_model = IsingModel(batch_size=total_num_models, num_nodes=num_nodes, dtype=float_type, device=device)
    individual_state_mean_rep = individual_state_mean.unsqueeze(dim=1).repeat( (1,num_parallel,1) )
    individual_product_mean_rep = individual_product_mean.unsqueeze(dim=1).repeat( (1,num_parallel,1) )
    individual_ising_model.init_with_target_means( target_mean=individual_state_mean_rep.flatten(start_dim=0, end_dim=1), target_product_mean=individual_product_mean_rep.flatten(start_dim=0, end_dim=1) )

    # Repeat each individual data FC for all the beta values, and flatten the batch dimensions together.
    individual_fc = individual_fc.unsqueeze(dim=1).repeat( (1,num_parallel,1) ).flatten(start_dim=0, end_dim=1)
    # Create a different set of beta values for each subject.
    # They start out with [0, 1] but get optimized independently.
    beta_start = torch.zeros( size=(num_subjects,), dtype=float_type, device=device )
    beta_end = torch.ones( size=(num_subjects,), dtype=float_type, device=device )
    best_beta = beta_end.clone()
    beta = torch.zeros( size=(num_subjects, num_parallel), dtype=float_type, device=device )
    for sim in range(num_updates):
        for subject in range(num_subjects):
            beta[subject,:] = torch.linspace(start=beta_start[subject], end=beta_end[subject], steps=num_parallel, dtype=float_type, device=device)
        # print( 'beta', beta.tolist() )
        # Flatten the batch dimensions into one.
        # Add a singleton dimension at the end so that it knows to broadcast the same beta to every node in a model.
        individual_ising_model.beta = beta.flatten(start_dim=0, end_dim=1).unsqueeze(dim=-1)
        # Run the Ising model, and find the sim FC.
        sim_individual_fc = individual_ising_model.simulate_and_record_fc(num_steps=sim_length)
        # Compare the sim FC to the data FC.
        # Unflatten the batch dimensions of the RMSE so that we can get the best beta for each individual.
        fc_rmse = isingmodel.get_pairwise_rmse(mat1=individual_fc, mat2=sim_individual_fc)
        min_fc_rmse, min_fc_rmse_index = fc_rmse.unflatten( dim=0, sizes=(num_subjects, num_parallel) ).min(dim=-1)
        for subject in range(num_subjects):
            min_fc_rmse_index_for_subject = min_fc_rmse_index[subject]
            best_beta_for_subject = beta[subject, min_fc_rmse_index_for_subject]
            best_beta[subject] = best_beta_for_subject
            # Narrow down the range in which we will search for the best beta.
            if min_fc_rmse_index_for_subject == 0:
                if best_beta_for_subject == 0:
                    # If the best value is 0,
                    # search the interval between 0 and the next highest value.
                    beta_start[subject] = 0.0
                    beta_end[subject] = beta[subject,min_fc_rmse_index_for_subject+1]
                else:
                    # If the best value is at the lower end of the interval and is non-0,
                    # search the interval from 0 to the this value.
                    beta_start[subject] = 0.0
                    beta_end[subject] = best_beta_for_subject
            elif min_fc_rmse_index_for_subject == num_parallel-1:
                # If the best value is at the upper end of the interval,
                # search a new interval with the same width starting at this upper limit.
                beta_width = best_beta_for_subject - beta_start[subject]
                beta_start[subject] = best_beta_for_subject
                beta_end[subject] = best_beta_for_subject + beta_width
            else:
                # If the best value is not at either end of the interval,
                # create a new interval centered on this value.
                beta_start[subject] = beta[subject,min_fc_rmse_index_for_subject-1]
                beta_end[subject] = beta[subject,min_fc_rmse_index_for_subject+1]
        # print( 'RMSE', fc_rmse.tolist() )
        print(f'best beta min {best_beta.min():.3g}, beta mean {best_beta.mean():.3g}, beta max {best_beta.max():.3g}, time {time.time() - code_start_time:.3f}')
        print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}, time {time.time() - code_start_time:.3f}')
        fc_correlation = isingmodel.get_pairwise_correlation(mat1=individual_fc, mat2=sim_individual_fc)
        # print( 'correlation', fc_correlation.tolist() )
        print(f'FC correlation min {fc_correlation.min():.3g}, mean {fc_correlation.mean():.3g}, max {fc_correlation.max():.3g}, time {time.time() - code_start_time:.3f}')
    
    file_suffix = f'individual_{data_subset}_parallel_{num_parallel}_sims_{num_updates}_steps_{sim_length}.pt'
    # Set beta to the single best value from the last iteration, and save the model.
    # Repeat the best value for a given subject for each instance of that subject.
    individual_ising_model.beta = best_beta.unsqueeze(dim=0).repeat( (1,num_parallel) ).flatten(start_dim=0, end_dim=1).unsqueeze(dim=-1)
    individual_ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    torch.save(obj=individual_ising_model, f=individual_ising_model_file)
    # Run all the copies of the model again with the best beta to test reproducibility.
    print( f'testing with beta min {individual_ising_model.beta.min():.3g}, mean {individual_ising_model.beta.mean():.3g}, max {individual_ising_model.beta.max():.3g}' )
    # Run the Ising model, and find the sim FC.
    sim_individual_fc = individual_ising_model.simulate_and_record_fc(num_steps=sim_length)
    # Compare the sim FC to the data FC.
    fc_rmse = isingmodel.get_pairwise_rmse(mat1=individual_fc, mat2=sim_individual_fc).unflatten( dim=0, sizes=(num_subjects, num_parallel) )
    # print( 'RMSE', fc_rmse.tolist() )
    print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}, time {time.time() - code_start_time:.3f}')
    fc_correlation = isingmodel.get_pairwise_correlation(mat1=individual_fc, mat2=sim_individual_fc).unflatten( dim=0, sizes=(num_subjects, num_parallel) )
    # print( 'correlation', fc_correlation.tolist() )
    print(f'FC correlation min {fc_correlation.min():.3g}, mean {fc_correlation.mean():.3g}, max {fc_correlation.max():.3g}, time {time.time() - code_start_time:.3f}')
    print(f'saved {individual_ising_model_file}, time {time.time() - code_start_time:.3f}')
    sim_individual_fc_file = os.path.join(output_directory, f'sim_fc_{file_suffix}')
    torch.save(obj=sim_individual_fc, f=sim_individual_fc_file)
    print(f'saved {sim_individual_fc_file}, time {time.time() - code_start_time:.3f}')
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{file_suffix}')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'saved {fc_rmse_file}, time {time.time() - code_start_time:.3f}')
    fc_correlation_file = os.path.join(output_directory, f'fc_correlation_{file_suffix}')
    torch.save(obj=fc_correlation, f=fc_correlation_file)
    print(f'saved {fc_correlation_file}, time {time.time() - code_start_time:.3f}')

print(f'done, time {time.time() - code_start_time:.3f}')