import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
    parser.add_argument("-a", "--data_directory", type=str, default='/data/agcraig', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='/data/agcraig', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--threshold", type=float, default=0, help="number of standard deviations above the mean at which to binarize each region time series, can be 0 or negative")
    parser.add_argument("-d", "--training_subject_start", type=int, default=0, help="index of first training subject")
    parser.add_argument("-e", "--training_subject_end", type=int, default=669, help="one after index of last training subject")
    parser.add_argument("-f", "--models_per_subject", type=int, default=10, help="number of separate models of each subject")
    parser.add_argument("-g", "--num_pseudolikelihood_steps", type=int, default=10, help="number of steps for which to run pseudolikelihood maximization on the group data")
    parser.add_argument("-i", "--learning_rate", type=float, default=0.01, help="amount by which to multiply updates to the model parameters during the Euler step")
    parser.add_argument("-j", "--block_length", type=int, default=6690, help="amount by which to multiply updates to the model parameters during the Euler step")
    parser.add_argument("-k", "--num_beta_opt_steps", type=int, default=10, help="number of steps for which to search for optimal beta on the group data")
    parser.add_argument("-l", "--sim_length", type=int, default=1200, help="number of steps in a simulation")
    parser.add_argument("-m", "--min_beta", type=float, default=10e-10, help="low end of initial beta search interval")
    parser.add_argument("-n", "--max_beta", type=float, default=1.0, help="high end of initial beta search interval")
    parser.add_argument("-o", "--num_sim_updates", type=int, default=10, help="number of updates in the simulate-and-update loop")
    parser.add_argument("-p", "--init_to_means", action='store_true', default=False, help="Set this flag in order to initialize h with the data mean state and J with the data mean state product instead of with randoms.")
    parser.add_argument("-q", "--verbose", action='store_true', default=False, help="Set this flag in order to print status updates during the various optimization steps.")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    threshold = args.threshold
    print(f'threshold={threshold}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')
    models_per_subject = args.models_per_subject
    print(f'models_per_subject={models_per_subject}')
    num_pseudolikelihood_steps = args.num_pseudolikelihood_steps
    print(f'num_pseudolikelihood_steps={num_pseudolikelihood_steps}')
    learning_rate = args.learning_rate
    print(f'learning_rate={learning_rate}')
    block_length = args.block_length
    print(f'block_length={block_length}')
    num_beta_opt_steps = args.num_beta_opt_steps
    print(f'num_beta_opt_steps={num_beta_opt_steps}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
    min_beta = args.min_beta
    print(f'min_beta={min_beta}')
    max_beta = args.max_beta
    print(f'min_beta={max_beta}')
    num_sim_updates = args.num_sim_updates
    print(f'num_sim_updates={num_sim_updates}')
    init_to_means = args.init_to_means
    print(f'init_to_means={init_to_means}')
    verbose = args.verbose
    print(f'verbose={verbose}')
    
    def get_group_time_series(data_directory:str, threshold:float, training_subject_start:int, training_subject_end:int):
        print('loading data time series')
        data_ts_file = os.path.join(data_directory, f'data_ts_all_as_is.pt')
        data_ts = torch.load(data_ts_file)
        print( f'time {time.time()-code_start_time:.3f}, loaded data_ts with size', data_ts.size() )
        data_ts_binary = isingmodellight.binarize_data_ts_z(data_ts=data_ts, threshold=0)
        print(f'time {time.time()-code_start_time:.3f}, binarized data_ts at threshold mean + {threshold:.3g} std.dev.')
        group_data_ts = torch.permute( input=data_ts_binary[:,training_subject_start:training_subject_end,:,:], dims=(2, 1, 0, 3) ).flatten(start_dim=1, end_dim=-1).unsqueeze(dim=0)
        print( f'time {time.time()-code_start_time:.3f}, selected training subject, reshaped time series to size', group_data_ts.size() )
        num_subjects = data_ts.size(dim=1)# Return the original number of subjects.
        return group_data_ts.clone(), num_subjects
    
    def get_group_means(data_directory:str, threshold:float, training_subject_start:int, training_subject_end:int):
        print('loading data time series')
        data_ts_file = os.path.join(data_directory, f'data_ts_all_as_is.pt')
        data_ts = torch.load(data_ts_file)
        num_subjects = data_ts.size(dim=1)# Return the original number of subjects.
        print( f'time {time.time()-code_start_time:.3f}, loaded data_ts with size', data_ts.size() )
        data_ts = isingmodellight.binarize_data_ts_z(data_ts=data_ts[:,training_subject_start:training_subject_end,:,:], threshold=threshold)
        print(f'time {time.time()-code_start_time:.3f}, binarized data_ts at threshold mean + {threshold:.3g} std.dev.')
        data_ts = torch.permute( input=data_ts, dims=(2, 1, 0, 3) ).flatten(start_dim=1, end_dim=-1).unsqueeze(dim=0)
        print( f'time {time.time()-code_start_time:.3f}, selected training subject, reshaped time series to size', data_ts.size() )
        group_mean_state, group_mean_state_product = isingmodellight.get_time_series_mean(time_series=data_ts)
        print( f'time {time.time()-code_start_time:.3f}, computed group mean state and state product with sizes', group_mean_state.size(), group_mean_state_product.size() )
        return group_mean_state, group_mean_state_product, num_subjects
    
    def create_group_model(data_directory:str, output_directory:str, threshold:float, training_subject_start:int, training_subject_end:int, models_per_subject:int, min_beta:float, max_beta:float, learning_rate:float, block_length:int, num_pseudolikelihood_steps:int, num_beta_opt_steps:int, num_sim_updates:int, sim_length:int, init_to_means:bool, verbose:bool):
        if num_pseudolikelihood_steps > 0:
            group_data_ts, num_subjects = get_group_time_series(data_directory=data_directory, threshold=threshold, training_subject_start=training_subject_start, training_subject_end=training_subject_end)
            group_mean_state, group_mean_state_product = isingmodellight.get_time_series_mean(time_series=group_data_ts)
        else:
            group_mean_state, group_mean_state_product, num_subjects = get_group_means(data_directory=data_directory, threshold=threshold, training_subject_start=training_subject_start, training_subject_end=training_subject_end)
        print( f'time {time.time()-code_start_time:.3f}, computed group mean state of size', group_mean_state.size(), 'and group mean state product of size', group_mean_state_product.size() )
        group_mean_state_file = os.path.join(output_directory, f'group_mean_state_mean_std_{threshold:.3g}.pt')
        torch.save(obj=group_mean_state, f=group_mean_state_file)
        print( f'time {time.time()-code_start_time:.3f}, saved {group_mean_state_file}' )
        group_mean_state_product_file = os.path.join(output_directory, f'group_mean_state_product_mean_std_{threshold:.3g}.pt')
        torch.save(obj=group_mean_state_product, f=group_mean_state_product_file)
        print( f'time {time.time()-code_start_time:.3f}, saved {group_mean_state_product_file}' )
        total_num_models = models_per_subject * num_subjects
        num_nodes = group_mean_state.size(dim=-1)
        group_beta = isingmodellight.get_linspace_beta(models_per_subject=total_num_models, num_subjects=1, dtype=float_type, device=device, min_beta=min_beta, max_beta=max_beta)
        if init_to_means:
            group_h = isingmodellight.get_h_from_means(models_per_subject=total_num_models, mean_state=group_mean_state)
            group_J = isingmodellight.get_J_from_means(models_per_subject=total_num_models, mean_state_product=group_mean_state_product)
        else:
            group_J = isingmodellight.get_random_J(models_per_subject=total_num_models, num_subjects=1, num_nodes=num_nodes, dtype=float_type, device=device)
            group_h = isingmodellight.get_random_h(models_per_subject=total_num_models, num_subjects=1, num_nodes=num_nodes, dtype=float_type, device=device)
        group_s = isingmodellight.get_neg_state_like(input=group_h)
        group_model = IsingModelLight(beta=group_beta, J=group_J, h=group_h, s=group_s)
        print( f'time {time.time()-code_start_time:.3f}, initialized {total_num_models} random models')
        if num_pseudolikelihood_steps > 0:
            group_model.fit_by_pseudolikelihood_model_by_model(num_updates=num_pseudolikelihood_steps, target_ts=group_data_ts, target_state_means=group_mean_state, target_state_product_means=group_mean_state_product, learning_rate=learning_rate, verbose=verbose)
            group_model_file = os.path.join(output_directory, f'ising_model_group_mean_std_{threshold:.3g}_models_{total_num_models}_lr_{learning_rate:.3g}_sim_steps_{sim_length:.3g}_pseudolikelihood_{num_pseudolikelihood_steps}.pt')
            torch.save(obj=group_model, f=group_model_file)
            print(f'time {time.time()-code_start_time:.3f}, saved {group_model_file}')
        # group_model.optimize_beta_msp(target_mean_state_product=group_mean_state_product, num_updates=num_beta_opt_steps, num_steps=sim_length, min_beta=min_beta, max_beta=max_beta, verbose=verbose)
        group_cov = isingmodellight.get_cov(state_mean=group_mean_state, state_product_mean=group_mean_state_product)
        group_model.optimize_beta_pmb(target_cov=group_cov, num_updates=num_beta_opt_steps, num_steps=sim_length, min_beta=min_beta, max_beta=max_beta, verbose=verbose)
        group_model_file = os.path.join(output_directory, f'ising_model_group_mean_std_{threshold:.3g}_models_{total_num_models}_lr_{learning_rate:.3g}_sim_steps_{sim_length:.3g}_pseudolikelihood_{num_pseudolikelihood_steps}_minb_{min_beta:.3g}_maxb_{max_beta:.3g}_betaopt_{num_beta_opt_steps}.pt')
        torch.save(obj=group_model, f=group_model_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {group_model_file}')
        group_model.fit_by_simulation_pmb(target_state_mean=group_mean_state, target_state_product_mean=group_mean_state_product, num_updates=num_sim_updates, steps_per_update=sim_length, learning_rate=learning_rate, verbose=True)
        group_model_file = os.path.join(output_directory, f'ising_model_group_mean_std_{threshold:.3g}_models_{total_num_models}_lr_{learning_rate:.3g}_sim_steps_{sim_length:.3g}_plupdates_{num_pseudolikelihood_steps}_minb_{min_beta:.3g}_maxb_{max_beta:.3g}_betaopt_{num_beta_opt_steps}_simupdates_{num_sim_updates}.pt')
        torch.save(obj=group_model, f=group_model_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {group_model_file}')
        return group_model
    
    def get_individual_data_mean(data_directory:str, threshold:float):
        print('loading data time series')
        data_ts_file = os.path.join(data_directory, f'data_ts_all_as_is.pt')
        data_ts = torch.load(data_ts_file)
        print( f'time {time.time()-code_start_time:.3f}, loaded data_ts with size', data_ts.size() )
        data_ts = isingmodellight.binarize_data_ts_z(data_ts=data_ts, threshold=threshold)
        print(f'time {time.time()-code_start_time:.3f}, binarized data_ts at threshold mean + {threshold:.3g} std.dev.')
        individual_mean_state, individual_mean_state_product = isingmodellight.get_time_series_mean(time_series=data_ts)
        print( f'time {time.time()-code_start_time:.3f}, computed group mean state and state product with sizes', individual_mean_state.size(), individual_mean_state_product.size() )
        individual_mean_state = individual_mean_state.mean(dim=0, keepdim=False)
        individual_mean_state_product = individual_mean_state_product.mean(dim=0, keepdim=False)
        print( f'time {time.time()-code_start_time:.3f}, combined means for individual scans so that sizes are now', individual_mean_state.size(), individual_mean_state_product.size() )
        return individual_mean_state, individual_mean_state_product
    
    individual_model = create_group_model(data_directory=data_directory, output_directory=output_directory, threshold=threshold, training_subject_start=training_subject_start, training_subject_end=training_subject_end, models_per_subject=models_per_subject, min_beta=min_beta, max_beta=max_beta, learning_rate=learning_rate, block_length=block_length, num_pseudolikelihood_steps=num_pseudolikelihood_steps, num_beta_opt_steps=num_beta_opt_steps, num_sim_updates=num_sim_updates, sim_length=sim_length, init_to_means=init_to_means, verbose=verbose)
    individual_mean_state, individual_mean_state_product = get_individual_data_mean(data_directory=data_directory, threshold=threshold)
    individual_mean_state_file = os.path.join(output_directory, f'individual_mean_state_mean_std_{threshold:.3g}.pt')
    torch.save(obj=individual_mean_state, f=individual_mean_state_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {individual_mean_state_file}' )
    individual_mean_state_product_file = os.path.join(output_directory, f'individual_mean_state_product_mean_std_{threshold:.3g}.pt')
    torch.save(obj=individual_mean_state_product, f=individual_mean_state_product_file)
    print( f'time {time.time()-code_start_time:.3f}, saved {individual_mean_state_product_file}' )
    total_num_models = individual_model.s.size(dim=0)
    individual_model.s = individual_model.s.squeeze(dim=1).unflatten( dim=0, sizes=(models_per_subject, -1) )
    individual_model.beta = individual_model.beta.squeeze(dim=1).unflatten( dim=0, sizes=(models_per_subject, -1) )
    individual_model.h = individual_model.h.squeeze(dim=1).unflatten( dim=0, sizes=(models_per_subject, -1) )
    individual_model.J = individual_model.J.squeeze(dim=1).unflatten( dim=0, sizes=(models_per_subject, -1) )
    individual_model.fit_by_simulation_pmb(target_state_mean=individual_mean_state, target_state_product_mean=individual_mean_state_product, num_updates=num_sim_updates, steps_per_update=sim_length, learning_rate=learning_rate, verbose=verbose)
    individual_model_file = os.path.join(output_directory, f'ising_model_individual_mean_std_{threshold:.3g}_models_{total_num_models}_lr_{learning_rate:.3g}_sim_steps_{sim_length:.3g}_plupdates_{num_pseudolikelihood_steps}_minb_{min_beta:.3g}_maxb_{max_beta:.3g}_betaopt_{num_beta_opt_steps}_simupdates_{num_sim_updates}.pt')
    torch.save(obj=individual_model, f=individual_model_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {individual_model_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')