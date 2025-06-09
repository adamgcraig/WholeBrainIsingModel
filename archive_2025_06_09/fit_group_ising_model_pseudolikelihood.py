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
    parser.add_argument("-c", "--data_file_fragment", type=str, default='mean_std_0', help="data state and count pickle (.pt) file excluding path and data_state_ and group_data_count_ and file extension")
    parser.add_argument("-d", "--num_inits", type=int, default=100, help="number of models to fit from independent, randomly chosen starting configurations")
    parser.add_argument("-e", "--rand_init_min", type=float, default=-100, help="minimum value to use in random initial configurations")
    parser.add_argument("-f", "--rand_init_max", type=float, default=100, help="maximum value to use in random initial configurations")
    parser.add_argument("-g", "--num_pl_updates", type=int, default=1200, help="number of steps of pseudolikelihood maximization to perform")
    parser.add_argument("-i", "--pl_learning_rate", type=float, default=0.01, help="learning rate of pseudolikelihood maximization")
    parser.add_argument("-j", "--print_every_steps", type=int, default=1, help="number of steps between fitting status printouts")
    parser.add_argument("-k", "--num_betas", type=int, default=100, help="number of betas to test in parallel per round of beta optimization")
    parser.add_argument("-l", "--min_beta", type=float, default=1.0e-10, help="minimum beta to try during beta optimization")
    parser.add_argument("-m", "--max_beta", type=float, default=1.0, help="maximum beta to try during beta optimization")
    parser.add_argument("-n", "--num_beta_updates", type=int, default=1000, help="maximum steps of beta optimization")
    parser.add_argument("-o", "--beta_sim_length", type=int, default=1200, help="length of simulations to use in beta optimization")
    parser.add_argument("-p", "--num_param_updates", type=int, default=100, help="number of parameter updates in simulate-and-update loop")
    parser.add_argument("-q", "--param_sim_length", type=int, default=1200, help="length of simulations to use in simulate-and-update loop")
    parser.add_argument("-r", "--param_learning_rate", type=float, default=0.01, help="learning rate of simulate-and-update loop")
    parser.add_argument("-s", "--num_saves", type=int, default=10, help="number of times to run the simulate-and-update loop and save the model")

    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_fragment = args.data_file_fragment
    print(f'data_file_fragment={data_file_fragment}')
    num_inits = args.num_inits
    print(f'num_models={num_inits}')
    rand_init_min = args.rand_init_min
    print(f'rand_init_min={rand_init_min}')
    rand_init_max = args.rand_init_max
    print(f'rand_init_max={rand_init_max}')
    num_pl_updates = args.num_pl_updates
    print(f'num_pl_updates={num_pl_updates}')
    pl_learning_rate = args.pl_learning_rate
    print(f'pl_learning_rate={pl_learning_rate}')
    print_every_steps = args.print_every_steps
    print(f'print_every_steps={print_every_steps}')
    num_betas = args.num_betas
    print(f'num_betas={num_betas}')
    min_beta = args.min_beta
    print(f'min_beta={min_beta}')
    max_beta = args.max_beta
    print(f'max_beta={max_beta}')
    num_beta_updates = args.num_beta_updates
    print(f'num_beta_updates={num_beta_updates}')
    beta_sim_length = args.beta_sim_length
    print(f'beta_sim_length={beta_sim_length}')
    num_param_updates = args.num_param_updates
    print(f'num_param_updates={num_param_updates}')
    param_sim_length = args.param_sim_length
    print(f'param_sim_length={param_sim_length}')
    param_learning_rate = args.param_learning_rate
    print(f'param_learning_rate={param_learning_rate}')
    num_saves = args.num_saves
    print(f'num_saves={num_saves}')

    def load_data_states_as_float(data_directory:str, data_file_fragment:str):
        data_states_file = os.path.join(data_directory, f'data_states_{data_file_fragment}.pt')
        data_states_bool = torch.load(data_states_file)
        print( f'{time.time()-code_start_time:.3f}, loaded data_states_bool size ', data_states_bool.size() )
        data_states = 2*data_states_bool.float() - 1
        print( f'{time.time()-code_start_time:.3f}, converted to float data_states size ', data_states.size() )
        data_counts_file = os.path.join(data_directory, f'group_data_counts_{data_file_fragment}.pt')
        data_counts = torch.load(data_counts_file)
        print( f'{time.time()-code_start_time:.3f}, loaded data_counts size ', data_counts.size() )
        return data_states, data_counts/torch.sum(data_counts)
    
    def get_mean_state(data_states:torch.Tensor, data_probs:torch.Tensor):
        # (1, num_states)@(num_states, num_nodes) = (1, num_nodes)
        return torch.matmul( data_probs.unsqueeze(dim=0), data_states )
    
    def get_mean_state_product(data_states:torch.Tensor, data_probs:torch.Tensor):
        # (num_states, num_nodes) * (num_states, 1) = (num_states, num_nodes)
        # (num_nodes, num_states) @ (num_states, num_nodes) = (num_nodes, num_nodes)
        return torch.matmul( data_states.transpose(dim0=0, dim1=1), data_states * data_probs.unsqueeze(dim=-1) )
    
    def get_cov(mean_state:torch.Tensor, mean_state_product:torch.Tensor):
        # (1, num_nodes, num_nodes) - (1, num_nodes, 1) * (1, 1, num_nodes) = (1, num_nodes, num_nodes)
        return mean_state_product - mean_state.unsqueeze(dim=-1) * mean_state.unsqueeze(dim=-2)

    def maximize_pseudolikelihood(data_states:torch.Tensor, data_probs:torch.Tensor, num_models:int, rand_init_min:float, rand_init_max:float, num_steps:int, learning_rate:float, print_every_steps:int=1):
        num_states, num_nodes = data_states.size()
        # size = (1, num_states)@(num_states, num_nodes) = (1, num_nodes) -unsqueeze(dim=0)-> (1, 1, num_nodes)
        mean_state = get_mean_state(data_states=data_states, data_probs=data_probs).unsqueeze(dim=0)
        # (num_states, num_nodes) * (num_states, 1) = (num_states, num_nodes)
        # (num_nodes, num_states) @ (num_states, num_nodes) = (num_nodes, num_nodes)  -unsqueeze(dim=0)-> (1, num_nodes, num_nodes)
        mean_state_product = get_mean_state_product(data_states=data_states, data_probs=data_probs).unsqueeze(dim=0)
        data_probs_for_mean_field = data_probs.unsqueeze(dim=0).unsqueeze(dim=-1)# (1, num_states, 1)
        data_states_for_h = data_states.unsqueeze(dim=0)# (1, num_states, num_nodes)
        data_states_for_J = data_states.transpose(dim0=0, dim1=1).unsqueeze(dim=0)# (1, num_nodes, num_states)
        rand_init_range = rand_init_max - rand_init_min
        h = rand_init_min + rand_init_range * torch.rand( size=(num_models, 1, num_nodes), dtype=float_type, device=data_states.device )
        J = rand_init_min + rand_init_range * torch.rand( size=(num_models, num_nodes, num_nodes), dtype=float_type, device=data_states.device )
        J -= torch.diag_embed( input=torch.diagonal(input=J, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
        # (1, num_states, num_nodes) @ (num_models, num_nodes, num_nodes) -> (num_models, num_states, num_nodes)
        mean_field = torch.zeros( size=(num_models, num_states, num_nodes), dtype=float_type, device=data_states.device )
        mean_mean_field_product = torch.zeros_like(J)# (num_models, num_nodes, num_nodes)
        for step in range(num_steps):
            # (1, num_states, num_nodes) @ (num_models, num_nodes, num_nodes) -> (num_models, num_states, num_nodes)
            torch.matmul(data_states_for_h, J, out=mean_field)
            # (num_models, num_states, num_nodes) + (num_models, 1, num_nodes) -> (num_models, num_states, num_nodes)
            mean_field += h
            mean_field.tanh_()
            # (num_models, num_states, num_nodes) * (1, num_states, 1) = (num_models, num_states, num_nodes)
            mean_field *= data_probs_for_mean_field
            # (num_models, 1, num_nodes) + (1, 1, num_nodes) -> (num_models, 1, num_nodes)
            # h += mean_state
            # sum( (num_models, num_states, num_nodes), dim=1, keepdim=True ) -> (num_models, 1, num_nodes)
            # mean_mean_field = torch.sum(mean_field, dim=1, keepdim=True)
            # (num_models, 1, num_nodes) - (num_models, 1, num_nodes) -> (num_models, 1, num_nodes)
            # h -= mean_mean_field
            h_diff = mean_state - torch.sum(mean_field, dim=1, keepdim=True)
            h += learning_rate*h_diff
            # (num_models, num_nodes, num_nodes) + (1, num_nodes, num_nodes) -> (num_models, num_nodes, num_nodes)
            # J += mean_state_product
            # -> (1, num_nodes, num_states) @ (num_models, num_states, num_nodes) -> (num_models, num_nodes, num_nodes)
            torch.matmul( data_states_for_J, mean_field, out=mean_mean_field_product )
            # (num_models, num_nodes, num_nodes) - (num_models, num_nodes, num_nodes) = (num_models, num_nodes, num_nodes)
            # J -= mean_mean_field_product
            J_diff = mean_state_product - mean_mean_field_product
            J_diff -= torch.diag_embed( input=torch.diagonal(input=J_diff, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
            J += learning_rate*(mean_state_product - mean_mean_field_product)
            if ( (step+1) % print_every_steps ) == 0:
                print(f'time {time.time()-code_start_time:.3f}, step {step+1}, h_diff min {h_diff.min():.3g}, mean {h_diff.mean():.3g}, max {h_diff.max():.3g}, J_diff min {J_diff.min():.3g}, J_diff mean {J_diff.mean():.3g}, J_diff max {J_diff.max():.3g}')
        # (num_models, 1, num_nodes) -squeeze(dim=1)-> (num_models, num_nodes)
        return h.squeeze(dim=1), J

    data_states, data_probs = load_data_states_as_float(data_directory=data_directory, data_file_fragment=data_file_fragment)
    h, J = maximize_pseudolikelihood(data_states=data_states, data_probs=data_probs, num_models=num_inits, rand_init_min=rand_init_min, rand_init_max=rand_init_max, num_steps=num_pl_updates, learning_rate=pl_learning_rate, print_every_steps=print_every_steps)
    pl_params_string = f'pseudolikelihood_{data_file_fragment}_models_{num_inits}_rand_min_{rand_init_min:.3g}_max_{rand_init_max:.3g}_lr_{pl_learning_rate:.3g}_steps_{num_pl_updates}'
    h_file = os.path.join(output_directory, f'h_{pl_params_string}.pt')
    torch.save(obj=h, f=h_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {h_file}')
    J_file = os.path.join(output_directory, f'J_{pl_params_string}.pt')
    torch.save(obj=J, f=J_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {J_file}')
    beta = isingmodellight.get_linspace_beta(models_per_subject=num_betas, num_subjects=num_inits, dtype=h.dtype, device=h.device, min_beta=min_beta, max_beta=max_beta)
    most_common_state_index = torch.argmax(data_probs)
    s = torch.unsqueeze(data_states[most_common_state_index,:], dim=0).unsqueeze(dim=0).repeat( repeats=(num_betas,num_inits,1) )
    print(f'initializing models to most common state, index {most_common_state_index}, num active {torch.count_nonzero(s > 0)}')
    model = IsingModelLight(  beta=beta, J=J.unsqueeze(dim=0).repeat( (num_betas,1,1,1) ), h=h.unsqueeze(dim=0).repeat( (num_betas,1,1) ), s=s  )
    
    # Compute our fitting target mean state and mean state product.
    # When saving,
    # unsqueeze to get dimensions (scans, subjects, nodes) and (scans, subjects, nodes, nodes),
    # because that is what test_ising_model_light_pmb.py is expecting.
    
    mean_state = get_mean_state(data_states=data_states, data_probs=data_probs)
    mean_state_file = os.path.join(output_directory, f'mean_state_{data_file_fragment}.pt')
    torch.save( obj=mean_state.unsqueeze(dim=0), f=mean_state_file )# (1, num_nodes) -> (1, 1, num_nodes)
    print(f'time {time.time()-code_start_time:.3f}, saved {mean_state_file}')

    # Add a subjects batch dimension to the version we use in subsequent steps,
    # since that is what the methods from isingmodellight.py are expecting.
    mean_state_product = get_mean_state_product(data_states=data_states, data_probs=data_probs).unsqueeze(dim=0)# (num_nodes, num_nodes) -> (1, num_nodes, num_nodes)
    mean_state_product_file = os.path.join(output_directory, f'mean_state_product_{data_file_fragment}.pt')
    torch.save( obj=mean_state_product.unsqueeze(dim=0), f=mean_state_product_file )# (1, num_nodes, num_nodes) -> (1, 1, num_nodes, num_nodes)
    print(f'time {time.time()-code_start_time:.3f}, saved {mean_state_product_file}')
    
    # Optimize beta to match target covariance.
    target_covariance = get_cov(mean_state=mean_state, mean_state_product=mean_state_product)
    model.optimize_beta_pmb( target_cov=target_covariance, num_updates=num_beta_updates, num_steps=beta_sim_length, min_beta=min_beta, max_beta=max_beta, verbose=True )
    model_beta_string = f'{pl_params_string}_num_beta_{num_betas}_min_{min_beta:.3g}_max_{max_beta:.3g}_updates_{num_beta_updates}_sim_{beta_sim_length}'
    model_file = os.path.join(output_directory, f'ising_model_{model_beta_string}.pt')
    torch.save(obj=model, f=model_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')

    # Simulate and update parameters to match target mean state and mean state product.
    model_params_string = f'{model_beta_string}_param_lr_{param_learning_rate:.3g}_sim_{param_sim_length}'
    for save_index in range(num_saves):
        model.fit_by_simulation_pmb( target_state_mean=mean_state, target_state_product_mean=mean_state_product, num_updates=num_param_updates, steps_per_update=param_sim_length, learning_rate=param_learning_rate, verbose=True )
        total_updates = (save_index+1)*num_param_updates
        model_file = os.path.join(output_directory, f'ising_model_{model_params_string}_updates_{total_updates}.pt')
        torch.save(obj=model, f=model_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')

print(f'time {time.time()-code_start_time:.3f}, done')