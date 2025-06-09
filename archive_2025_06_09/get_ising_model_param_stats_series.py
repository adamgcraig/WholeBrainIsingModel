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
    parser.add_argument("-a", "--data_directory", type=str, default='D:\\ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--model_file_fragment", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates', help="the part of the Ising model file between ising_model_ and _[the number of parameter update steps].pt.")
    parser.add_argument("-d", "--update_increment", type=int, default=1000, help="number of updates between models to test")
    parser.add_argument("-e", "--min_updates", type=int, default=0, help="first number of updates to test")
    parser.add_argument("-f", "--max_updates", type=int, default=63000, help="last number of updates to test")
    parser.add_argument("-g", "--num_targets", type=int, default=31, help="number of fitting targets (subjects or thresholds), used to preallocate memory")
    parser.add_argument("-i", "--num_nodes", type=int, default=360, help="number of nodes in each model, used to preallocate memory")
    parser.add_argument("-j", "--include_min", action='store_true', default=True, help="Set this flag to record the minimum values of parameters.")
    parser.add_argument("-k", "--include_median", action='store_true', default=True, help="Set this flag to record the median values of parameters.")
    parser.add_argument("-l", "--include_max", action='store_true', default=True, help="Set this flag to record the maximum values of parameters.")
    parser.add_argument("-m", "--include_mean", action='store_true', default=True, help="Set this flag to record the mean values of parameters.")
    parser.add_argument("-n", "--include_std", action='store_true', default=True, help="Set this flag to record the standard deviations of parameters.")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    model_file_fragment = args.model_file_fragment
    print(f'model_file_fragment={model_file_fragment}')
    update_increment = args.update_increment
    print(f'update_increment={update_increment}')
    min_updates = args.min_updates
    print(f'min_updates={min_updates}')
    max_updates = args.max_updates
    print(f'max_updates={max_updates}')
    num_targets = args.num_targets
    print(f'num_targets={num_targets}')
    num_nodes = args.num_nodes
    print(f'num_nodes={num_nodes}')
    include_min = args.include_min
    print(f'include_min={include_min}')
    include_median = args.include_median
    print(f'include_median={include_median}')
    include_max = args.include_max
    print(f'include_max={include_max}')
    include_mean = args.include_mean
    print(f'include_mean={include_mean}')
    include_std = args.include_std
    print(f'include_std={include_std}')

    epsilon = 0.0

    summary_lambdas = []
    summary_names = []
    if include_min:
        # summary_lambdas += [lambda x: torch.min(input=x, dim=0).values]
        summary_names += ['min']
    if include_median:
        # summary_lambdas += [lambda x: torch.median(input=x, dim=0).values]
        summary_names += ['median']
    if include_max:
        # summary_lambdas += [lambda x: torch.max(input=x, dim=0).values]
        summary_names += ['max']
    if include_mean:
        # summary_lambdas += [lambda x: torch.mean(input=x, dim=0)]
        summary_names += ['mean']
    if include_std:
        # summary_lambdas += [lambda x: torch.std(input=x, dim=0).values]
        summary_names += ['std']
    # summarize_params = lambda params: torch.cat( tensors=[ summary_lambda(params) for summary_lambda in summary_lambdas ], dim=-1 )
    summary_string = '_'.join(summary_names)
    num_summaries = len(summary_names)

    def summarize_params(params:torch.Tensor):
        summaries = []
        if include_min:
            summaries += [torch.min(input=params, dim=0).values]
        if include_median:
            summaries += [torch.median(input=params, dim=0).values]
        if include_max:
            summaries += [torch.max(input=params, dim=0).values]
        if include_mean:
            summaries += [torch.mean(input=params, dim=0)]
        if include_std:
            summaries += [torch.std(input=params, dim=0)]
        return torch.stack( tensors=summaries, dim=-1 )

    def summarize_model(model_file_fragment_with_updates:str):
        summary_file_fragment = f'{summary_string}_{model_file_fragment_with_updates}'
        model_file = os.path.join(data_directory, f'{model_file_fragment_with_updates}.pt')
        model = torch.load(f=model_file, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file}' )
        h_summary = summarize_params(params=model.h)
        print( 'h_summary size', h_summary.size() )
        h_summary_file = os.path.join(output_directory, f'h_{summary_file_fragment}.pt')
        torch.save(obj=h_summary, f=h_summary_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {h_summary_file}')
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=model.J.size(dim=-1), device=model.J.device )
        J_summary = summarize_params(params=model.J[:,:,triu_rows,triu_cols])
        print( 'J_summary size', J_summary.size() )
        J_summary_file = os.path.join(output_directory, f'J_{summary_file_fragment}.pt')
        torch.save(obj=J_summary, f=J_summary_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {J_summary_file}')
        return h_summary, J_summary
    
    def print_and_save(summary:torch.Tensor, stat_name:str):
        file_name = os.path.join(output_directory, f'{stat_name}_{model_file_fragment}_updates_min_{min_updates}_max_{max_updates}_increment_{update_increment}.pt')
        torch.save(obj=summary, f=file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {file_name}')
    
    update_counts = torch.arange(start=min_updates, end=max_updates+1, step=update_increment, dtype=int_type, device=device)
    num_model_files = update_counts.numel()
    h_summary = torch.zeros( size=(num_model_files, num_targets, num_nodes, num_summaries), dtype=float_type, device=device )
    num_pairs = ( num_nodes*(num_nodes-1) )//2
    J_summary = torch.zeros( size=(num_model_files, num_targets, num_pairs, num_summaries), dtype=float_type, device=device )
    for model_file_index in range(num_model_files):
        num_updates = update_counts[model_file_index]
        print(f'model file {model_file_index+1} of {num_model_files}, update count {num_updates}')
        model_file_fragment_with_updates = f'{model_file_fragment}_{num_updates}'
        h_summary[model_file_index,:,:,:], J_summary[model_file_index,:,:,:] = summarize_model(model_file_fragment_with_updates=model_file_fragment_with_updates)
    print_and_save(summary=h_summary, stat_name=f'h_{summary_string}')
    print_and_save(summary=J_summary, stat_name=f'J_{summary_string}')
print(f'time {time.time()-code_start_time:.3f}, done')