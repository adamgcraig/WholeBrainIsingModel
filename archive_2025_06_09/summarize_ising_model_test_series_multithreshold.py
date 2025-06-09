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

    parser = argparse.ArgumentParser(description="Summarize the results of Ising model multithreshold group model tests.")

    parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--zero_model_file_fragment", type=str, default='group_init_means_aal_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_1_steps_1200_lr_0.01_beta_updates_9', help="the part of the 0-param-update saved stat file between fc_corr_(or other stat name) and .pt.")
    parser.add_argument("-d", "--model_file_fragment", type=str, default='ising_model_light_aal_group_init_means_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_1_steps_1200_lr_0.01_beta_updates_9_param_updates', help="the part of the fitted saved stat files between fc_corr_(or other stat name) and _[the number of parameter update steps].pt.")
    parser.add_argument("-e", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
    parser.add_argument("-f", "--num_thresholds", type=int, default=31, help="number of thresholds")
    parser.add_argument("-g", "--update_increment", type=int, default=1000, help="number of updates between models to test")
    parser.add_argument("-i", "--min_updates", type=int, default=0, help="first number of updates to test")
    parser.add_argument("-j", "--max_updates", type=int, default=31000, help="last number of updates to test")
    parser.add_argument("-k", "--combine_scans", action='store_true', default=False, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-l", "--reset_params", action='store_true', default=False, help="Set this flag in order to do a simulation with h and J reset to the mean states and mean state products, respectively.")
    parser.add_argument("-m", "--zero_h", action='store_true', default=False, help="Set this flag to zero out the h values.")

    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    zero_model_file_fragment = args.zero_model_file_fragment
    print(f'zero_model_file_fragment={zero_model_file_fragment}')
    model_file_fragment = args.model_file_fragment
    print(f'model_file_fragment={model_file_fragment}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
    num_thresholds = args.num_thresholds
    print(f'num_thresholds={num_thresholds}')
    update_increment = args.update_increment
    print(f'update_increment={update_increment}')
    min_updates = args.min_updates
    print(f'min_updates={min_updates}')
    max_updates = args.max_updates
    print(f'max_updates={max_updates}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')
    reset_params = args.reset_params
    print(f'reset_params={reset_params}')
    zero_h = args.zero_h
    print(f'zero_h={zero_h}')
    
    def quantiles_and_min_max(input:torch.Tensor, quantiles:torch.Tensor):
        # Transpose reps and thresholds so that thresholds is first.
        # Then flatten together all dimensions except threshold in case we have a per-node or per-pair stat instead of a whole-model stat.
        input = input.transpose(dim0=0, dim1=1).flatten(start_dim=1, end_dim=-1)
        quant_vals = torch.transpose( torch.quantile(input=input, q=quantiles, dim=-1), dim0=0, dim1=1 )
        min_vals = torch.unsqueeze( torch.min(input=input, dim=-1).values, dim=-1 )
        max_vals = torch.unsqueeze( torch.max(input=input, dim=-1).values, dim=-1 )
        return torch.cat( (quant_vals, min_vals, max_vals), dim=-1 )

    def print_and_save(summary:torch.Tensor, stat_name:str):
        file_name = os.path.join(output_directory, f'summary_{stat_name}_{model_file_fragment}_updates_min_{min_updates}_max_{max_updates}_increment_{update_increment}.pt')
        torch.save(obj=summary, f=file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {file_name}')
    
    def get_series_summary_no_split(stat_name:str, zero_model_file_fragment:str, model_file_fragment:str, update_counts:torch.Tensor, quantiles:torch.Tensor):
        num_model_files = update_counts.numel()
        num_quants = quantiles.numel()
        all_summary = torch.zeros( size=(num_model_files, num_thresholds, num_quants+2), dtype=float_type, device=device )
        for file_index in range(num_model_files):
            update_count = update_counts[file_index]
            if update_count == 0:
                stat_file = os.path.join(output_directory, f'{stat_name}_{zero_model_file_fragment}_test_length_{sim_length}.pt')
            else:
                stat_file = os.path.join(output_directory, f'{stat_name}_{model_file_fragment}_{update_count}_test_length_{sim_length}.pt')
            stat_values = torch.load(stat_file)
            print( f'time {time.time()-code_start_time:.3f}, loaded {stat_file}, size', stat_values.size() )
            summary = quantiles_and_min_max(input=stat_values, quantiles=quantiles)
            # print(summary)
            all_summary[file_index,:,:] = summary
        print_and_save(summary=all_summary, stat_name=f'{stat_name}_all')
        return all_summary
    
    update_counts = torch.arange(start=min_updates, end=max_updates+1, step=update_increment, dtype=int_type, device=device)
    num_model_files = update_counts.numel()
    quantiles = torch.tensor(data=[0.5, 0.025, 0.975], dtype=float_type, device=device)
    # We do not need the returned values, because we just want to save the summaries to files, which the function already does.
    # flip_rate
    for stat_name in ['cov_rmse', 'fc_rmse', 'fc_corr']:
        get_series_summary_no_split(stat_name=stat_name, zero_model_file_fragment=zero_model_file_fragment, model_file_fragment=model_file_fragment, update_counts=update_counts, quantiles=quantiles)
print(f'time {time.time()-code_start_time:.3f}, done')