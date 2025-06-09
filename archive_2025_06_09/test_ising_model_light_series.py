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
    parser.add_argument("-a", "--data_directory", type=str, default='D:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_mean_std_1', help="part of the output file name between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--model_file_fragment", type=str, default='ising_model_light_group_thresholds_31_min_0_max_3_betas_101_min_1e-10_max_0.05_steps_1200_lr_0.01_beta_updates_8_param_updates_40000_to_thresh_1_reps_5_subj_837_individual_updates', help="the part of the Ising model file between ising_model_ and _[the number of parameter update steps].pt.")
    parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
    parser.add_argument("-m", "--update_increment", type=int, default=1000, help="number of updates between models to test")
    parser.add_argument("-k", "--min_updates", type=int, default=0, help="first number of updates to test")
    parser.add_argument("-l", "--max_updates", type=int, default=62000, help="last number of updates to test")
    parser.add_argument("-g", "--combine_scans", action='store_true', default=True, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-i", "--reset_params", action='store_true', default=False, help="Set this flag in order to do a simulation with h and J reset to the mean states and mean state products, respectively.")
    parser.add_argument("-j", "--zero_h", action='store_true', default=False, help="Set this flag to zero out the h values.")
    parser.add_argument("-n", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-e", "--training_subject_end", type=int, default=837, help="1 past last training subject index")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    model_file_fragment = args.model_file_fragment
    print(f'model_file_fragment={model_file_fragment}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
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
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')

    epsilon = 0.0

    def load_target_means(data_file_name_part:str):
        print('loading data time series state and state product means')
        target_state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
        target_state_mean = torch.load(target_state_mean_file, weights_only=False)
        # On load, the dimensions of target_state_mean should be subject x node or scan x subject x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded target_state_mean with size', target_state_mean.size() )
        target_state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
        target_state_product_mean = torch.load(target_state_product_mean_file, weights_only=False)
        # On load, the dimensions of target_state_product_mean should be subject x node-pair, subject x node x node, scan x subject x node-pair, or scan x subject x node x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', target_state_product_mean.size() )
        # Assume that either both target Tensors have a scan dimension, or neither does.
        # If they have a scan dimension, then first remove it, either by averaging over scans or flattening together the subject and scan dimensions.
        target_state_mean_size = target_state_mean.size()
        # target_state_product_mean_size = target_state_product_mean.size()
        num_batch_dims = len(target_state_mean_size) - 1
        if num_batch_dims > 1:
            if combine_scans:
                extra_dim_range = tuple( range(num_batch_dims-1) )# average over all extra batch dimensions
                print( 'averaging over extra batch dimensions', extra_dim_range )
                target_state_mean = torch.mean(target_state_mean, dim=extra_dim_range, keepdim=False)
                target_state_product_mean = torch.mean(target_state_product_mean, dim=extra_dim_range, keepdim=False)
            else:
                print('flattening extra batch dimensions')
                target_state_mean = torch.flatten(target_state_mean, start_dim=0, end_dim=-2)
                target_state_product_mean = torch.flatten(target_state_product_mean, start_dim=0, end_dim=-3)
        # Regardless of whether the data originally had a scan dimension, we add in a singleton model replica dimension so that we can broadcast with h and J.
        print('prepending singleton model replica dimension')
        target_state_mean = target_state_mean.unsqueeze(dim=0)
        target_state_product_mean = target_state_product_mean.unsqueeze(dim=0)
        # This is getting too complicated. Just assume  target_state_product_mean is square.
        # We want to work with the mean state products as square matrices, not upper triangular part vectors.
        # if len( target_state_product_mean.size() ) < 4:
        #     print('converting mean state products from upper triangular parts to square matrices')
        #     target_state_product_mean = isingmodellight.triu_to_square_pairs( triu_pairs=target_state_product_mean, diag_fill=0 )
        print( f'time {time.time()-code_start_time:.3f}, target_state_mean size', target_state_mean.size() )
        print( f'time {time.time()-code_start_time:.3f}, target_state_product_mean size', target_state_product_mean.size() )
        return target_state_mean, target_state_product_mean

    def test_model(model_file_fragment_with_updates:str, target_state_mean:torch.Tensor, target_state_product_mean:torch.Tensor):
        model_file = os.path.join(data_directory, f'{model_file_fragment_with_updates}.pt')
        model = torch.load(f=model_file, weights_only=False)
        if reset_params:
            model.h[:,:,:] = target_state_mean
            model.J[:,:,:,:] = target_state_product_mean
            reset_str = '_reset'
        else:
            reset_str = ''
        if zero_h:
            model.h.zero_()
            zero_h_str = '_no_h'
        else:
            zero_h_str = ''
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file}' )
        print('running simulation...')
        sim_state_mean, sim_state_product_mean, flip_rate = model.simulate_and_record_means_and_flip_rate_pmb(num_steps=sim_length)
        sim_file_fragment = f'{model_file_fragment_with_updates}{reset_str}{zero_h_str}_test_length_{sim_length}'
        print( 'sim_state_mean size', sim_state_mean.size() )
        print( 'sim_state_product_mean size', sim_state_product_mean.size() )
        print( 'flip_rate size', flip_rate.size() )
        print( f'time {time.time()-code_start_time:.3f}, done simulating {sim_length} steps' )
        print(f'flip rate  min {flip_rate.min():.3g}, mean {flip_rate.mean():.3g}, max {flip_rate.max():.3g}')
        flip_rate_file = os.path.join(output_directory, f'flip_rate_{sim_file_fragment}.pt')
        torch.save(obj=flip_rate, f=flip_rate_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file}')
        state_mean_rmse = isingmodellight.get_pairwise_rmse(mat1=sim_state_mean, mat2=target_state_mean)
        print(f'state mean RMSE  min {state_mean_rmse.min():.3g}, mean {state_mean_rmse.mean():.3g}, max {state_mean_rmse.max():.3g}')
        state_mean_rmse_file = os.path.join(output_directory, f'state_mean_rmse_{sim_file_fragment}.pt')
        torch.save(obj=state_mean_rmse, f=state_mean_rmse_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {state_mean_rmse_file}')
        state_product_mean_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_state_product_mean, mat2=target_state_product_mean)
        print(f'state product mean RMSE  min {state_product_mean_rmse.min():.3g}, mean {state_product_mean_rmse.mean():.3g}, max {state_product_mean_rmse.max():.3g}')
        state_product_mean_rmse_file = os.path.join(output_directory, f'state_product_mean_rmse_{sim_file_fragment}.pt')
        torch.save(obj=state_product_mean_rmse, f=state_product_mean_rmse_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {state_product_mean_rmse_file}')
        target_cov = isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
        sim_cov = isingmodellight.get_cov(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
        cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_cov, mat2=target_cov)
        print(f'covariance RMSE  min {cov_rmse.min():.3g}, mean {cov_rmse.mean():.3g}, max {cov_rmse.max():.3g}')
        cov_rmse_file = os.path.join(output_directory, f'cov_rmse_{sim_file_fragment}.pt')
        torch.save(obj=cov_rmse, f=cov_rmse_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {cov_rmse_file}')
        target_fc = isingmodellight.get_fc_binary(state_mean=target_state_mean, state_product_mean=target_state_product_mean, epsilon=epsilon)
        sim_fc = isingmodellight.get_fc_binary(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean, epsilon=epsilon)
        fc_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_fc, mat2=target_fc)
        print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}')
        fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{sim_file_fragment}.pt')
        torch.save(obj=fc_rmse, f=fc_rmse_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file}')
        fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc, epsilon=epsilon)
        print(f'FC correlation min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}')
        fc_corr_file = os.path.join(output_directory, f'fc_corr_{sim_file_fragment}.pt')
        torch.save(obj=fc_corr, f=fc_corr_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {fc_corr_file}')
        return flip_rate, cov_rmse, fc_rmse, fc_corr
    
    def quantiles_and_min_max(input:torch.Tensor, quantiles:torch.Tensor):
        input = input[:,training_subject_start:training_subject_end]
        return torch.cat(  ( torch.quantile(input=input, q=quantiles), torch.min(input).unsqueeze(dim=0), torch.max(input).unsqueeze(dim=0) ), dim=0  )
    
    def print_and_save(summary:torch.Tensor, stat_name:str):
        file_name = os.path.join(output_directory, f'{stat_name}_summary_{model_file_fragment}_updates_min_{min_updates}_max_{max_updates}_increment_{update_increment}.pt')
        torch.save(obj=summary, f=file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {file_name}')
    
    target_state_mean, target_state_product_mean = load_target_means(data_file_name_part=data_file_name_part)
    update_counts = torch.arange(start=min_updates, end=max_updates+1, step=update_increment, dtype=int_type, device=device)
    num_model_files = update_counts.numel()
    quantiles = torch.tensor(data=[0.5, 0.025, 0.975], dtype=float_type, device=device)
    num_quants = len(quantiles)
    flip_rate_summary = torch.zeros(  size=( num_model_files, num_quants+2 ), dtype=float_type, device=device  )
    cov_rmse_summary = torch.zeros_like(flip_rate_summary)
    fc_rmse_summary = torch.zeros_like(flip_rate_summary)
    fc_corr_summary = torch.zeros_like(flip_rate_summary)
    for model_file_index in range(num_model_files):
        num_updates = update_counts[model_file_index]
        print(f'model file {model_file_index+1} of {num_model_files}, update count {num_updates}')
        model_file_fragment_with_updates = f'{model_file_fragment}_{num_updates}'
        flip_rate, cov_rmse, fc_rmse, fc_corr = test_model(model_file_fragment_with_updates=model_file_fragment_with_updates, target_state_mean=target_state_mean, target_state_product_mean=target_state_product_mean)
        flip_rate_summary[model_file_index,:] = quantiles_and_min_max(input=flip_rate, quantiles=quantiles)
        cov_rmse_summary[model_file_index,:] = quantiles_and_min_max(input=cov_rmse, quantiles=quantiles)
        fc_rmse_summary[model_file_index,:] = quantiles_and_min_max(input=fc_rmse, quantiles=quantiles)
        fc_corr_summary[model_file_index,:] = quantiles_and_min_max(input=fc_corr, quantiles=quantiles)
    print_and_save(summary=flip_rate_summary, stat_name='flip_rate')
    print_and_save(summary=cov_rmse_summary, stat_name='cov_rmse')
    print_and_save(summary=fc_rmse_summary, stat_name='fc_rmse')
    print_and_save(summary=fc_corr_summary, stat_name='fc_corr')
print(f'time {time.time()-code_start_time:.3f}, done')