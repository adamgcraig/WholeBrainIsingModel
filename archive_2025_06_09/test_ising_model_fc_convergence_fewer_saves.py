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
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_mean_std_1', help="part of the data mean state and state product file names between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--init_params_file_name_part", type=str, default='all_mean_std_1', help="similar to data_file_name_part but used along with reset_params to select the data mean state and state product files to use as parameter values .pt")
    parser.add_argument("-e", "--model_file_fragment", type=str, default='all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_45000', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
    parser.add_argument("-f", "--combine_scans", action='store_true', default=False, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-g", "--reset_params", action='store_true', default=False, help="Set this flag in order to do a simulation with h and J reset to the mean states and mean state products, respectively.")
    parser.add_argument("-i", "--zero_h", action='store_true', default=False, help="Set this flag to zero out the h values.")
    parser.add_argument("-j", "--min_steps", type=int, default=100, help="minimum number of simulation steps before we compute the first FC RMSE")
    parser.add_argument("-k", "--max_steps", type=int, default=120000, help="maximum number of simulation steps for which we compute the last FC RMSE")
    parser.add_argument("-l", "--step_increment", type=int, default=100, help="number of simulation steps between checks of FC RMSE")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    init_params_file_name_part = args.init_params_file_name_part
    print(f'init_params_file_name_part={init_params_file_name_part}')
    model_file_fragment = args.model_file_fragment
    print(f'model_file_fragment={model_file_fragment}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')
    reset_params = args.reset_params
    print(f'reset_params={reset_params}')
    zero_h = args.zero_h
    print(f'zero_h={zero_h}')
    min_steps = args.min_steps
    print(f'min_steps={min_steps}')
    max_steps = args.max_steps
    print(f'max_steps={max_steps}')
    step_increment = args.step_increment
    print(f'step_increment={step_increment}')

    def load_data_means(data_file_name_part:str):
        print('loading data time series state and state product means')
        state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
        state_mean = torch.load(state_mean_file)
        # On load, the dimensions of target_state_mean should be subject x node or scan x subject x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded state_mean with size', state_mean.size() )
        state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
        state_product_mean = torch.load(state_product_mean_file)
        # On load, the dimensions of target_state_product_mean should be subject x node-pair, subject x node x node, scan x subject x node-pair, or scan x subject x node x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', state_product_mean.size() )
        # Assume that either both target Tensors have a scan dimension, or neither does.
        # If they have a scan dimension, then first remove it, either by averaging over scans or flattening together the subject and scan dimensions.
        state_mean_size = state_mean.size()
        # target_state_product_mean_size = target_state_product_mean.size()
        num_batch_dims = len(state_mean_size) - 1
        if num_batch_dims > 1:
            if combine_scans:
                extra_dim_range = tuple( range(num_batch_dims-1) )# average over all extra batch dimensions
                print( 'averaging over extra batch dimensions', extra_dim_range )
                state_mean = torch.mean(state_mean, dim=extra_dim_range, keepdim=False)
                state_product_mean = torch.mean(state_product_mean, dim=extra_dim_range, keepdim=False)
            else:
                print('flattening extra batch dimensions')
                state_mean = torch.flatten(state_mean, start_dim=0, end_dim=-2)
                state_product_mean = torch.flatten(state_product_mean, start_dim=0, end_dim=-3)
        # Regardless of whether the data originally had a scan dimension, we add in a singleton model replica dimension so that we can broadcast with h and J.
        print('prepending singleton model replica dimension')
        state_mean = state_mean.unsqueeze(dim=0)
        state_product_mean = state_product_mean.unsqueeze(dim=0)
        # This was getting too complicated.
        # Just assume we are reading in batches of square matrices.
        # # We want to work with the mean state products as square matrices, not upper triangular part vectors.
        # if len( state_product_mean.size() ) < 4:
        #     print('converting mean state products from upper triangular parts to square matrices')
        #     state_product_mean = isingmodellight.triu_to_square_pairs(triu_pairs=state_product_mean, diag_fill=0)
        return state_mean, state_product_mean
    
    print('loading target data time series state and state product means')
    target_state_mean, target_state_product_mean = load_data_means(data_file_name_part=data_file_name_part)
    print( f'time {time.time()-code_start_time:.3f}, target_state_mean size', target_state_mean.size() )
    print( f'time {time.time()-code_start_time:.3f}, target_state_product_mean size', target_state_product_mean.size() )
    target_fc = isingmodellight.get_fc_binary(state_mean=target_state_mean, state_product_mean=target_state_product_mean)

    model_file = os.path.join(data_directory, f'ising_model_light_{model_file_fragment}.pt')
    model = torch.load(f=model_file)
    if reset_params:
        print('loading initial parameter state and state product means')
        param_state_mean, param_state_product_mean = load_data_means(data_file_name_part=init_params_file_name_part)
        print( f'time {time.time()-code_start_time:.3f}, param_state_mean size', param_state_mean.size() )
        print( f'time {time.time()-code_start_time:.3f}, param_state_product_mean size', param_state_product_mean.size() )
        model.h[:,:,:] = param_state_mean
        model.J[:,:,:,:] = param_state_product_mean
        reset_str = '_reset'
    else:
        reset_str = ''
    if zero_h:
        model.h.zero_()
        zero_h_str = '_no_h'
    else:
        zero_h_str = ''
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file}' )
    
    sim_state_mean, sim_state_product_mean = model.simulate_and_record_means_pmb(num_steps=min_steps)
    steps_completed = min_steps
    print( f'time {time.time()-code_start_time:.3f}, done simulating {steps_completed} steps' )
    print( 'sim_state_mean size', sim_state_mean.size() )
    print( 'sim_state_product_mean size', sim_state_product_mean.size() )
    sim_state_mean_sum = steps_completed * sim_state_mean
    sim_state_product_mean_sum = steps_completed * sim_state_product_mean
    sim_fc = isingmodellight.get_fc_binary(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)

    state_mean_rmse = isingmodellight.get_pairwise_rmse(mat1=sim_state_mean, mat2=target_state_mean)
    print(f'state mean RMSE  min {state_mean_rmse.min():.3g}, mean {state_mean_rmse.mean():.3g}, max {state_mean_rmse.max():.3g}')
    state_product_mean_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_state_product_mean, mat2=target_state_product_mean)
    print(f'state product mean RMSE  min {state_product_mean_rmse.min():.3g}, mean {state_product_mean_rmse.mean():.3g}, max {state_product_mean_rmse.max():.3g}')
    fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc)
    print(f'FC correlation min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}')

    def expand_to_series(mat:torch.Tensor, num_reps:int):
        mat_series = torch.zeros(  size=[num_reps]+list( mat.size() ), dtype=mat.dtype, device=mat.device  )
        mat_series[0] = mat
        return mat_series
    num_windows = (max_steps - min_steps)//step_increment + 1
    state_mean_rmse_series = expand_to_series(mat=state_mean_rmse, num_reps=num_windows)
    state_product_mean_rmse_series = expand_to_series(mat=state_product_mean_rmse, num_reps=num_windows)
    fc_corr_series = expand_to_series(mat=fc_corr, num_reps=num_windows)

    for window_index in range(1,num_windows):
        sim_state_mean_increment, sim_state_product_mean_increment, flip_rate_increment = model.simulate_and_record_means_pmb(num_steps=step_increment)
        steps_completed += step_increment
        print( f'time {time.time()-code_start_time:.3f}, done simulating {steps_completed} steps' )
        sim_state_mean_sum += step_increment * sim_state_mean_increment
        sim_state_mean = sim_state_mean_sum/steps_completed
        sim_state_product_mean_sum += step_increment * sim_state_product_mean_increment
        sim_state_product_mean = sim_state_product_mean_sum/steps_completed
        sim_fc = isingmodellight.get_fc_binary(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
        
        state_mean_rmse = isingmodellight.get_pairwise_rmse(mat1=sim_state_mean, mat2=target_state_mean)
        print(f'state mean RMSE min {state_mean_rmse.min():.3g}, mean {state_mean_rmse.mean():.3g}, max {state_mean_rmse.max():.3g}')
        state_mean_rmse_series[window_index] = state_mean_rmse
        state_product_mean_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_state_product_mean, mat2=target_state_product_mean)
        print(f'state product mean RMSE min {state_product_mean_rmse.min():.3g}, mean {state_product_mean_rmse.mean():.3g}, max {state_product_mean_rmse.max():.3g}')
        state_product_mean_rmse_series[window_index] = state_product_mean_rmse
        fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc)
        print(f'FC correlation min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}')
        fc_corr_series[window_index] = fc_corr

    sim_file_fragment = f'{model_file_fragment}{reset_str}{zero_h_str}_steps_min_{min_steps}_max_{max_steps}_inc_{step_increment}'
    state_mean_rmse_file = os.path.join(output_directory, f'state_mean_rmse_{sim_file_fragment}.pt')
    torch.save(obj=state_mean_rmse_series, f=state_mean_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {state_mean_rmse_file}')
    state_product_mean_rmse_file = os.path.join(output_directory, f'state_product_mean_rmse_{sim_file_fragment}.pt')
    torch.save(obj=state_product_mean_rmse_series, f=state_product_mean_rmse_file)
    fc_corr_file = os.path.join(output_directory, f'fc_corr_{sim_file_fragment}.pt')
    torch.save(obj=fc_corr_series, f=fc_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_corr_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')