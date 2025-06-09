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

    parser = argparse.ArgumentParser(description="Expand window while tracking the expected values of observables. Compare to expected values from data.")
    parser.add_argument("-a", "--data_directory", type=str, default='D:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_mean_std_1', help="part of the data mean state and state product file names between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-f", "--combine_scans", action='store_true', default=True, help="Set this flag to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-n", "--reverse", action='store_true', default=True, help="Set this flag to expand from the end of the time series instead of the beginning.")
    parser.add_argument("-j", "--min_steps", type=int, default=100, help="minimum number of data steps before we compute the first FC RMSE")
    parser.add_argument("-k", "--max_steps", type=int, default=1200, help="maximum number of data steps for which we compute the last FC RMSE")
    parser.add_argument("-l", "--step_increment", type=int, default=100, help="number of data steps between checks of FC RMSE")
    parser.add_argument("-m", "--threshold", type=float, default=1.0, help="threshold at which to binarize time series in SD above the mean")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')
    min_steps = args.min_steps
    print(f'min_steps={min_steps}')
    max_steps = args.max_steps
    print(f'max_steps={max_steps}')
    step_increment = args.step_increment
    print(f'step_increment={step_increment}')
    threshold = args.threshold
    print(f'threshold={threshold}')
    reverse = args.reverse
    print(f'reverse={reverse}')

    data_ts_file = os.path.join(data_directory, f'data_ts_all_as_is.pt')
    def load_and_binarize_ts(data_ts_file:str, threshold:float, reverse:bool=False):
        data_ts = torch.load(data_ts_file, weights_only=False)
        ts_std, ts_mean = torch.std_mean(input=data_ts, dim=-1, keepdim=True)
        data_ts -= ts_mean
        data_ts /= ts_std
        data_ts = (data_ts >= threshold).float()
        data_ts *= 2.0
        data_ts -= 1.0
        if reverse:
            data_ts = data_ts.flip( dims=(-1,) )
        # data_ts is originally scans x subjects x nodes x steps.
        # Permute it to subjects x nodes x scans x steps.
        # data_ts = data_ts.permute( dims=(1, 2, 0, 3) )
        # Then flatten to subjects x nodes x scans*steps.
        # data_ts = data_ts.flatten(start_dim=-2, end_dim=-1)
        return data_ts
    data_ts = load_and_binarize_ts(data_ts_file=data_ts_file, threshold=threshold, reverse=reverse)
    print( f'time {time.time()-code_start_time:.3f}, loaded {data_ts_file}, size', data_ts.size() )

    def load_data_means(data_file_name_part:str):
        print('loading data time series state and state product means')
        state_mean_file = os.path.join(output_directory, f'mean_state_{data_file_name_part}.pt')
        state_mean = torch.load(state_mean_file)
        # On load, the dimensions of target_state_mean should be subject x node or scan x subject x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded state_mean with size', state_mean.size() )
        state_product_mean_file = os.path.join(output_directory, f'mean_state_product_{data_file_name_part}.pt')
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
    target_cov = isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
    target_fc = isingmodellight.get_fc_binary(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
    
    def get_data_means(data_ts:torch.Tensor, start_step:int, num_steps:int):
        data_ts_interval = data_ts[:,:,:,start_step:(start_step+num_steps)]
        return data_ts_interval.mean(dim=-1), torch.matmul( data_ts_interval, data_ts_interval.transpose(dim0=-2, dim1=-1) )/num_steps, torch.mean( torch.abs(data_ts_interval[:,:,:,1:] - data_ts_interval[:,:,:,:-1])/2.0, dim=-1 )

    steps_completed = 0
    sim_state_mean, sim_state_product_mean, flip_rate = get_data_means(data_ts=data_ts, start_step=steps_completed, num_steps=min_steps)
    steps_completed += min_steps
    print( f'time {time.time()-code_start_time:.3f}, done simulating {steps_completed} steps' )
    print( 'sim_state_mean size', sim_state_mean.size() )
    print( 'sim_state_product_mean size', sim_state_product_mean.size() )
    print( 'flip_rate size', flip_rate.size() )
    print(f'flip rate  min {flip_rate.min():.3g}, mean {flip_rate.mean():.3g}, max {flip_rate.max():.3g}')

    sim_state_mean_sum = steps_completed * sim_state_mean
    sim_state_product_mean_sum = steps_completed * sim_state_product_mean
    flip_rate_sum = steps_completed * flip_rate
    state_mean_rmse = isingmodellight.get_pairwise_rmse(mat1=sim_state_mean, mat2=target_state_mean)
    print(f'state mean RMSE  min {state_mean_rmse.min():.3g}, mean {state_mean_rmse.mean():.3g}, max {state_mean_rmse.max():.3g}')
    state_product_mean_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_state_product_mean, mat2=target_state_product_mean)
    print(f'state product mean RMSE  min {state_product_mean_rmse.min():.3g}, mean {state_product_mean_rmse.mean():.3g}, max {state_product_mean_rmse.max():.3g}')
    sim_cov = isingmodellight.get_cov(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
    cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_cov, mat2=target_cov)
    print(f'covariance RMSE  min {cov_rmse.min():.3g}, mean {cov_rmse.mean():.3g}, max {cov_rmse.max():.3g}')
    cov_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_cov, mat2=target_cov)
    print(f'covariance correlation  min {cov_corr.min():.3g}, mean {cov_corr.mean():.3g}, max {cov_corr.max():.3g}')
    sim_fc = isingmodellight.get_fc_binary(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
    fc_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_fc, mat2=target_fc)
    print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}')
    fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc)
    print(f'FC correlation min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}')

    def expand_to_series(mat:torch.Tensor, num_reps:int):
        mat_series = torch.zeros(  size=[num_reps]+list( mat.size() ), dtype=mat.dtype, device=mat.device  )
        mat_series[0] = mat
        return mat_series
    num_windows = (max_steps - min_steps)//step_increment + 1
    flip_rate_series = expand_to_series(mat=flip_rate, num_reps=num_windows)
    state_mean_rmse_series = expand_to_series(mat=state_mean_rmse, num_reps=num_windows)
    state_product_mean_rmse_series = expand_to_series(mat=state_product_mean_rmse, num_reps=num_windows)
    cov_rmse_series = expand_to_series(mat=cov_rmse, num_reps=num_windows)
    cov_corr_series = expand_to_series(mat=cov_corr, num_reps=num_windows)
    fc_rmse_series = expand_to_series(mat=fc_rmse, num_reps=num_windows)
    fc_corr_series = expand_to_series(mat=fc_corr, num_reps=num_windows)

    for window_index in range(1,num_windows):
        sim_state_mean_increment, sim_state_product_mean_increment, flip_rate_increment = get_data_means(data_ts=data_ts, start_step=steps_completed, num_steps=step_increment)
        steps_completed += step_increment
        print( f'time {time.time()-code_start_time:.3f}, done simulating {steps_completed} steps' )
        sim_state_mean_sum += step_increment * sim_state_mean_increment
        sim_state_mean = sim_state_mean_sum/steps_completed
        sim_state_product_mean_sum += step_increment * sim_state_product_mean_increment
        sim_state_product_mean = sim_state_product_mean_sum/steps_completed
        flip_rate_sum += step_increment * flip_rate_increment
        flip_rate = flip_rate_sum/steps_completed
        
        print(f'flip rate min {flip_rate.min():.3g}, mean {flip_rate.mean():.3g}, max {flip_rate.max():.3g}')
        flip_rate_series[window_index] = flip_rate
        state_mean_rmse = isingmodellight.get_pairwise_rmse(mat1=sim_state_mean, mat2=target_state_mean)
        print(f'state mean RMSE min {state_mean_rmse.min():.3g}, mean {state_mean_rmse.mean():.3g}, max {state_mean_rmse.max():.3g}')
        state_mean_rmse_series[window_index] = state_mean_rmse
        state_product_mean_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_state_product_mean, mat2=target_state_product_mean)
        print(f'state product mean RMSE min {state_product_mean_rmse.min():.3g}, mean {state_product_mean_rmse.mean():.3g}, max {state_product_mean_rmse.max():.3g}')
        state_product_mean_rmse_series[window_index] = state_product_mean_rmse
        
        sim_cov = isingmodellight.get_cov(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
        cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_cov, mat2=target_cov)
        print(f'covariance RMSE min {cov_rmse.min():.3g}, mean {cov_rmse.mean():.3g}, max {cov_rmse.max():.3g}')
        cov_rmse_series[window_index] = cov_rmse
        cov_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_cov, mat2=target_cov)
        print(f'covariance correlation min {cov_corr.min():.3g}, mean {cov_corr.mean():.3g}, max {cov_corr.max():.3g}')
        cov_corr_series[window_index] = cov_corr

        sim_fc = isingmodellight.get_fc_binary(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
        fc_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_fc, mat2=target_fc)
        print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}')
        fc_rmse_series[window_index] = fc_rmse
        fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc)
        print(f'FC correlation min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}')
        fc_corr_series[window_index] = fc_corr
    
    reverse_string = '_reverse' if reverse else ''
    sim_file_fragment = f'{data_file_name_part}{reverse_string}_steps_min_{min_steps}_max_{max_steps}_inc_{step_increment}'
    flip_rate_file = os.path.join(output_directory, f'flip_rate_{sim_file_fragment}.pt')
    torch.save(obj=flip_rate_series, f=flip_rate_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file}')
    state_mean_rmse_file = os.path.join(output_directory, f'state_mean_rmse_{sim_file_fragment}.pt')
    torch.save(obj=state_mean_rmse_series, f=state_mean_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {state_mean_rmse_file}')
    state_product_mean_rmse_file = os.path.join(output_directory, f'state_product_mean_rmse_{sim_file_fragment}.pt')
    torch.save(obj=state_product_mean_rmse_series, f=state_product_mean_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {state_product_mean_rmse_file}')
    cov_rmse_file = os.path.join(output_directory, f'cov_rmse_{sim_file_fragment}.pt')
    torch.save(obj=cov_rmse_series, f=cov_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {cov_rmse_file}')
    cov_corr_file = os.path.join(output_directory, f'cov_corr_{sim_file_fragment}.pt')
    torch.save(obj=cov_corr_series, f=cov_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {cov_rmse_file}')
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{sim_file_fragment}.pt')
    torch.save(obj=fc_rmse_series, f=fc_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file}')
    fc_corr_file = os.path.join(output_directory, f'fc_corr_{sim_file_fragment}.pt')
    torch.save(obj=fc_corr_series, f=fc_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_corr_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')