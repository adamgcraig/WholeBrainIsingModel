import os
import torch
import time
import argparse
import hcpdatautilsnopandas as hcp
import isingmodellight
from isingmodellight import IsingModelLight

def get_binarized_info(data_ts:torch.Tensor, threshold:torch.Tensor, data_cov:torch.Tensor, data_fc:torch.Tensor):
    binarized_ts = 2*(data_ts > threshold).float() - 1
    flip_rate = (binarized_ts[:,:,:,1:] != binarized_ts[:,:,:,:-1]).float().mean(dim=-1)
    state_mean = binarized_ts.mean(dim=-1)
    state_product_mean = torch.matmul( binarized_ts, binarized_ts.transpose(dim0=-2, dim1=-1) )/binarized_ts.size(dim=-1)
    cov_binary = isingmodellight.get_cov(state_mean=state_mean, state_product_mean=state_product_mean)
    cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=cov_binary, mat2=data_cov)
    fc_binary = isingmodellight.get_fc_binary(state_mean=state_mean, state_product_mean=state_product_mean)
    fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=fc_binary, mat2=data_fc, epsilon=0)
    return flip_rate, cov_rmse, fc_corr, state_mean.mean( dim=(0,1) ), state_product_mean.mean( dim=(0,1) )

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float32
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Test the effect of binarization threshold on our ability to fit a group Ising model.")
    parser.add_argument("-a", "--input_directory", type=str, default='D:\\Ising_model_results_daai', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-d", "--data_ts_string", type=str, default='all_as_is', help="file name of the data time series file between data_ts_ and .pt")
    parser.add_argument("-c", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-e", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    parser.add_argument("-f", "--num_thresholds", type=int, default=31, help="number of binarization thresholds in std. dev.s above the mean to try")
    parser.add_argument("-g", "--min_threshold", type=float, default=0, help="minimum threshold in std. dev.s")
    parser.add_argument("-i", "--max_threshold", type=float, default=3, help="minimum threshold in std. dev.s")

    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_ts_string = args.data_ts_string
    print(f'data_ts_string={data_ts_string}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')
    num_thresholds = args.num_thresholds
    print(f'num_thresholds={num_thresholds}')
    min_threshold = args.min_threshold
    print(f'min_threshold={min_threshold}')
    max_threshold = args.max_threshold
    print(f'max_threshold={max_threshold}')

    data_ts_file = os.path.join(input_directory, f'data_ts_{data_ts_string}.pt')
    data_ts = torch.load(data_ts_file, weights_only=False)[:,training_subject_start:training_subject_end,:,:].type(dtype=float_type)
    num_ts, num_subjects, num_nodes, num_steps = data_ts.size()
    print(f'time {time.time()-code_start_time:.3f}, computing unbinarized means and flip rates...')
    # data_ts /= data_ts.abs().max(dim=-1).values# Divide the whole time series by the largest absolute value so that the FC stays the same but the means and mean products are <= 1.
    target_state_mean = data_ts.mean(dim=-1, keepdim=False)
    print( f'state mean', target_state_mean.size() )
    state_mean_product = torch.matmul( data_ts, data_ts.transpose(dim0=-2, dim1=-1) )/data_ts.size(dim=-1)
    print( f'state mean', state_mean_product.size() )
    data_cov = isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=state_mean_product)
    data_fc = isingmodellight.get_fc(state_mean=target_state_mean, state_product_mean=state_mean_product, epsilon=0)
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    threshold_choices = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device)
    num_choices = threshold_choices.numel()
    quantiles = torch.tensor(data=[0.025, 0.5, 0.975], dtype=float_type, device=device)
    num_quantiles = len(quantiles)
    flip_rate_quantiles = torch.zeros( size=(num_choices, num_quantiles), dtype=float_type, device=device )
    cov_rmse_quantiles = torch.zeros( size=(num_choices, num_quantiles), dtype=float_type, device=device )
    fc_corr_quantiles = torch.zeros( size=(num_choices, num_quantiles), dtype=float_type, device=device )
    target_state_mean = torch.zeros( size=(num_choices, num_nodes), dtype=data_ts.dtype, device=data_ts.device )
    target_state_product_mean = torch.zeros( size=(num_choices, num_nodes, num_nodes), dtype=data_ts.dtype, device=data_ts.device )
    for choice_index in range(num_choices):
        choice = threshold_choices[choice_index]
        print(f'threshold {choice_index+1} of {num_choices}: {choice:.3g}')
        threshold = data_ts_mean + choice*data_ts_std
        flip_rate, cov_rmse, fc_corr, target_state_mean[choice_index,:], target_state_product_mean[choice_index,:,:] = get_binarized_info(data_ts=data_ts, threshold=threshold, data_cov=data_cov, data_fc=data_fc)
        flip_rate_quantiles[choice_index,:] = torch.quantile(input=flip_rate, q=quantiles)
        cov_rmse_quantiles[choice_index,:] = torch.quantile(input=cov_rmse, q=quantiles)
        fc_corr_quantiles[choice_index,:] = torch.quantile(input=fc_corr, q=quantiles)
    target_state_mean_file = os.path.join(output_directory, f'mean_state_thresholds_{num_choices}_min_{min_threshold:.3g}_max_{max_threshold:.3g}.pt')
    torch.save( obj=target_state_mean, f=target_state_mean_file )
    print(f'time {time.time()-code_start_time:.3f}, saved {target_state_mean_file}')
    target_state_product_mean_file = os.path.join(output_directory, f'mean_state_product_thresholds_{num_choices}_min_{min_threshold:.3g}_max_{max_threshold:.3g}.pt')
    torch.save( obj=target_state_product_mean, f=target_state_product_mean_file )
    print(f'time {time.time()-code_start_time:.3f}, saved {target_state_product_mean_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')