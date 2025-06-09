import os
import torch
import time
import argparse
import hcpdatautilsnopandas as hcp
import isingmodellight

def get_binarized_info(data_ts:torch.Tensor, threshold:torch.Tensor, data_cov:torch.Tensor, data_fc:torch.Tensor):
    binarized_ts = 2*(data_ts > threshold).float() - 1
    flip_rate = (binarized_ts[:,:,:,1:] != binarized_ts[:,:,:,:-1]).float().mean(dim=-1)
    state_mean = binarized_ts.mean(dim=-1)
    state_product_mean = torch.matmul( binarized_ts, binarized_ts.transpose(dim0=-2, dim1=-1) )/binarized_ts.size(dim=-1)
    cov_binary = isingmodellight.get_cov(state_mean=state_mean, state_product_mean=state_product_mean)
    cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=cov_binary, mat2=data_cov)
    fc_binary = isingmodellight.get_fc_binary(state_mean=state_mean, state_product_mean=state_product_mean)
    fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=fc_binary, mat2=data_fc, epsilon=0)
    return flip_rate, cov_rmse, fc_corr

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Rescale each fMRI data time series so that the maximum absolute value of any region state is 1. Then compute and save the means of the states and state products.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\HCP_data', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-c", "--data_subset", type=str, default='all', help="'training', 'validation', 'testing', or 'all'")
    parser.add_argument("-d", "--file_name_fragment", type=str, default='as_is', help="part of the output file name between mean_state_[data_subset]_ or mean_state_product_[data_subset]_ and .pt")
    parser.add_argument("-e", "--num_thresholds", type=int, default=1000, help="number of binarization thresholds in std. dev.s above the mean to try")
    parser.add_argument("-f", "--min_threshold", type=float, default=-4, help="minimum threshold in std. dev.s")
    parser.add_argument("-g", "--max_threshold", type=float, default=4, help="minimum threshold in std. dev.s")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_subset = args.data_subset
    print(f'data_subset={data_subset}')
    file_name_fragment = args.file_name_fragment
    print(f'file_name_fragment={file_name_fragment}')
    file_name_fragment = args.file_name_fragment
    print(f'file_name_fragment={file_name_fragment}')
    num_thresholds = args.num_thresholds
    print(f'num_thresholds={num_thresholds}')
    min_threshold = args.min_threshold
    print(f'min_threshold={min_threshold}')
    max_threshold = args.max_threshold
    print(f'max_threshold={max_threshold}')

    data_ts_file = os.path.join(output_directory, f'data_ts_{data_subset}_{file_name_fragment}.pt')
    data_ts = torch.load(data_ts_file)
    print(f'time {time.time()-code_start_time:.3f}, computing unbinarized means and flip rates...')
    # data_ts /= data_ts.abs().max(dim=-1).values# Divide the whole time series by the largest absolute value so that the FC stays the same but the means and mean products are <= 1.
    state_mean = data_ts.mean(dim=-1, keepdim=False)
    print( f'state mean', state_mean.size() )
    state_mean_product = torch.matmul( data_ts, data_ts.transpose(dim0=-2, dim1=-1) )/data_ts.size(dim=-1)
    print( f'state mean', state_mean_product.size() )
    data_cov = isingmodellight.get_cov(state_mean=state_mean, state_product_mean=state_mean_product)
    data_fc = isingmodellight.get_fc(state_mean=state_mean, state_product_mean=state_mean_product, epsilon=0)
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    threshold_choices = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device)
    num_choices = threshold_choices.numel()
    quantiles = torch.tensor(data=[0.025, 0.5, 0.975], dtype=float_type, device=device)
    num_quantiles = len(quantiles)
    flip_rate_quantiles = torch.zeros( size=(num_choices,num_quantiles), dtype=float_type, device=device )
    cov_rmse_quantiles = torch.zeros( size=(num_choices,num_quantiles), dtype=float_type, device=device )
    fc_corr_quantiles = torch.zeros( size=(num_choices,num_quantiles), dtype=float_type, device=device )
    for choice_index in range(num_choices):
        choice = threshold_choices[choice_index]
        print(f'threshold {choice_index+1} of {num_choices}: {choice:.3g}')
        threshold = data_ts_mean + choice*data_ts_std
        flip_rate, cov_rmse, fc_corr = get_binarized_info( data_ts=data_ts, threshold=threshold, data_cov=data_cov, data_fc=data_fc )
        flip_rate_quantiles[choice_index,:] = torch.quantile(input=flip_rate, q=quantiles)
        cov_rmse_quantiles[choice_index,:] = torch.quantile(input=cov_rmse, q=quantiles)
        fc_corr_quantiles[choice_index,:] = torch.quantile(input=fc_corr, q=quantiles)
    summary_file = os.path.join(output_directory, f'flip_rate_and_fc_rmse_ci_{data_subset}_{file_name_fragment}_choices_{num_choices}_from_{min_threshold:.3g}_to_{max_threshold:.3g}.pt')
    torch.save(obj=(threshold_choices, flip_rate_quantiles, cov_rmse_quantiles, fc_corr_quantiles), f=summary_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {summary_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')