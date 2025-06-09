import os
import torch
import time
import argparse
import hcpdatautilsnopandas as hcp
import isingmodellight

def get_binarized_info(data_ts:torch.Tensor, threshold:torch.Tensor, data_fc:torch.Tensor):
    binarized_ts = 2*(data_ts > threshold).float() - 1
    flip_rate = (binarized_ts[:,:,:,1:] != binarized_ts[:,:,:,:-1]).float().mean(dim=-1)
    state_mean = binarized_ts.mean(dim=-1)
    state_product_mean = torch.matmul( binarized_ts, binarized_ts.transpose(dim0=-2, dim1=-1) )/binarized_ts.size(dim=-1)
    fc_binary = isingmodellight.get_fc_binary(state_mean=state_mean, state_product_mean=state_product_mean)
    fc_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=fc_binary, mat2=data_fc)
    fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=fc_binary, mat2=data_fc)
    return flip_rate, fc_rmse, fc_corr

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
    parser.add_argument("-f", "--min_threshold", type=float, default=-5, help="minimum threshold in std. dev.s")
    parser.add_argument("-g", "--max_threshold", type=float, default=5, help="minimum threshold in std. dev.s")
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
    print( f'time {time.time()-code_start_time:.3f}, saved {data_ts_file}, size', data_ts.size() )
    print(f'time {time.time()-code_start_time:.3f}, computing unbinarized means and flip rates...')
    # data_ts /= data_ts.abs().max(dim=-1).values# Divide the whole time series by the largest absolute value so that the FC stays the same but the means and mean products are <= 1.
    state_mean = data_ts.mean(dim=-1, keepdim=False)
    print( f'state mean', state_mean.size() )
    state_mean_product = torch.matmul( data_ts, data_ts.transpose(dim0=-2, dim1=-1) )/data_ts.size(dim=-1)
    print( f'state mean', state_mean_product.size() )
    data_fc = isingmodellight.get_fc(state_mean=state_mean, state_product_mean=state_mean_product)
    flip_rate, fc_rmse, fc_corr = get_binarized_info( data_ts=data_ts, threshold=data_ts.median(dim=-1, keepdim=True).values, data_fc=data_fc )
    flip_rate_file = os.path.join(output_directory, f'flip_rate_{data_subset}_{file_name_fragment}_binarized_median.pt')
    torch.save(obj=flip_rate, f=flip_rate_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file}')
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{data_subset}_{file_name_fragment}_binarized_median_vs_original.pt')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file}')
    fc_corr_file = os.path.join(output_directory, f'fc_corr_{data_subset}_{file_name_fragment}_binarized_median_vs_original.pt')
    torch.save(obj=fc_corr, f=fc_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_corr_file}')
    data_ts_std = data_ts.std(dim=-1, keepdim=True)
    threshold_choices = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device)
    num_choices = threshold_choices.numel()
    std_flip_rates = torch.zeros( size=(num_choices,), dtype=float_type, device=device )
    mean_flip_rates = torch.zeros( size=(num_choices,), dtype=float_type, device=device )
    std_fc_rmse = torch.zeros( size=(num_choices,), dtype=float_type, device=device )
    mean_fc_rmse = torch.zeros( size=(num_choices,), dtype=float_type, device=device )
    std_fc_corr = torch.zeros( size=(num_choices,), dtype=float_type, device=device )
    mean_fc_corr = torch.zeros( size=(num_choices,), dtype=float_type, device=device )
    for choice_index in range(num_choices):
        choice = threshold_choices[choice_index]
        print(f'threshold {choice_index+1} of {num_choices}: {choice:.3g}')
        flip_rate, fc_rmse, fc_corr = get_binarized_info( data_ts=data_ts, threshold=choice*data_ts_std, data_fc=data_fc )
        # flip_rate_file = os.path.join(output_directory, f'flip_rate_{data_subset}_{file_name_fragment}_binarized_std_times_{choice}.pt')
        # torch.save(obj=flip_rate, f=flip_rate_file)
        # print(f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file}')
        # fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{data_subset}_{file_name_fragment}_binarized_std_times_{choice}_vs_original.pt')
        # torch.save(obj=fc_rmse, f=fc_rmse_file)
        # print(f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file}')
        std_flip_rates[choice_index], mean_flip_rates[choice_index] = torch.std_mean(flip_rate)
        std_fc_rmse[choice_index], mean_fc_rmse[choice_index] = torch.std_mean(fc_rmse)
        std_fc_corr[choice_index], mean_fc_corr[choice_index] = torch.std_mean(fc_corr)
    summary_file = os.path.join(output_directory, f'flip_rate_and_fc_rmse_summary_{data_subset}_{file_name_fragment}_choices_{num_choices}_from_{min_threshold:.3g}_to_{max_threshold:.3g}.pt')
    torch.save(obj=(threshold_choices, std_flip_rates, mean_flip_rates, std_fc_rmse, mean_fc_rmse, std_fc_corr, mean_fc_corr), f=summary_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {summary_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')