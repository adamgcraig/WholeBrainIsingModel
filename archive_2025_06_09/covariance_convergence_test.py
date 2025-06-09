import os
import torch
import time
import argparse
from isingutilsslow import prep_individual_data_ts
from isingutilsslow import get_data_means_and_covs_slower
from isingutilsslow import get_fc
from isingutilsslow import get_rmse
from isingutilsslow import get_triu_rmse

parser = argparse.ArgumentParser(description="How long a window do we need for the FC to converge to that of the full data time series?")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-t", "--threshold", type=str, default='0.1', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, or the string 'median', or the string 'none'.")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
threshold_str = args.threshold
if threshold_str == 'median':
    threshold = threshold_str
else:
    threshold = float(threshold_str)
print(f'threshold={threshold_str}')

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    data_subset = 'training'
    data_ts_file = os.path.join(data_directory, f'data_ts_{data_subset}.pt')
    # Swap dimensions so that nodes are in the second-to-last dimension, time points in the last.
    data_ts = prep_individual_data_ts( data_ts=torch.load(f=data_ts_file), num_reps=1, threshold=threshold )
    print(f'loaded all time series, time {time.time() - code_start_time:.3f}')
    print( 'data_ts size', data_ts.size() )
    num_subjects = data_ts.size(dim=1)
    num_time_points = data_ts.size(dim=-1)
    total_mean, total_cov = get_data_means_and_covs_slower(data_ts=data_ts, window=num_time_points)
    total_fc = get_fc(s_mean=total_mean, s_cov=total_cov)
    print( 'total mean size', total_mean.size() )
    print( 'total cov size', total_cov.size() )
    # To keep memory usage manageable, summarize the distribution of RMSEs via the min, median, max, and 95% CI.
    quantiles = [0.0, 0.025, 0.5, 0.975, 1.0]
    num_quantiles = len(quantiles)
    # RMSE is never negative, so we can use -1 as a placeholder and check for it later.
    mean_rmse_quantiles = torch.ones( (num_quantiles, num_time_points), dtype=float_type, device=device )
    cov_rmse_quantiles = torch.ones( (num_quantiles, num_time_points), dtype=float_type, device=device )
    fc_rmse_quantiles = torch.ones( (num_quantiles, num_time_points), dtype=float_type, device=device )
    for window_length_index in range(num_time_points):
        window_length = window_length_index+1
        num_windows = num_time_points // window_length
        current_mean_rmse = torch.zeros( (1, num_subjects, num_windows, num_time_points), dtype=float_type, device=device )
        current_cov_rmse = torch.zeros( (1, num_subjects, num_windows, num_time_points), dtype=float_type, device=device )
        current_fc_rmse = torch.zeros( (1, num_subjects, num_windows, num_time_points), dtype=float_type, device=device )
        for offset in range(num_time_points):
            shifted_ts = torch.roll(input=data_ts, shifts=offset, dims=-1)
            window_means, window_covs = get_data_means_and_covs_slower(data_ts=shifted_ts, window=window_length)
            window_fcs = get_fc(s_mean=window_means, s_cov=window_covs)
            num_windows = window_means.size(dim=-1)
            current_mean_rmse[:,:,:,offset] = get_rmse(total_mean, window_means)
            current_cov_rmse[:,:,:,offset] = get_triu_rmse(total_cov, window_covs)
            current_fc_rmse[:,:,:,offset] = get_triu_rmse(total_fc, window_fcs)
        print( f'window length {window_length_index}, time {time.time() - code_start_time:.3f}, mean mean RMSE {current_mean_rmse.mean():.3g}, mean cov RMSE {current_cov_rmse.mean():.3g}, mean FC RMSE {current_fc_rmse.mean():.3g}, current mean RMSE size', current_mean_rmse.size() )
        mean_rmse_quantiles[:,window_length_index] = torch.quantile(input=current_mean_rmse, q=quantiles)
        cov_rmse_quantiles[:,window_length_index] = torch.quantile(input=current_mean_rmse, q=quantiles)
        fc_rmse_quantiles[:,window_length_index] = torch.quantile(input=current_mean_rmse, q=quantiles)
    file_name_suffix = f'window_convergence_test_subjects_{data_subset}_threshold_{threshold_str}.pt'
    mean_rmse_file = os.path.join( output_directory, f'mean_rmse_quantiles_{file_name_suffix}' )
    torch.save(obj=mean_rmse_quantiles, f=mean_rmse_file)
    print(f'saved file {mean_rmse_file}, time {time.time() - code_start_time:.3f}')
    cov_rmse_file = os.path.join( output_directory, f'cov_rmse_quantiles_{file_name_suffix}' )
    torch.save(obj=cov_rmse_quantiles, f=cov_rmse_file)
    print(f'saved file {cov_rmse_file}, time {time.time() - code_start_time:.3f}')
    fc_rmse_file = os.path.join( output_directory, f'fc_rmse_quantiles_{file_name_suffix}' )
    torch.save(obj=fc_rmse_quantiles, f=fc_rmse_file)
    print(f'saved file {fc_rmse_file}, time {time.time() - code_start_time:.3f}')
    print(f'done, time {time.time() - code_start_time:.3f}')