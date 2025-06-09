import os
import torch
import time
import argparse
import hcpdatautils as hcp
from isingutilsslow import prep_individual_data_ts
from isingutilsslow import get_data_means_and_covs_slower
from isingutilsslow import get_fc
from isingutilsslow import get_rmse
from isingutilsslow import get_triu_rmse

parser = argparse.ArgumentParser(description="How long a window do we need for the FC to converge to that of the full data time series?")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-t", "--threshold", type=str, default='median', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, or the string 'median', or the string 'none'.")
args = parser.parse_args()
# print('getting arguments...')
data_directory = args.data_directory
# print(f'data_directory={data_directory}')
output_directory = args.output_directory
# print(f'output_directory={output_directory}')
threshold_str = args.threshold
if threshold_str == 'median':
    threshold = threshold_str
elif threshold_str == 'none':
    threshold = threshold_str
else:
    threshold = float(threshold_str)
# print(f'threshold={threshold_str}')

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
    # print( 'data_ts size', data_ts.size() )
    num_subjects = data_ts.size(dim=1)
    num_time_points = data_ts.size(dim=-1)
    total_mean, total_cov = get_data_means_and_covs_slower(data_ts=data_ts, window=num_time_points)
    data_ts = data_ts[0,:,:,:]
    total_fc = get_fc(s_mean=total_mean[:,:,:,0], s_cov=total_cov[:,:,:,:,0])[0,:,:,:]
    print( 'total_fc size', total_fc.size() )
    # print(total_fc)
    total_mean = total_mean[0,:,:,0]
    print( 'total_mean size', total_mean.size() )
    # print(total_mean)
    total_cov = total_cov[0,:,:,:,0]
    print( 'total_cov size', total_cov.size() )
    # print(total_cov)
    # RMSE is never negative, so we can use -1 as a placeholder and check for it later.
    state_sum = torch.zeros_like(data_ts[:,:,0])
    print( 'state_sum size', state_sum.size() )
    cross_sum = state_sum[:,:,None] * state_sum[:,None,:]
    print( 'cross_sum size', cross_sum.size() )
    epsilon = 10e-10# Add this to the denominator of the FC to prevent NaNs.
    # For each RMSE, for each window size, save the sum, sum of squares, min, and max.
    sum_mean_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    sum_square_mean_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    min_mean_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    max_mean_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    sum_cov_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    sum_square_cov_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    min_cov_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    max_cov_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    sum_fc_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    sum_square_fc_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    min_fc_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    max_fc_rmse = torch.zeros( (num_time_points,), dtype=float_type, device=device )
    print('subject,offset,windowlength,meanrmse,covrmse,fcrmse,time')
    # Grow the window from the selected starting point.
    # When we reach the end, wrap around to the start.
    # For each subject, track the mean of each node and the mean product of each pair of nodes.
    # We can then use these to compute the FC of each node pair. 
    for start_time in range(num_time_points):
        state_sum.zero_()
        cross_sum.zero_()
        for window_length_index in range(num_time_points):
            current_time = (start_time + window_length_index) % num_time_points
            current_state = data_ts[:,:,current_time]
            state_sum += current_state
            cross_sum += current_state[:,:,None] * current_state[:,None,:]
            window_length = window_length_index+1
            state_mean = state_sum/window_length
            cross_mean = cross_sum/window_length
            # Center the covariance by taking mean(s * s^T) - mean(s) * mean(s)^T.
            centered_cov = cross_mean - state_mean[:,:,None] * state_mean[:,None,:]
            # The Pearson correlation is then the centered covariance rescaled by the outer product of the standard deviation with itself.
            # variance = mean(state^2) - mean(state)^2. The square of the state is always 1.
            state_std = torch.sqrt( torch.diagonal(input=cross_mean, offset=0, dim1=-2, dim2=-1) - state_mean.square() )
            std_product = state_std[:,:,None] * state_std[:,None,:]#s_std * s_std.transpose(dim0=self.node_dim0, dim1=self.node_dim1)
            fc = centered_cov/(std_product + epsilon)
            # Now take the RMSEs vs the mean, uncentered covariance, and FC of the full time series.
            mean_rmse = get_rmse(state_mean, total_mean)
            cov_rmse = get_triu_rmse(cross_mean, total_cov)
            fc_rmse = get_triu_rmse(fc, total_fc)
            sum_mean_rmse[window_length_index] += mean_rmse.sum()
            sum_square_mean_rmse[window_length_index] += mean_rmse.square().sum()
            min_mean_rmse[window_length_index] = torch.minimum( min_mean_rmse[window_length_index], mean_rmse.min() )
            max_mean_rmse[window_length_index] = torch.maximum( max_mean_rmse[window_length_index], mean_rmse.max() )
            sum_cov_rmse[window_length_index] += cov_rmse.sum()
            sum_square_cov_rmse[window_length_index] += cov_rmse.square().sum()
            min_cov_rmse[window_length_index] = torch.minimum( min_cov_rmse[window_length_index], cov_rmse.min() )
            max_cov_rmse[window_length_index] = torch.maximum( max_cov_rmse[window_length_index], cov_rmse.max() )
            sum_fc_rmse[window_length_index] += fc_rmse.sum()
            sum_square_fc_rmse[window_length_index] += fc_rmse.square().sum()
            min_fc_rmse[window_length_index] = torch.minimum( min_fc_rmse[window_length_index], fc_rmse.min() )
            max_fc_rmse[window_length_index] = torch.maximum( max_fc_rmse[window_length_index], fc_rmse.max() )
            # for subject in range(num_subjects):
            #     print(f'{subject},{start_time},{window_length},{mean_rmse[subject]},{cov_rmse[subject]},{fc_rmse[subject]},{time.time()-code_start_time}')
        print(f'offset {start_time}, time {time.time()-code_start_time:.3f}')
    num_samples_per_window_length = num_time_points * num_subjects
    file_name_suffix = f'window_convergence_test_subjects_{data_subset}_threshold_{threshold_str}.pt'
    mean_mean_rmse = sum_mean_rmse/num_samples_per_window_length
    std_mean_rmse = torch.sqrt( sum_square_mean_rmse/num_samples_per_window_length - mean_mean_rmse.square() )
    stacked_mean_rmse = torch.stack( (mean_mean_rmse, std_mean_rmse, min_mean_rmse, max_mean_rmse), dim=0 )
    mean_cov_rmse = sum_cov_rmse/num_samples_per_window_length
    std_cov_rmse = torch.sqrt( sum_square_cov_rmse/num_samples_per_window_length - mean_cov_rmse.square() )
    stacked_cov_rmse = torch.stack( (mean_cov_rmse, std_cov_rmse, min_cov_rmse, max_cov_rmse), dim=0 )
    mean_fc_rmse = sum_fc_rmse/num_samples_per_window_length
    std_fc_rmse = torch.sqrt( sum_square_fc_rmse/num_samples_per_window_length - mean_fc_rmse.square() )
    stacked_fc_rmse = torch.stack( (mean_fc_rmse, std_fc_rmse, min_fc_rmse, max_fc_rmse), dim=0 )
    stacked_results_rmse = torch.stack( (stacked_mean_rmse, stacked_cov_rmse, stacked_fc_rmse), dim=0 )
    summary_file = os.path.join(output_directory, f'window_convergence_test_mean_cov_fc_mean_std_min_max_rmse_subjects_{data_subset}_threshold_{threshold_str}.pt')
    torch.save(obj=stacked_results_rmse, f=summary_file)
print(f'done, time {time.time() - code_start_time:.3f}')