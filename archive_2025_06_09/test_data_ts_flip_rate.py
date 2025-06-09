import os
import torch
import time
import argparse

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the number of times each node flips.")
parser.add_argument("-f", "--file_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we read and write files")
parser.add_argument("-m", "--ts_file_suffix", type=str, default='binary_data_ts_all', help="part of the binarized data time series file name, data time series should have shape num_subjects x num_nodes x num_time_points")
parser.add_argument("-t", "--time_series_per_subject", type=int, default=4, help="number of time series into which to break the combined time series")
args = parser.parse_args()
print('getting arguments...')
file_directory = args.file_directory
print(f'file_directory={file_directory}')
ts_file_suffix = args.ts_file_suffix
print(f'ts_file_suffix={ts_file_suffix}')
time_series_per_subject = args.time_series_per_subject
print(f'time_series_per_subject={time_series_per_subject}')

data_ts_file = os.path.join(file_directory, f'{ts_file_suffix}.pt')
data_ts = torch.load(data_ts_file)
print(f'time {time.time() - code_start_time:.3f},\t loaded {data_ts_file}')
print( 'data_ts size', data_ts.size() )
num_time_points = data_ts.size(dim=-1)
time_points_per_ts = num_time_points//time_series_per_subject
data_ts = data_ts.unflatten( dim=-1, sizes=(time_series_per_subject, time_points_per_ts) )
print( 'unflattened to', data_ts.size() )

# Check that the mean states are all 0.
mean_state = data_ts.sum(dim=-1)
num_nonzero_means = torch.count_nonzero(mean_state)
print(f'number of time series with non-0 means {num_nonzero_means}')

# Sum over the time points in each time series, then combine the sums of individual time series of the same subject.
flip_rate = torch.abs(data_ts[:,:,:,1:] - data_ts[:,:,:,:-1]).sum(dim=-1).sum(dim=-1)/( 2.0*time_series_per_subject*(time_points_per_ts-1) )
flip_rate_file = os.path.join(file_directory, f'flip_rate_{ts_file_suffix}.pt')
torch.save(obj=flip_rate, f=flip_rate_file)
print(f'saved {flip_rate_file}')
print(f'time {time.time() - code_start_time:.3f},\t done')