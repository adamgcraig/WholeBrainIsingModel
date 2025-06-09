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
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_as_is', help="part of the data time series file between data_ts_ and .pt")
    parser.add_argument("-d", "--window_increment", type=int, default=100, help="amount by which to increment window length")
    parser.add_argument("-e", "--min_window", type=int, default=100, help="first window length to test")
    parser.add_argument("-f", "--max_window", type=int, default=1200, help="last window length to test")
    parser.add_argument("-g", "--threshold", type=float, default=0.0, help="threshold at which to binarize time series in SD above the mean")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    window_increment = args.window_increment
    print(f'window_increment={window_increment}')
    min_window = args.min_window
    print(f'min_window={min_window}')
    max_window = args.max_window
    print(f'max_window={max_window}')
    threshold = args.threshold
    print(f'threshold={threshold}')

    def matmul_by_slice(ts:torch.Tensor, out:torch.Tensor):
        out.zero_()
        slice_product = torch.zeros_like(out)
        for ts_at_step in ts.split(split_size=1, dim=-1):
            torch.matmul( ts_at_step, ts_at_step.transpose(dim0=-2, dim1=-1), out=slice_product )
            out += slice_product
        # return out

    data_ts_file = os.path.join(data_directory, f'data_ts_{data_file_name_part}.pt')
    def load_and_binarize_ts(data_ts_file:str, threshold:float):
        data_ts = torch.load(data_ts_file, weights_only=False)
        ts_std, ts_mean = torch.std_mean(input=data_ts, dim=-1, keepdim=True)
        data_ts -= ts_mean
        data_ts /= ts_std
        data_ts = (data_ts >= threshold).float()
        data_ts *= 2.0
        data_ts -= 1.0
        # data_ts is originally scans x subjects x nodes x steps.
        # Permute it to subjects x nodes x scans x steps.
        data_ts = data_ts.permute( dims=(1, 2, 0, 3) )
        # Then flatten to subjects x nodes x scans*steps.
        data_ts = data_ts.flatten(start_dim=-2, end_dim=-1)
        return data_ts
    data_ts = load_and_binarize_ts(data_ts_file=data_ts_file, threshold=threshold)
    num_subjects, num_nodes, num_steps = data_ts.size()
    window_lengths = torch.arange(start=min_window, end=max_window+1, step=window_increment, dtype=int_type, device=data_ts.device)
    num_window_lengths = window_lengths.numel()
    fc_std = torch.zeros( size=(num_window_lengths, num_subjects, num_nodes, num_nodes), dtype=data_ts.dtype, device=data_ts.device )
    for window_length_index in range(num_window_lengths):
        window_length = window_lengths[window_length_index]
        num_windows = num_steps//window_length
        num_steps_in_windows = num_windows*window_length
        # data_ts is subjects x nodes x scans*steps.
        # unflatten gives us subjects x nodes x windows x window_length.
        # permute gives us subjects x windows x nodes x window_length.
        # The FC is then subjects x windows x nodes x nodes.
        # We then take the FC SD over all windows in all scans to get subjects x nodes x nodes. 
        data_ts_windows = torch.unflatten( input=data_ts[:,:,:num_steps_in_windows], dim=-1, sizes=(num_windows, window_length) )
        data_ts_windows.transpose_(dim0=1, dim1=2)
        fc_windows = torch.zeros( size=(num_subjects, num_windows, num_nodes, num_nodes), dtype=data_ts_windows.dtype, device=data_ts_windows.device )
        # fc_windows = torch.matmul( input=data_ts_windows, other=data_ts_windows.transpose(dim0=-2, dim1=-1) )
        matmul_by_slice(ts=data_ts_windows, out=fc_windows)
        window_std, window_mean = torch.std_mean(data_ts_windows, dim=-1, keepdim=True)
        fc_windows /= window_length
        fc_windows -= ( window_mean * window_mean.transpose(dim0=-2, dim1=-1) )
        fc_windows /= ( window_std * window_std.transpose(dim0=-2, dim1=-1) )
        torch.std(input=fc_windows, dim=1, keepdim=False, out=fc_std[window_length_index,:,:,:])
        this_fc_std = fc_std[window_length_index,:,:,:]
        fc_std_min = this_fc_std.min()
        fc_std_mean = this_fc_std.mean()
        fc_std_max = this_fc_std.max()
        print(f'time {time.time()-code_start_time:.3f}, window length {window_length} ({window_length_index+1} of {num_window_lengths}) min {fc_std_min:.3g}, mean {fc_std_mean:.3g}, max {fc_std_max:.3g}')
    fc_std_file = os.path.join(output_directory, f'fc_std_{data_file_name_part}_threshold_{threshold:.3g}_window_min_{min_window}_max_{max_window}_inc_{window_increment}.pt')
    torch.save(obj=fc_std, f=fc_std_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_std_file}')
print('done')