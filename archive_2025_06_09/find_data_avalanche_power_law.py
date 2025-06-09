import os
import torch
import time
import argparse

def get_avalanche_counts(data_ts:torch.Tensor, threshold:torch.Tensor):
    binarized_ts = data_ts > threshold
    node_dim = -2
    num_nodes = binarized_ts.size(dim=node_dim)
    flip_count_at_step = torch.count_nonzero(binarized_ts[:,:,:,1:] != binarized_ts[:,:,:,:-1], dim=node_dim).flatten()
    count_choices = torch.arange(start=0, end=num_nodes+1, dtype=torch.int, device=flip_count_at_step.device)
    return torch.count_nonzero(count_choices[:,None] == flip_count_at_step[None,:], dim=-1)

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Load and threshold unbinarized time series data. Make and save a Tensor counts such that counts[x] is the number of times x nodes flip in a single step.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-c", "--data_subset", type=str, default='all', help="'training', 'validation', 'testing', or 'all'")
    parser.add_argument("-d", "--file_name_fragment", type=str, default='as_is', help="part of the output file name between mean_state_[data_subset]_ or mean_state_product_[data_subset]_ and .pt")
    parser.add_argument("-f", "--min_threshold", type=float, default=0, help="minimum threshold in std. dev.s")
    parser.add_argument("-g", "--max_threshold", type=float, default=2, help="minimum threshold in std. dev.s")
    parser.add_argument("-i", "--num_thresholds", type=int, default=21, help="number of thresholds to try")
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
    min_threshold = args.min_threshold
    print(f'min_threshold={min_threshold}')
    max_threshold = args.max_threshold
    print(f'max_threshold={max_threshold}')
    num_thresholds = args.num_thresholds
    print(f'num_thresholds={num_thresholds}')

    data_ts_file = os.path.join(output_directory, f'data_ts_{data_subset}_{file_name_fragment}.pt')
    data_ts = torch.load(data_ts_file)
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    num_nodes = data_ts.size(dim=-2)
    thresholds = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device)
    avalanche_sizes = torch.arange(start=0, end=num_nodes+1, step=1, dtype=float_type, device=device)# 0 through 360, inclusive
    num_sizes = avalanche_sizes.numel()
    avalanche_counts = torch.zeros( size=(num_thresholds, num_sizes), dtype=int_type, device=data_ts.device )
    for threshold_index in range(num_thresholds):
        threshold = thresholds[threshold_index]
        print(f'threshold {threshold_index+1} of {num_thresholds}: {threshold:.3g}')
        threshold_for_region = data_ts_mean + threshold*data_ts_std
        avalanche_counts[threshold_index,:] = get_avalanche_counts(data_ts=data_ts, threshold=threshold_for_region)
    avalanche_counts_file = os.path.join(output_directory, f'avalanche_counts_{data_subset}_choices_{num_thresholds}.pt')
    torch.save(obj=avalanche_counts, f=avalanche_counts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {avalanche_counts_file}')
    log_sizes = avalanche_sizes[1:].log()
    count_is_nonzero = avalanche_counts[:,1:] > 0
    num_nonzero_counts = torch.count_nonzero(count_is_nonzero, dim=-1)
    log_freqs = ( avalanche_counts[:,1:]/avalanche_counts.sum(dim=-1, keepdim=True) ).log()
    A = torch.ones( size=(num_nodes,2), dtype=float_type, device=device )
    B = torch.zeros( size=(num_nodes,1), dtype=float_type, device=device )
    slope = torch.zeros_like(thresholds)
    intercept = torch.zeros_like(thresholds)
    r_squared = torch.zeros_like(thresholds)
    for threshold_index in range(num_thresholds):
        current_count_nonzero = count_is_nonzero[threshold_index,:]
        A[:,0] = log_sizes
        B[:,0] = log_freqs[threshold_index,:]
        A_nz = A[current_count_nonzero,:]
        B_nz = B[current_count_nonzero,:]
        X = torch.linalg.lstsq(A_nz, B_nz).solution
        B_pred = torch.matmul(A_nz,X)
        B_err = torch.nn.functional.mse_loss(input=B_pred, target=B_nz)
        B_var = B_nz.var()
        slope[threshold_index] = X[0,0]
        intercept[threshold_index] = X[1,0]
        r_squared[threshold_index] = 1 - B_err/B_var
        print(f'threshold {thresholds[threshold_index]:.3g}, num points: {num_nonzero_counts[threshold_index]}, best fit: log(prob)={slope[threshold_index]:.3g}*log(size)+{intercept[threshold_index]:.3g}, R^2={r_squared[threshold_index]:.3g}')
    log_log_fits_file = os.path.join(output_directory, f'avalanche_freq_prob_log_log_fits_{data_subset}_selected_choices_{num_thresholds}.pt')
    torch.save( obj=(thresholds, slope, intercept, r_squared, num_nonzero_counts), f=log_log_fits_file )
    print(f'time {time.time()-code_start_time:.3f}, saved {log_log_fits_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')