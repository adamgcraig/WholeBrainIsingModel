import os
import torch
import time
import argparse

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
    parser.add_argument("-d", "--threshold_type", type=str, default='as_is', help="'as_is' -> no binarization, 'median' -> median, 'quantile' -> quantile, 'mean' -> mean +/- std.dev.")
    parser.add_argument("-e", "--threshold", type=float, default=0, help="ignored for as_is or median, quantile for quantile, or std.dev.s above the mean")
    parser.add_argument("-f", "--weighted", action='store_true', default=False, help="Set this flag to use the quasi-entropy of each time point to weight it.")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_subset = args.data_subset
    print(f'data_subset={data_subset}')
    threshold_type = args.threshold_type
    print(f'threshold_type={threshold_type}')
    threshold = args.threshold
    print(f'threshold={threshold}')
    weighted = args.weighted
    print(f'weighted={weighted}')

    # We need to start with the as-is, unbinarized time series data.
    data_ts_file = os.path.join(output_directory, f'data_ts_{data_subset}_as_is.pt')
    data_ts = torch.load(data_ts_file)
    print( f'time {time.time()-code_start_time:.3f}, loaded {data_ts_file}, size', data_ts.size() )
    if threshold_type == 'as_is':
        # Divide the whole time series by the largest absolute value so that the FC stays the same but the means and mean products are <= 1.
        data_ts /= data_ts.abs().max(dim=-1, keepdim=True).values
        threshold_str = 'as_is'
        print(f'time {time.time()-code_start_time:.3f}, rescaled time series to have unit maximum amplitude')
    else:
        if threshold_type == 'median':
            threshold_for_ts = torch.median(data_ts, dim=-1, keepdim=True).values
            threshold_str = 'median'
        elif threshold_type == 'quantile':
            threshold_for_ts = torch.quantile(data_ts, q=threshold, dim=-1, keepdim=True)
            threshold_str = f'quantile_{threshold:.3g}'
        elif threshold_type == 'mean':
            ts_std, ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
            threshold_for_ts = ts_mean + threshold*ts_std
            threshold_str = f'mean_std_{threshold:.3g}'
        else:
            print(f'unrecognized threshold type {threshold_type}')
            exit(1)
        data_ts_bool = data_ts > threshold_for_ts
        data_ts = ( 2*data_ts_bool.float() - 1 )
        # Add in a small offset to avoid taking logarithms of 0.
        # The next smallest probability is 1/num_nodes.
        # So long as num_nodes << 10**10, 1/num_nodes + 1e-10 should be very close to 1/num_nodes.
        if weighted:
            epsilon = 1e-10
            prob_up = torch.count_nonzero(data_ts_bool, dim=-2)/data_ts_bool.size(dim=-2)
            prob_down = 1 - prob_up
            quasi_entropy = prob_up*torch.log(prob_up+epsilon) + prob_down*torch.log(prob_down+epsilon)
            data_ts *= quasi_entropy.unsqueeze(dim=-2)
            threshold_str = f'weighted_{threshold_str}'
        print(f'time {time.time()-code_start_time:.3f}, binarized time series')

    mean_state = torch.mean(data_ts, dim=-1, keepdim=False)
    mean_state_file = os.path.join(output_directory, f'mean_state_{data_subset}_{threshold_str}.pt')
    torch.save(obj=mean_state, f=mean_state_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {mean_state_file}')

    mean_state_product = torch.matmul( data_ts, torch.transpose(data_ts, dim0=-2, dim1=-1) )/data_ts.size(dim=-1)
    mean_state_product_file = os.path.join(output_directory, f'mean_state_product_{data_subset}_{threshold_str}.pt')
    torch.save(obj=mean_state_product, f=mean_state_product_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {mean_state_product_file}')

    print(f'time {time.time()-code_start_time:.3f}, done')