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

    data_ts_file = os.path.join(output_directory, f'data_ts_{data_subset}_{file_name_fragment}.pt')
    data_ts = torch.load(data_ts_file)
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    num_nodes = data_ts.size(dim=-2)
    threshold_choices = torch.tensor(data=[0, 0.1, 0.5, 1, 2], dtype=float_type, device=device)
    num_choices = threshold_choices.numel()
    avalanche_counts = torch.zeros( size=(num_choices, num_nodes+1), dtype=int_type, device=data_ts.device )
    for choice_index in range(num_choices):
        choice = threshold_choices[choice_index]
        print(f'threshold {choice_index+1} of {num_choices}: {choice:.3g}')
        threshold = data_ts_mean + choice*data_ts_std
        avalanche_counts[choice_index,:] = get_avalanche_counts(data_ts=data_ts, threshold=threshold)
    avalanche_counts_file = os.path.join(output_directory, f'avalanche_counts_{data_subset}_selected_choices_{num_choices}.pt')
    torch.save(obj=avalanche_counts, f=avalanche_counts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {avalanche_counts_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')