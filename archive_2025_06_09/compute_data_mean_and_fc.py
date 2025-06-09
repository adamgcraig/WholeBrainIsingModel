import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingmodel

parser = argparse.ArgumentParser(description="Compute the means of the states and state products and the FC of the data time series.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='all', help="the subset of subjects over which to search for unique states")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    subject_list = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_list)

    print(f'allocating memory for time series of {num_subjects} subjects, time {time.time() - code_start_time:.3f}')
    num_nodes = hcp.num_brain_areas
    num_steps = hcp.time_series_per_subject * hcp.num_time_points
    binary_data_ts = torch.zeros( size=(num_subjects, num_nodes, num_steps), dtype=float_type, device=device )

    print(f'loading time series data, time {time.time() - code_start_time:.3f}')
    for subject in range(num_subjects):
        subject_id = subject_list[subject]
        # We initially load the data time series with dimensions time series x step x node,
        # but we want to work with them as time series x node x step.
        data_ts = hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device).transpose(dim0=-2, dim1=-1)
        # Once we have binarized each time series, we want to concatenate together the time series for a single subject.
        binary_data_ts[subject,:,:] = isingmodel.binarize_data_ts(data_ts).transpose(dim0=0, dim1=1).flatten(start_dim=-2, end_dim=-1)
    
    print(f'computing means, time {time.time() - code_start_time:.3f}')
    individual_state_mean, individual_product_mean = isingmodel.get_time_series_mean(binary_data_ts)
    print(f'individual state mean min {individual_state_mean.min():.3g}, mean {individual_state_mean.mean():.3g}, max {individual_state_mean.max():.3g}')
    print(f'individual state product mean min {individual_product_mean.min():.3g}, mean {individual_product_mean.mean():.3g}, max {individual_product_mean.max():.3g}')
    print(f'time {time.time() - code_start_time:.3f}')
    individual_state_mean_file = os.path.join(output_directory, f'mean_state_individual_{data_subset}.pt')
    torch.save(obj=individual_state_mean, f=individual_state_mean_file)
    print(f'saved {individual_state_mean_file}, time {time.time() - code_start_time:.3f}')
    individual_product_mean_file = os.path.join(output_directory, f'mean_state_product_individual_{data_subset}.pt')
    torch.save(obj=individual_product_mean, f=individual_product_mean_file)
    print(f'saved {individual_product_mean_file}, time {time.time() - code_start_time:.3f}')
    
    # Since all time series are the same length, we can get the means for the concatenated group time series just by taking the mean of the means of individual time series.
    group_state_mean = individual_state_mean.mean(dim=0, keepdim=True)
    group_product_mean = individual_product_mean.mean(dim=0, keepdim=True)
    print(f'group state mean min {group_state_mean.min():.3g}, mean {group_state_mean.mean():.3g}, max {group_state_mean.max():.3g}')
    print(f'group state product mean min {group_product_mean.min():.3g}, mean {group_product_mean.mean():.3g}, max {group_product_mean.max():.3g}')
    print(f'time {time.time() - code_start_time:.3f}')
    group_state_mean_file = os.path.join(output_directory, f'mean_state_group_{data_subset}.pt')
    torch.save(obj=group_state_mean, f=group_state_mean_file)
    print(f'saved {group_state_mean_file}, time {time.time() - code_start_time:.3f}')
    group_product_mean_file = os.path.join(output_directory, f'mean_state_product_group_{data_subset}.pt')
    torch.save(obj=group_product_mean, f=group_product_mean_file)
    print(f'saved {group_product_mean_file}, time {time.time() - code_start_time:.3f}')

    # Also compute and save the individual and group FC.
    individual_fc = isingmodel.get_fc_binary(s_mean=individual_state_mean, s_product_mean=individual_product_mean)
    print(f'individual FC min {individual_fc.min():.3g}, mean {individual_fc.mean():.3g}, max {individual_fc.max():.3g}')
    print(f'time {time.time() - code_start_time:.3f}')
    individual_fc_file = os.path.join(output_directory, f'fc_individual_{data_subset}.pt')
    torch.save(obj=individual_fc, f=individual_fc_file)
    print(f'saved {individual_fc_file}, time {time.time() - code_start_time:.3f}')

    group_fc = isingmodel.get_fc_binary(s_mean=group_state_mean, s_product_mean=group_product_mean)
    print(f'group FC min {group_fc.min():.3g}, mean {group_fc.mean():.3g}, max {group_fc.max():.3g}')
    print(f'time {time.time() - code_start_time:.3f}')
    group_fc_file = os.path.join(output_directory, f'fc_group_{data_subset}.pt')
    torch.save(obj=group_fc, f=group_fc_file)
    print(f'saved {group_fc_file}, time {time.time() - code_start_time:.3f}')

print(f'done, time {time.time() - code_start_time:.3f}')