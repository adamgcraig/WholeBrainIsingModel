import os
import torch
import time
import argparse
import hcpdatautilsnopandas as hcp

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
    parser.add_argument("-d", "--file_name_fragment", type=str, default='unit_scale', help="part of the output file name between mean_state_[data_subset]_ or mean_state_product_[data_subset]_ and .pt")
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
    subject_list = hcp.load_subject_subset(directory_path=input_directory, subject_subset=data_subset, require_sc=True)
    ts_per_subject = hcp.time_series_per_subject
    num_subjects = len(subject_list)
    num_nodes = hcp.num_brain_areas
    triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, dtype=int_type, device=device)
    triu_rows = triu_indices[0]
    triu_cols = triu_indices[1]
    num_pairs = triu_rows.numel()
    num_steps = hcp.num_time_points
    mean_state = torch.zeros( size=(ts_per_subject, num_subjects, num_nodes), dtype=float_type, device=device )
    mean_state_product = torch.zeros( size=(ts_per_subject, num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    num_ts = ts_per_subject * num_subjects
    num_ts_completed = 0
    print(f'time {time.time()-code_start_time:.3f}, starting processing of {ts_per_subject} time series per subject x {num_subjects} subjects = {ts_per_subject*num_subjects} files...')
    for ts_index in range(ts_per_subject):
        for subject_index in range(num_subjects):
            data_ts_file = hcp.get_time_series_file_path(directory_path=input_directory, subject_id=subject_list[subject_index], time_series_suffix=hcp.time_series_suffixes[ts_index])
            data_ts = hcp.load_matrix_from_binary(file_path=data_ts_file, dtype=float_type, device=device)
            data_ts /= data_ts.abs().max()# Divide the whole time series by the largest absolute value so that the FC stays the same but the means and mean products are <= 1.
            mean_state[ts_index,subject_index,:] = data_ts.mean(dim=0, keepdim=False)
            mean_state_product[ts_index,subject_index,:,:] = torch.matmul( data_ts.transpose(dim0=0, dim1=1), data_ts )/num_steps
            num_ts_completed += 1
            print(f'time {time.time()-code_start_time:.3f}, processed {data_ts_file}, {num_ts_completed} of {num_ts}')
    mean_state_file = os.path.join(output_directory, f'mean_state_{data_subset}_{file_name_fragment}.pt')
    torch.save(obj=mean_state, f=mean_state_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {mean_state_file}')
    mean_state_product_file = os.path.join(output_directory, f'mean_state_product_{data_subset}_{file_name_fragment}.pt')
    torch.save(obj=mean_state_product, f=mean_state_product_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {mean_state_product_file}, done')