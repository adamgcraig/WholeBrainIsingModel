import os
import torch
import time
import argparse
import hcpdatautils as hcp

parser = argparse.ArgumentParser(description="Load the individual binary fMRI time series from simple binary files and put them in three PyTorch pickle files according to data subset.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='all', help="training, validation, testing, or all")
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
    device = torch.device('cuda')
    # device = torch.device('cpu')
    subject_ids = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_ids)
    print(f'Data subset {data_subset} has {num_subjects} subjects.')
    # Load, normalize, binarize, and flatten the fMRI time series data.
    data_ts = torch.zeros( (num_subjects, hcp.time_series_per_subject, hcp.num_time_points, hcp.num_brain_areas), dtype=float_type, device=device )
    print(f'preallocated space for each unique subject time series..., time {time.time() - code_start_time:.3f}')
    for subject_index in range(num_subjects):
        subject_id = subject_ids[subject_index]
        print(f'subject {subject_index} of {num_subjects}, ID {subject_id}, time {time.time() - code_start_time:.3f}')
        # We originally load a 4 x T/4 x N' Tensor with values over a continuous range.
        # N' is the original total number of nodes. Cut the dimensions down to N, the desired number of nodes.
        data_ts[:,subject_index,:,:] = hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device)
    print(f'loaded all time series, time {time.time() - code_start_time:.3f}')
    data_ts_file = os.path.join(output_directory, f'data_ts_{data_subset}.pt')
    torch.save(obj=data_ts, f=data_ts_file)
    print(f'done, time {time.time() - code_start_time:.3f}')