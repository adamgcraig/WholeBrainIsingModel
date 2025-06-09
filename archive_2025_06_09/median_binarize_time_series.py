import os
import torch
import time
import argparse
import hcpdatautils as hcp

parser = argparse.ArgumentParser(description="Binarize the time series so that each individual region time sereis has an equal number of 0s and 1s. Save to a single file.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='training', help="the subset of subjects over which to search for unique states")
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
    num_reps = hcp.time_series_per_subject
    num_time_points = hcp.num_time_points
    num_nodes = hcp.num_brain_areas
    data_ts = torch.zeros( (num_subjects, num_reps, num_nodes, num_time_points), dtype=torch.bool, device=device )
    for subject in range(num_subjects):
        subject_id = subject_list[subject]
        for rep in range(num_reps):
            rep_string = hcp.time_series_suffixes[rep]
            data_ts_file = hcp.get_time_series_file_path(directory_path=data_directory, subject_id=subject_id, time_series_suffix=rep_string)
            data_ts_unbinarized = hcp.load_matrix_from_binary(file_path=data_ts_file, dtype=float_type, device=device).transpose(dim0=0, dim1=1)
            # print( 'data_ts size', data_ts.size() )
            data_ts[subject, rep, :, :] = data_ts_unbinarized > torch.median(data_ts_unbinarized, dim=-1, keepdim=True).values
            print( f'subject {subject} of {num_subjects}, ts {subject_id} {rep_string}'  )
    data_ts_file = os.path.join(output_directory, f'data_ts_gt_median_{data_subset}.pt')
    torch.save(obj=data_ts, f=data_ts_file)
    print(f'saved {data_ts_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')