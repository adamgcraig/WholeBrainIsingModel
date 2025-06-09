import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingutils as ising

parser = argparse.ArgumentParser(description="Train a bunch of Ising models on fMRI time series data.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-a", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate by which we multiply updates to weights and biases at each step")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes to model")
parser.add_argument("-w", "--num_reps", type=int, default=1000, help="maximum window length to try")
parser.add_argument("-d", "--num_time_points", type=int, default=4800, help="number of time points per simulation")
parser.add_argument("-o", "--window_length", type=int, default=50, help="number of time points between model parameter updates")
parser.add_argument("-e", "--num_epochs", type=int, default=10000, help="number of times to repeat the training time series")
parser.add_argument("-s", "--epochs_per_save", type=int, default=1000, help="number of epochs between saves of the models")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
output_directory = args.output_directory
beta = args.beta
print(f'beta={beta:.3g}')
threshold = args.threshold
print(f'threshold={threshold:.3g}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate:.3g}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
num_reps = args.num_reps
print(f'num_reps={num_reps}')
num_time_points = args.num_time_points
print(f'num_time_points={num_time_points}')
window_length = args.window_length
print(f'window_length={window_length}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
epochs_per_save = args.epochs_per_save
print(f'epochs_per_save={epochs_per_save}')
num_total_time_points = num_time_points * num_epochs
scaled_learning_rate = learning_rate/window_length

def get_num_nan(mat:torch.Tensor):
    return torch.count_nonzero( torch.isnan(mat) )

with torch.no_grad():
    code_start_time = time.time()
    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    subject_ids = hcp.load_training_subjects(directory_path=data_directory) + hcp.load_validation_subjects(directory_path=data_directory) + hcp.load_testing_subjects(directory_path=data_directory)
    num_subjects = len(subject_ids)
    # Load, normalize, binarize, and flatten the fMRI time series data.
    print('loading fMRI data...')
    data_ts = torch.zeros( (num_subjects, num_time_points, num_nodes), dtype=float_type, device=device )
    for subject_index in range(num_subjects):
        subject_id = subject_ids[subject_index]
        data_ts[subject_index,:,:] = ising.standardize_and_binarize_ts_data( ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device), threshold=threshold, time_dim=1 ).flatten(start_dim=0, end_dim=-2)[:num_time_points,:num_nodes]# .unsqueeze(dim=0).repeat( repeats=(num_reps, 1, 1) )
    data_ts = data_ts[None,:,:,:].expand(num_reps, -1, -1, -1)# .repeat( (num_reps, 1, 1, 1) ).flatten(start_dim=0, end_dim=1)
    print( 'data ts size: ', data_ts.size() )
    J, h, s = ising.get_batched_ising_models(batch_size=num_reps*num_subjects, num_nodes=num_nodes, dtype=float_type, device=device)
    delta_J = torch.zeros_like(J)
    delta_h = torch.zeros_like(h)
    window_lengths = torch.arange(start=1, end=num_reps, step=1, device=device, dtype=int_type)
    window_lengths_float = window_lengths.float()
    epoch = 1
    print('starting training...')
    for time_point in range(num_total_time_points):
        s = ising.run_batched_balanced_metropolis_sim_step(J=J, h=h, s=s, beta=beta)
        steps_into_data = time_point % num_time_points
        d = data_ts[:,:,steps_into_data,:].flatten(start_dim=0, end_dim=1)
        delta_h += (d - s)
        delta_J += (d[:,:,None] * d[:,None,:] - s[:,:,None] * s[:,None,:])
        if (time_point % window_length) == 0:
            h += scaled_learning_rate * delta_h
            delta_h[:,:] = 0.0
            J += scaled_learning_rate * delta_J
            delta_J[:,:,:] = 0.0
        is_epoch_end = steps_into_data == 0
        if is_epoch_end and (epoch % epochs_per_save) == 0:
            print('saving Ising model J and h...')
            h_stacked = h.reshape( (num_reps, num_subjects, num_nodes) )
            J_stacked = J.reshape( (num_reps, num_subjects, num_nodes, num_nodes) )
            for subject_index in range(num_subjects):
                subject_id = subject_ids[subject_index]
                file_suffix = f'_parallel_nodes_{num_nodes}_epochs_{epoch}_reps_{num_reps}_window_{window_length}_lr_{learning_rate:.3f}_threshold_{threshold:.3f}_beta_{beta:.3f}_subject_{subject_id}'
                J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
                torch.save( J_stacked[:,subject_index,:,:].clone(), J_file_name )
                print(J_file_name)
                h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
                torch.save( h_stacked[:,subject_index,:].clone(), h_file_name )
                print(h_file_name)
            time_elapsed = time.time() - code_start_time
            print(f'epoch {epoch}, time {time_elapsed:.3f}, max abs delta h {delta_h.abs().max():.3g}, max abs delta J {delta_J.abs().max():.3g}')
        epoch += int(is_epoch_end)
time_elapsed = time.time() - code_start_time
print(f'done, time: {time_elapsed:.3f}')