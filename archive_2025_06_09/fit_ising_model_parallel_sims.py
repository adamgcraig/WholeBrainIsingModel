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
parser.add_argument("-n", "--num_nodes", type=int, default=360, help="number of nodes to model")
parser.add_argument("-u", "--num_sims", type=int, default=1200, help="number of simulations to run in parallel")
parser.add_argument("-w", "--window_length", type=int, default=10, help="number of steps to simulate between updates")
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
num_sims = args.num_sims
print(f'num_sims={num_sims}')
window_length = args.window_length
print(f'window_length={window_length}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
epochs_per_save = args.epochs_per_save
print(f'epochs_per_save={epochs_per_save}')
scaled_learning_rate = learning_rate/float(window_length)

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
    for subject_index in range(num_subjects):
        subject_id = subject_ids[subject_index]
        # Load, normalize, binarize, and flatten the fMRI time series data.
        print('loading fMRI data...')
        data_ts = ising.standardize_and_binarize_ts_data( ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device), threshold=threshold, time_dim=1 ).flatten(start_dim=0, end_dim=-2)[:,:num_nodes]# .unsqueeze(dim=0).repeat( repeats=(num_reps, 1, 1) )
        print( 'data ts size: ', data_ts.size() )
        num_data_time_points = data_ts.size(dim=1)
        num_total_time_points = num_data_time_points * num_epochs
        J, h, s = ising.get_batched_ising_models(batch_size=1, num_nodes=num_nodes, dtype=float_type, device=device)
        delta_J = torch.zeros_like(J)
        delta_h = torch.zeros_like(h)
        window_lengths = torch.arange(start=1, end=num_sims, step=1, device=device, dtype=int_type)
        window_lengths_float = window_lengths.float()
        epoch = 0
        print('starting training...')
        for time_point in range(num_total_time_points):
            s = ising.run_batched_balanced_metropolis_sim_step(J=J, h=h, s=s, beta=beta)
            steps_into_data = time_point % num_data_time_points
            d = data_ts[steps_into_data, :]
            delta_h += (d - s)
            delta_J += (d[None,:,None] * d[None,None,:] - s[:,:,None] * s[:,None,:])
            if (time_point % window_length) == 0:
                h += scaled_learning_rate * delta_h.mean(dim=0, keepdim=True)
                delta_h[:,:] = 0.0
                J += scaled_learning_rate * delta_J.mean(dim=0, keepdim=True)
                delta_J[:,:,:] = 0.0
            if ( steps_into_data == (num_data_time_points-1) ):
                time_elapsed = time.time() - code_start_time
                print(f'subject {subject_id} ({subject_index} of {num_subjects}), epoch {epoch}, time {time_elapsed:.3f}')
                if ( (epoch % epochs_per_save) == epochs_per_save-1 ):
                    print('saving Ising model J and h...')
                    file_suffix = f'parallel_sim_nodes_{num_nodes}_epochs_{epoch}_sims_{num_sims}_window_{window_length}_lr_{learning_rate:.3f}_threshold_{threshold:.3f}_subject_{subject_id}'
                    J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
                    torch.save(J, J_file_name)
                    print(J_file_name)
                    h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
                    torch.save(h, h_file_name)
                    print(h_file_name)
                epoch += 1

    print('saving Ising model J and h...')
    file_suffix = f'parallel_sim_nodes_{num_nodes}_epochs_{epoch}_sims_{num_sims}_window_{window_length}_lr_{learning_rate:.3f}_threshold_{threshold:.3f}_subject_{subject_id}'
    J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
    torch.save(J, J_file_name)
    print(J_file_name)
    h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
    torch.save(h, h_file_name)
    print(h_file_name)
    time_elapsed = time.time() - code_start_time
    print(f'done, time: {time_elapsed:.3f}')