import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingutils as ising

parser = argparse.ArgumentParser(description="Fit multiple Ising models to each subject's fMRI data.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-c", "--data_subset", type=str, default='all', help="which list of subjects to use, either training, validation, testing, or all")
parser.add_argument("-a", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate by which we multiply updates to weights and biases at each step")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes to model")
parser.add_argument("-d", "--num_time_points", type=int, default=4800, help="number of time points per simulation")
parser.add_argument("-z", "--batch_size", type=int, default=1000, help="number of models to fit at one time. For fastest performance, set it to the highest value where CUDA does not run out of memory.")
parser.add_argument("-w", "--num_reps", type=int, default=1000, help="number of models to fit per subject")
parser.add_argument("-o", "--window_length", type=int, default=50, help="number of time points between model parameter updates")
parser.add_argument("-e", "--num_epochs", type=int, default=10000, help="number of times to repeat the training time series")
parser.add_argument("-s", "--epochs_per_save", type=int, default=1000, help="number of epochs between saves of the models")
parser.add_argument("-y", "--epochs_per_test", type=int, default=10000, help="number of epochs between test runs of the models")
parser.add_argument("-x", "--test_length", type=int, default=4800, help="number of sim steps in test run")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
beta = args.beta
print(f'beta={beta:.3g}')
threshold = args.threshold
print(f'threshold={threshold:.3g}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate:.3g}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
batch_size = args.batch_size
print(f'batch_size={batch_size}')
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
epochs_per_test = args.epochs_per_test
print(f'epochs_per_test={epochs_per_test}')
test_length = args.test_length
print(f'test_length={test_length}')
num_total_time_points = num_time_points * num_epochs
scaled_learning_rate = learning_rate / window_length

def get_num_nan(mat:torch.Tensor):
    return torch.count_nonzero( torch.isnan(mat) )

with torch.no_grad():
    code_start_time = time.time()
    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    subject_ids = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_ids)
    print(f'Data subset {data_subset} has {num_subjects} subjects.')
    ising_model_param_string = f'nodes_{num_nodes}_window_{window_length}_lr_{scaled_learning_rate:.3f}_threshold_{threshold:.3f}_beta_{beta:.3f}'
    # Load, normalize, binarize, and flatten the fMRI time series data.
    data_ts = torch.zeros( (batch_size, num_time_points, num_nodes), dtype=float_type, device=device )
    data_fc = torch.zeros( (batch_size, num_nodes, num_nodes), dtype=float_type, device=device )
    fc_rmse = torch.zeros( (num_subjects, num_reps), dtype=float_type, device=device )
    fc_corr = torch.zeros( (num_subjects, num_reps), dtype=float_type, device=device )
    num_models = num_subjects * num_reps
    partial_batch_size = num_models % batch_size
    has_partial_batch = partial_batch_size > 0
    num_full_batches = num_models // batch_size
    num_batches = num_full_batches + int(has_partial_batch)
    subject_index = -1
    rep_index = -1
    subject_rep_strings = [f'placeholder_{rep}' for rep in range(batch_size)]
    subject_in_batch = [0]*batch_size
    rep_in_batch = [0]*batch_size
    print('starting training...')
    for batch_index in range(num_batches):
        print(f'batch {batch_index} of {num_batches}')
        # Load the time series into data_ts.
        if (batch_index == num_batches-1) and has_partial_batch:
            current_batch_size = partial_batch_size
        else:
            current_batch_size = batch_size
        current_data_ts = data_ts[:current_batch_size,:,:]
        current_data_fc = data_fc[:current_batch_size,:,:]
        for index_into_batch in range(current_batch_size):
            subject_index = (subject_index+1) % num_subjects
            subject_in_batch[index_into_batch] = subject_index
            if subject_index == 0:
                rep_index += 1
            rep_in_batch[index_into_batch] = rep_index
            subject_id = subject_ids[subject_index]
            print(f'subject {subject_index} of {num_subjects}, {subject_id}, rep {rep_index} of {num_reps}')
            single_data_ts = ising.standardize_and_binarize_ts_data( ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device), threshold=threshold, time_dim=1 ).flatten(start_dim=0, end_dim=-2)[:num_time_points,:num_nodes]
            current_data_ts[index_into_batch,:,:] = single_data_ts
            single_data_fc = hcp.get_fc(single_data_ts)
            current_data_fc[index_into_batch,:,:] = single_data_fc
            subject_rep_strings[index_into_batch] = f'subject_{subject_id}_rep_{rep_index}'
        J, h, s = ising.get_batched_ising_models(batch_size=current_batch_size, num_nodes=num_nodes, dtype=float_type, device=device)
        delta_J = torch.zeros_like(J)
        delta_h = torch.zeros_like(h)
        for time_point in range(num_total_time_points):
            s = ising.run_batched_balanced_metropolis_sim_step(J=J, h=h, s=s, beta=beta)
            epoch = time_point // num_time_points
            steps_into_data = time_point % num_time_points
            d = current_data_ts[:,steps_into_data,:]
            delta_h += (d - s)
            delta_J += (d[:,:,None] * d[:,None,:] - s[:,:,None] * s[:,None,:])
            if (time_point % window_length) == (window_length-1):
                h += scaled_learning_rate * delta_h
                delta_h[:,:] = 0.0
                J += scaled_learning_rate * delta_J
                delta_J[:,:,:] = 0.0
            is_epoch_end = steps_into_data == (num_time_points-1)
            if is_epoch_end:
                print(f'end of epoch {epoch} of {num_epochs}')
            if is_epoch_end and ( (epoch % epochs_per_test) == (epochs_per_test-1) ):
                sim_fc, s = ising.run_batched_balanced_metropolis_sim_for_fc(J=J, h=h, s=s, num_steps=test_length, beta=beta)
                fc_corr_batch = hcp.get_triu_corr_batch(sim_fc, current_data_fc)
                fc_rmse_batch = hcp.get_triu_rmse_batch(sim_fc, current_data_fc)
                for index_into_batch in range(current_batch_size):
                    subject_index = subject_in_batch[index_into_batch]
                    rep_index = rep_in_batch[index_into_batch]
                    fc_corr[subject_index, rep_index] = fc_corr_batch[index_into_batch]
                    fc_rmse[subject_index, rep_index] = fc_rmse_batch[index_into_batch]
                print(f'FC RMSE min {fc_rmse_batch.min():.3g}, mean {fc_rmse_batch.mean():.3g}, max {fc_rmse_batch.max():.3g}, corr min {fc_corr_batch.min():.3g}, mean {fc_corr_batch.mean():.3g}, max {fc_corr_batch.max():.3g}')
            if is_epoch_end and ( (epoch % epochs_per_save) == (epochs_per_save-1) ):
                for index_into_batch in range(current_batch_size):
                    subject_rep_string = subject_rep_strings[index_into_batch]
                    file_suffix = f'{ising_model_param_string}_{subject_rep_string}_epoch_{epoch}'
                    J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
                    torch.save( J[index_into_batch,:,:].clone(), J_file_name )
                    print(J_file_name)
                    h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
                    torch.save( h[index_into_batch,:].clone(), h_file_name )
                    print(h_file_name)
                time_elapsed = time.time() - code_start_time
                print(f'saved J and h, {batch_index} of {num_batches}, covering through {subject_rep_string}, epoch {epoch} of {num_epochs}, time {time_elapsed:.3f}')
    for index_into_batch in range(current_batch_size):
        subject_rep_string = subject_rep_strings[index_into_batch]
        file_suffix = f'{ising_model_param_string}_{subject_rep_string}_epoch_{epoch}'
        J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
        torch.save( J[index_into_batch,:,:].clone(), J_file_name )
        print(J_file_name)
        h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
        torch.save( h[index_into_batch,:].clone(), h_file_name )
        print(h_file_name)
    file_suffix = f'{ising_model_param_string}_{data_subset}_epoch_{epoch}'
    corr_file_name = os.path.join(output_directory, f'corr_{file_suffix}.pt')
    torch.save(fc_corr, corr_file_name)
    print(corr_file_name)
    rmse_file_name = os.path.join(output_directory, f'rmse_{file_suffix}.pt')
    torch.save(fc_rmse, rmse_file_name)
    print(rmse_file_name)
    time_elapsed = time.time() - code_start_time
    print(f'done, time {time_elapsed:.3f}')