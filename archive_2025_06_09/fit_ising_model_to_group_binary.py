import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingutilsbool as ising

parser = argparse.ArgumentParser(description="Fit an Ising model to the concatenated time series of a set of subjects, using median binarization.' fMRI data.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-c", "--data_subset", type=str, default='training', help="which list of subjects to use, either training, validation, testing, or all")
parser.add_argument("-a", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate by which we multiply updates to weights and biases at each step")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes to model")
parser.add_argument("-d", "--num_time_points", type=int, default=4800, help="number of time points per simulation")
parser.add_argument("-z", "--batch_size", type=int, default=1000, help="number of models to fit at one time. For fastest performance, set it to the highest value where CUDA does not run out of memory.")
parser.add_argument("-w", "--num_reps", type=int, default=1000, help="number of models to fit per subject")
parser.add_argument("-o", "--window_length", type=int, default=50, help="number of time points between model parameter updates")
parser.add_argument("-e", "--num_epochs", type=int, default=10000, help="number of times to repeat the training time series")
parser.add_argument("-s", "--epochs_per_save", type=int, default=10000, help="number of epochs between saves of the models")
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
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate:.3g}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
batch_size = args.batch_size
print(f'batch_size={batch_size}')
num_reps = args.num_reps
print(f'num_reps={num_reps}')
num_time_points_per_subject = args.num_time_points
print(f'num_time_points={num_time_points_per_subject}')
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
scaled_learning_rate = learning_rate/window_length

def get_group_fc(subject_list:list, num_nodes:int=num_nodes, data_directory:str=data_directory, epsilon:float=10e-10):
    num_subjects = len(subject_list)
    ts_sum = torch.zeros( (num_nodes,), dtype=int_type, device=device )
    ts_product_sum = ts_sum[:,None] * ts_sum[None,:]
    num_steps = 0
    for subject_index in range(num_subjects):
        subject_id = subject_list[subject_index]
        subject_ts = ising.median_binarize_ts_data( ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device), time_dim=1 ).flatten(start_dim=0, end_dim=-2)[:,:num_nodes]
        ts_sum += torch.count_nonzero(subject_ts, dim=0)
        ts_product_sum += torch.count_nonzero( torch.logical_xor(subject_ts[:,:,None], subject_ts[:,None,:]), dim=0 )
        num_steps += subject_ts.size(dim=0)
    ts_mean = (2*ts_sum - 1).float()/num_steps
    ts_product_mean = (2*ts_product_sum - 1).float()/num_steps
    ts_squared_mean = torch.diagonal(ts_product_mean, dim1=-2, dim2=-1)
    ts_std = torch.sqrt( ts_squared_mean - ts_mean * ts_mean )
    ts_cov = ts_product_mean - ts_mean[:,None] * ts_mean[None,:]
    ts_std_prod = ts_std[:,None] * ts_std[None,:]
    if torch.any( ts_std_prod == 0.0 ):
        ts_cov += epsilon
        ts_std_prod += epsilon
    group_fc = ts_cov/ts_std_prod
    return group_fc

with torch.no_grad():
    code_start_time = time.time()
    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    subject_ids = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_ids)
    num_time_points_per_epoch = num_subjects * num_time_points_per_subject
    num_total_time_points = num_time_points_per_epoch * num_epochs
    ising_model_param_string = f'nodes_{num_nodes}_window_{window_length}_lr_{scaled_learning_rate:.3f}_median_binary_beta_{beta:.3f}'
    print(f'computing combined FC of {data_subset} subjects...')
    # This wastes some memory, but it is safer to just make it the same size as the batched sim_fc.
    data_fc = get_group_fc(subject_list=subject_ids, num_nodes=num_nodes, data_directory=data_directory)[None,:,:].repeat( (batch_size, 1, 1) )
    time_elapsed = time.time() - code_start_time
    print(f'time {time_elapsed}')
    fc_rmse = torch.zeros( (num_reps,), dtype=float_type, device=device )
    fc_corr = torch.zeros( (num_reps,), dtype=float_type, device=device )
    num_models = num_reps
    partial_batch_size = num_models % batch_size
    has_partial_batch = partial_batch_size > 0
    num_full_batches = num_models // batch_size
    num_batches = num_full_batches + int(has_partial_batch)
    subject_index = -1
    rep_index = -1
    rep_strings = [f'placeholder_{rep}' for rep in range(batch_size)]
    rep_in_batch = [0]*batch_size
    print('starting training...')
    for batch_index in range(num_batches):
        print(f'batch {batch_index} of {num_batches}')
        # Load the time series into data_ts.
        if (batch_index == num_batches-1) and has_partial_batch:
            current_batch_size = partial_batch_size
        else:
            current_batch_size = batch_size
        current_data_fc = data_fc[:current_batch_size,:,:]
        for index_into_batch in range(current_batch_size):
            rep_index += 1
            rep_in_batch[index_into_batch] = rep_index
            rep_strings[index_into_batch] = f'rep_{rep_index}'
        J, h, s = ising.get_batched_ising_models(batch_size=current_batch_size, num_nodes=num_nodes, dtype=float_type, device=device)
        delta_J = torch.zeros_like(J)
        delta_h = torch.zeros_like(h)
        for time_point in range(num_total_time_points):
            epoch = time_point // num_time_points_per_epoch
            steps_into_epoch = time_point % num_time_points_per_epoch
            subject_index = steps_into_epoch // num_time_points_per_subject
            steps_into_subject = time_point % num_time_points_per_subject
            if steps_into_subject == 0:
                subject_id = subject_ids[subject_index]
                print(f'subject {subject_index} of {num_subjects}, {subject_id}')
                single_data_ts = ising.median_binarize_ts_data( ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device), time_dim=-2 ).flatten(start_dim=0, end_dim=-2)[:num_time_points_per_subject,:num_nodes]
            s = ising.run_batched_balanced_metropolis_sim_step(J=J, h=h, s=s, beta=beta)
            d = single_data_ts[steps_into_subject,:].unsqueeze(dim=0)
            delta_h += ( d.int() - s.int() )
            delta_J += ( torch.logical_xor(d[:,:,None], d[:,None,:]).int() - torch.logical_xor(s[:,:,None], s[:,None,:]).int() )
            if (time_point % window_length) == (window_length-1):
                h += scaled_learning_rate * (1 - 2*delta_h).float()
                delta_h[:,:] = 0.0
                J += scaled_learning_rate * (1 - 2*delta_J).float()
                delta_J[:,:,:] = 0.0
            is_epoch_end = steps_into_epoch == (num_time_points_per_epoch-1)
            if is_epoch_end:
                print(f'end of epoch {epoch} of {num_epochs}')
            if is_epoch_end and ( (epoch % epochs_per_test) == (epochs_per_test-1) ):
                sim_fc, s = ising.run_batched_balanced_metropolis_sim_for_fc(J=J, h=h, s=s, num_steps=test_length, beta=beta)
                fc_corr_batch = hcp.get_triu_corr_batch(sim_fc, current_data_fc)
                fc_rmse_batch = hcp.get_triu_rmse_batch(sim_fc, current_data_fc)
                for index_into_batch in range(current_batch_size):
                    rep_index = rep_in_batch[index_into_batch]
                    fc_corr[rep_index] = fc_corr_batch[index_into_batch]
                    fc_rmse[rep_index] = fc_rmse_batch[index_into_batch]
                print(f'FC RMSE min {fc_rmse_batch.min():.3g}, mean {fc_rmse_batch.mean():.3g}, max {fc_rmse_batch.max():.3g}, corr min {fc_corr_batch.min():.3g}, mean {fc_corr_batch.mean():.3g}, max {fc_corr_batch.max():.3g}')
            if is_epoch_end and ( (epoch % epochs_per_save) == (epochs_per_save-1) ):
                for index_into_batch in range(current_batch_size):
                    rep_string = rep_strings[index_into_batch]
                    file_suffix = f'{ising_model_param_string}_group_{data_subset}_{rep_string}_epoch_{epoch}'
                    J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
                    torch.save( J[index_into_batch,:,:].clone(), J_file_name )
                    print(J_file_name)
                    h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
                    torch.save( h[index_into_batch,:].clone(), h_file_name )
                    print(h_file_name)
                time_elapsed = time.time() - code_start_time
                print(f'saved J and h batch {batch_index+1} of {num_batches}, covering through {rep_string}, epoch {epoch} of {num_epochs}, time {time_elapsed:.3f}')
    for index_into_batch in range(current_batch_size):
        rep_string = rep_strings[index_into_batch]
        file_suffix = f'{ising_model_param_string}_group_{data_subset}_{rep_string}_epoch_{epoch}'
        J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
        torch.save( J[index_into_batch,:,:].clone(), J_file_name )
        print(J_file_name)
        h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
        torch.save( h[index_into_batch,:].clone(), h_file_name )
        print(h_file_name)
    file_suffix = f'{ising_model_param_string}_group_{data_subset}_epoch_{epoch}'
    corr_file_name = os.path.join(output_directory, f'corr_{file_suffix}.pt')
    torch.save(fc_corr, corr_file_name)
    print(corr_file_name)
    rmse_file_name = os.path.join(output_directory, f'rmse_{file_suffix}.pt')
    torch.save(fc_rmse, rmse_file_name)
    print(rmse_file_name)
    time_elapsed = time.time() - code_start_time
    print(f'done, time {time_elapsed:.3f}')