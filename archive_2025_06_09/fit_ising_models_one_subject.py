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
parser.add_argument("-w", "--window_length", type=int, default=20, help="steps between updates to model parameters")
# parser.add_argument("-p", "--prob_update", type=float, default=0.02, help="probability of updating the model parameters on any given step")
parser.add_argument("-n", "--num_nodes", type=int, default=90, help="number of nodes to model")
parser.add_argument("-e", "--num_epochs", type=int, default=1000, help="number of times to repeat the training time series")
parser.add_argument("-p", "--print_every_epochs", type=int, default=20, help="print min, median, and max FC correlation and RMSE once every this many epochs")
parser.add_argument("-s", "--save_every_epochs", type=int, default=100, help="save the model once every this many epochs")
parser.add_argument("-r", "--num_reps", type=int, default=100, help="number of models to train for the subject")
# parser.add_argument("-d", "--data_subset", type=str, default='training', help="which data subset to use, either training, validation or testing")
parser.add_argument("-o", "--subject_id", type=int, default=516742, help="ID of the subject on whose fMRI data we will train")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
output_directory = args.output_directory
beta = args.beta
print(f'beta={beta:.3g}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate:.3g}')
threshold = args.threshold
print(f'threshold={threshold:.3g}')
# prob_update = args.prob_update
# print(f'learning_rate={learning_rate:.3g}')
window_length = args.window_length
print(f'window_length={window_length}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
num_epochs = args.num_epochs
print(f'num_nodes={num_nodes}')
save_every_epochs = args.save_every_epochs
print(f'save_every_epochs={save_every_epochs}')
print_every_epochs = args.print_every_epochs
print(f'print_every_epochs={print_every_epochs}')
num_reps = args.num_reps
print(f'num_reps={num_reps}')
# data_subset = args.data_subset
subject_id = args.subject_id
print(f'subject_id={subject_id}')
    
def get_num_nan(mat:torch.Tensor):
    return torch.count_nonzero( torch.isnan(mat) )

with torch.no_grad():
    code_start_time = time.time()
    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    # Load, normalize, binarize, and flatten the fMRI time series data.
    print('loading fMRI data...')
    data_ts = ising.standardize_and_binarize_ts_data( ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device), threshold=threshold, time_dim=1 ).flatten(start_dim=0, end_dim=-2)[:,:num_nodes]
    print( 'data ts size: ', data_ts.size() )
    data_fc = hcp.get_fc(ts=data_ts).unsqueeze(dim=0).repeat( repeats=(num_reps, 1, 1) )
    print( 'data fc size: ', data_fc.size() )
    num_time_points = data_ts.size(dim=0)
    num_windows = num_time_points // window_length
    num_time_points_in_window = num_windows * window_length
    data_ts_window = data_ts[:num_time_points_in_window,:].reshape( shape=(num_windows, window_length, num_nodes) )
    data_mean_window = data_ts_window.mean(dim=1, keepdim=True)
    print( 'data_mean_window size:', data_mean_window.size() )
    data_cov_window = torch.mean(data_ts_window[:,:,:,None] * data_ts_window[:,:,None,:], dim=1, keepdim=True)
    print( 'data_cov_window size:', data_cov_window.size() )
    J, h, s = ising.get_batched_ising_models(batch_size=num_reps, num_nodes=num_nodes, dtype=float_type, device=device)
    for epoch in range(num_epochs):
        for window in range(num_windows):
            sim_ts, s = ising.run_batched_balanced_metropolis_sim(J=J, h=h, s=s, num_steps=window_length, beta=beta)
            sim_mean = sim_ts.mean(dim=1)
            sim_cov = torch.mean(sim_ts[:,:,:,None] * sim_ts[:,:,None,:], dim=1)
            h += learning_rate * ( data_mean_window[window,:,:] - sim_mean )
            J += learning_rate * ( data_cov_window[window,:,:,:] - sim_cov )
        if epoch % print_every_epochs == 0:
            sim_ts, s = ising.run_batched_balanced_metropolis_sim(J=J, h=h, s=s, num_steps=num_time_points, beta=beta)
            sim_fc = hcp.get_fc_batch(ts_batch=sim_ts)
            fc_corr = hcp.get_triu_corr_batch(sim_fc, data_fc)
            fc_corr_min = fc_corr.min()
            fc_corr_median = fc_corr.median()
            fc_corr_max = fc_corr.max()
            fc_rmse = hcp.get_triu_rmse_batch(sim_fc, data_fc)
            fc_rmse_min = fc_rmse.min()
            fc_rmse_median = fc_rmse.median()
            fc_rmse_max = fc_rmse.max()
            time_elapsed = time.time() - code_start_time
            print(f'epoch {epoch}\tFC corr min {fc_corr_min:.3g}\tmedian {fc_corr_median:.3g}\tmax {fc_corr_max:.3g}\tRMSE min {fc_rmse_min:.3g}\tmedian {fc_rmse_median:.3g}\tmax {fc_rmse_max:.3g}\ttime {time_elapsed:.3f}')
        if epoch % save_every_epochs == 0:
            # Save the h and J of the trained models.
            file_suffix = f'nodes_{num_nodes}_reps_{num_reps}_epochs_{epoch}_window_{window_length}_lr_{learning_rate:.3f}_threshold_{threshold:.3f}_subject_{subject_id}'
            print('saving Ising model J and h...')
            J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
            torch.save(J, J_file_name)
            h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
            torch.save(h, h_file_name)
    time_elapsed = time.time() - code_start_time
    print(f'done, time: {time_elapsed:.3f}')