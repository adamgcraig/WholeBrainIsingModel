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
parser.add_argument("-r", "--num_reps", type=int, default=100, help="number of models to train for the subject")
parser.add_argument("-w", "--windows_per_epoch", type=int, default=96, help="number of updates to model parameters in one pass over the time series data")
parser.add_argument("-e", "--epochs_per_print", type=int, default=20, help="number of times to repeat the training time series before printing out an update")
parser.add_argument("-p", "--prints_per_save", type=int, default=20, help="number of times to print min, median, and max FC correlation and RMSE before saving")
parser.add_argument("-s", "--num_saves", type=int, default=20, help="number of models to save")
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
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
num_reps = args.num_reps
print(f'num_reps={num_reps}')
windows_per_epoch = args.windows_per_epoch
print(f'windows_per_epoch={windows_per_epoch}')
epochs_per_print = args.epochs_per_print
print(f'epochs_per_print={epochs_per_print}')
prints_per_save = args.prints_per_save
print(f'prints_per_save={prints_per_save}')
num_saves = args.num_saves
print(f'num_saves={num_saves}')
    
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
        data_ts = ising.standardize_and_binarize_ts_data( ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device), threshold=threshold, time_dim=1 ).flatten(start_dim=0, end_dim=-2)[:,:num_nodes]
        print( 'data ts size: ', data_ts.size() )
        data_fc = hcp.get_fc(ts=data_ts).unsqueeze(dim=0).repeat( repeats=(num_reps, 1, 1) )
        print( 'data fc size: ', data_fc.size() )
        num_time_points = data_ts.size(dim=0)
        window_length = num_time_points // windows_per_epoch
        num_time_points_in_window = window_length * windows_per_epoch
        data_ts_window = data_ts[:num_time_points_in_window,:].reshape( shape=(windows_per_epoch, window_length, num_nodes) )
        data_mean_window = data_ts_window.mean(dim=1, keepdim=True)
        print( 'data_mean_window size:', data_mean_window.size() )
        data_cov_window = torch.mean(data_ts_window[:,:,:,None] * data_ts_window[:,:,None,:], dim=1, keepdim=True)
        print( 'data_cov_window size:', data_cov_window.size() )
        J, h, s = ising.get_batched_ising_models(batch_size=num_reps, num_nodes=num_nodes, dtype=float_type, device=device)
        sim_ts_window = torch.zeros( (num_reps, window_length, num_nodes), dtype=float_type, device=device )
        sim_ts_full = torch.zeros( (num_reps, num_time_points, num_nodes), dtype=float_type, device=device )
        total_updates = 0
        print('starting training...')
        for save_stage in range(num_saves):
            for print_stage in range(prints_per_save):
                for epoch in range(epochs_per_print):
                    for window in range(windows_per_epoch):
                        sim_ts_window, s = ising.run_batched_balanced_metropolis_sim(sim_ts=sim_ts_window, J=J, h=h, s=s, num_steps=window_length, beta=beta)
                        h += learning_rate * ( data_mean_window[window,:,:] - sim_ts_window.mean(dim=1) )
                        J += learning_rate * ( data_cov_window[window,:,:,:] - torch.mean(sim_ts_window[:,:,:,None] * sim_ts_window[:,:,None,:], dim=1) )
                        total_updates += 1
                sim_ts_full, s = ising.run_batched_balanced_metropolis_sim(sim_ts=sim_ts_full, J=J, h=h, s=s, num_steps=num_time_points, beta=beta)
                sim_fc = hcp.get_fc_batch(ts_batch=sim_ts_full)
                fc_corr = hcp.get_triu_corr_batch(sim_fc, data_fc)
                fc_corr_min = fc_corr.min()
                fc_corr_median = fc_corr.median()
                fc_corr_max = fc_corr.max()
                fc_rmse = hcp.get_triu_rmse_batch(sim_fc, data_fc)
                fc_rmse_min = fc_rmse.min()
                fc_rmse_median = fc_rmse.median()
                fc_rmse_max = fc_rmse.max()
                time_elapsed = time.time() - code_start_time
                print(f'subject index {subject_index}\tID {subject_id}\tupdate {total_updates}\tFC corr min {fc_corr_min:.3g}\tmedian {fc_corr_median:.3g}\tmax {fc_corr_max:.3g}\tRMSE min {fc_rmse_min:.3g}\tmedian {fc_rmse_median:.3g}\tmax {fc_rmse_max:.3g}\ttime {time_elapsed:.3f}')
            # Save the h and J of the trained models.
            file_suffix = f'nodes_{num_nodes}_reps_{num_reps}_epochs_{total_updates}_window_{windows_per_epoch}_lr_{learning_rate:.3f}_threshold_{threshold:.3f}_subject_{subject_id}'
            print('saving Ising model J and h...')
            J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
            torch.save(J, J_file_name)
            h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
            torch.save(h, h_file_name)
    time_elapsed = time.time() - code_start_time
    print(f'done, time: {time_elapsed:.3f}')