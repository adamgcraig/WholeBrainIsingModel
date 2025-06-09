import os
import torch
import time
import argparse
import hcpdatautils as hcp
from isingutilsslow import IsingModel
from isingutilsslow import binarize_data_ts
from isingutilsslow import get_triu_corr
from isingutilsslow import get_triu_rmse
from isingutilsslow import get_data_means_and_covs
from isingutilsslow import get_fc

parser = argparse.ArgumentParser(description="Fit multiple Ising models to each subject's fMRI data.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='training', help="which list of subjects to use, either training, validation, testing, or all")
parser.add_argument("-b", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate by which we multiply updates to weights and biases at each step")
parser.add_argument("-t", "--threshold", type=str, default='median', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, or the string 'median'")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes to model")
parser.add_argument("-d", "--num_time_points", type=int, default=4800, help="number of time points to expect per data time series")
parser.add_argument("-r", "--reps_per_batch", type=int, default=10, help="number of replicas of each subject to train in a single batch")
parser.add_argument("-c", "--num_batches", type=int, default=100, help="number of batches to train serially and save to separate files")
parser.add_argument("-w", "--window_length", type=int, default=50, help="number of time points between model parameter updates")
parser.add_argument("-e", "--epochs_per_save", type=int, default=1, help="number of epochs between saves of the models")
parser.add_argument("-v", "--num_saves", type=int, default=10, help="number of times to save a given batch of models during training")
parser.add_argument("-z", "--test_length", type=int, default=4800, help="number of sim steps in test run")
parser.add_argument("-f", "--compute_fim", type=bool, default=False, help="set to True to compute the Fisher information matrices of the Ising models")
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
threshold_str = args.threshold
if threshold_str == 'median':
    threshold = threshold_str
else:
    threshold = float(threshold_str)
print(f'threshold={threshold_str}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
num_time_points = args.num_time_points
print(f'num_time_points={num_time_points}')
reps_per_batch = args.reps_per_batch
print(f'reps_per_batch={reps_per_batch}')
num_batches = args.num_batches
print(f'num_batches={num_batches}')
window_length = args.window_length
print(f'window_length={window_length}')
epochs_per_save = args.epochs_per_save
print(f'epochs_per_save={epochs_per_save}')
num_saves = args.num_saves
print(f'num_saves={num_saves}')
test_length = args.test_length
print(f'test_length={test_length}')
compute_fim = args.compute_fim
print(f'compute_fim={compute_fim}')

def get_num_nan(mat:torch.Tensor):
    return torch.count_nonzero( torch.isnan(mat) )

def prep_ts_data(data_directory:str, data_subset:str, num_nodes:int, window_length:int):
    subject_ids = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_ids)
    print(f'Data subset {data_subset} has {num_subjects} subjects.')
    # Load, normalize, binarize, and flatten the fMRI time series data.
    data_ts = torch.zeros( (num_subjects, hcp.time_series_per_subject, hcp.num_time_points, num_nodes), dtype=float_type, device=device )
    print(f'preallocated space for each unique subject time series..., time {time.time() - code_start_time:.3f}')
    for subject_index in range(num_subjects):
        subject_id = subject_ids[subject_index]
        print(f'subject {subject_index} of {num_subjects}, ID {subject_id}')
        # We originally load a 4 x T/4 x N' Tensor with values over a continuous range.
        # N' is the original total number of nodes. Cut the dimensions down to N, the desired number of nodes.
        data_ts[subject_index,:,:,:] = hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device)[:,:,:num_nodes]
    print(f'loaded all time series, time {time.time() - code_start_time:.3f}')
    # Binarize each individual dimension of each individual time series separately in case of batch effects.
    # Then rearrange things: num_subjects x 4 x T/4 x num_nodes -> num_subjects x T x num_nodes -> num_subjects x num_nodes x T -> 1 x num_subjects x num_nodes x T 
    data_ts = binarize_data_ts(data_ts=data_ts, step_dim=-2, threshold=threshold).flatten(start_dim=1, end_dim=2).transpose(dim0=-2, dim1=-1).unsqueeze(dim=0)
    print(f'binarized time series with threshold {threshold_str} and rearranged dimensions, time {time.time() - code_start_time:.3f}')
    print( 'data_ts size', data_ts.size() )
    # Precompute means and covs to compare to those of the Ising model sim windows.
    # data_means is 1 x num_subjects x num_nodes x num_windows
    # data_covs is 1 x num_subjects x num_nodes x num_nodes x num_windows
    data_means, data_covs = get_data_means_and_covs(data_ts=data_ts, window=window_length)
    # Since the windows are of consistent size, we can take the means of the mean and covs over all of them to get a mean and cov over all time points,
    # except for the truncated ones.
    data_fc = get_fc( s_mean=data_means.mean(dim=-1), s_cov=data_covs.mean(dim=-1) )
    print(f'computed data FC, time {time.time() - code_start_time:.3f}')
    # Add a singleton batch dimension along which we can broadcast to individual replications.
    return data_means, data_covs, data_fc

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    data_means, data_covs, data_fc = prep_ts_data(data_directory=data_directory, data_subset=data_subset, num_nodes=num_nodes, window_length=window_length)
    num_subjects = data_means.size(dim=1)
    # Make a string of the parts of the file name that are constant across all output files.
    file_const_params = f'nodes_{num_nodes}_window_{window_length}_lr_{learning_rate:.3f}_threshold_{threshold_str}_beta_{beta:.3f}_subjects_{data_subset}_reps_{reps_per_batch}'
    for batch_index in range(num_batches):
        # Create the Ising models.
        ising_model = IsingModel(reps_per_subject=reps_per_batch, num_subjects=num_subjects, num_nodes=num_nodes, beta=beta, dtype=float_type, device=device)
        print(f'created Ising model for batch {batch_index}, time {time.time() - code_start_time:.3f}')
        for super_epoch in range(num_saves):
            epoch = (super_epoch+1) * epochs_per_save
            print(f'fitting at epoch {epoch}, time {time.time() - code_start_time:.3f}')
            ising_model.fit_faster(data_means=data_means, data_covs=data_covs, num_epochs=epochs_per_save, window_length=window_length, learning_rate=learning_rate)
            print(f'simulating at epoch {epoch}, time {time.time() - code_start_time:.3f}')
            # sim_fc = ising_model.simulate_and_record_fc(num_steps=test_length)
            if compute_fim:
                sim_fc, fim = ising_model.simulate_and_record_fc_and_fim(num_steps=test_length)
            else:
                sim_fc = ising_model.simulate_and_record_fc_faster(num_steps=test_length)
                fim = None
            fc_rmse = get_triu_rmse(sim_fc, data_fc)
            fc_corr = get_triu_corr(sim_fc, data_fc)
            print(f'FC corr min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}, RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}, time {time.time() - code_start_time:.3f}')
            file_suffix = f'{file_const_params}_batch_{batch_index}_epoch_{epoch}.pt'
            ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
            torch.save(obj=ising_model, f=ising_model_file)
            print(f'saved {ising_model_file}, time {time.time() - code_start_time:.3f}')
            fc_corr_file = os.path.join(output_directory, f'fc_corr_{file_suffix}')
            torch.save(obj=fc_corr, f=fc_corr_file)
            print(f'saved {fc_corr_file}, time {time.time() - code_start_time:.3f}')
            fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{file_suffix}')
            torch.save(obj=fc_rmse, f=fc_rmse_file)
            print(f'saved {fc_rmse_file}, time {time.time() - code_start_time:.3f}')
            if compute_fim:
                fim_file = os.path.join(output_directory, f'fim_{file_suffix}')
                torch.save(obj=fim, f=fim_file)
                print(f'saved {fim_file}, time {time.time() - code_start_time:.3f}')
    print(f'done, time {time.time() - code_start_time:.3f}')