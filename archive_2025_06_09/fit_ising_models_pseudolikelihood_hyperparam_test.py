# Based on the definitions of pseudolikelihood for Ising model parameters in
# H. Chau Nguyen, Riccardo Zecchina & Johannes Berg (2017)
# Inverse statistical problems: from the inverse Ising problem to data science,
# Advances in Physics, 66:3, 197-261, DOI: 10.1080/00018732.2017.1341604
# and
# Aurell, E., & Ekeberg, M. (2012).
# Inverse Ising inference using all the data.
# Physical review letters, 108(9), 090201.
# This version tries different learning rates and batch sizes for the optimization.

import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingutils as ising

code_start_time = time.time()
int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="Train a bunch of Ising models on fMRI time series data using pseudolikelihood maximization.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-a", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-n", "--num_nodes", type=int, default=360, help="number of nodes to model")
parser.add_argument("-r", "--num_reps", type=int, default=100, help="number of models to train for the subject")
parser.add_argument("-s", "--num_epochs", type=int, default=100, help="number of times we iterate over the full training data set")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
beta = args.beta
print(f'beta={beta:.3g}')
threshold = args.threshold
print(f'threshold={threshold:.3g}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
num_reps = args.num_reps
print(f'num_reps={num_reps}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')

# Use this model to train the bias and input weights to a single node of the Ising model.
# The input to either get_prob_accept() or forward() should be a 3D stack of system states
# with dimensions (num_subjects,T,num_nodes).
# num_subjects and num_nodes are pre-specified when we create the model.
# T can be any value.
# beta is a fixed, scalar hyperparameter of the Ising model.
# The return value is a scalar, the mean negative log pseudolikelihood
# of getting the observed time series/state given model parameters h and J.
# The stack in the 3rd dimension is a stack of separate models, one for each subject.
# Consequently, h has size (num_subjects, 1, num_nodes),
# and J has size (num_subjects, num_nodes, num_nodes).
class IsingModelNegativeLogPseudoLikelihood(torch.nn.Module):
    def __init__(self, num_models:int, num_nodes:int, dtype=float_type, device=device, beta:float=beta):
        super(IsingModelNegativeLogPseudoLikelihood, self).__init__()
        self.beta = beta
        self.model_dim = -3
        self.time_dim = -2
        self.node_dim = -1
        self.h = torch.nn.Parameter( torch.randn( (num_models, 1, num_nodes), dtype=dtype, device=device ) )
        self.J = torch.nn.Parameter(  torch.randn( (num_models, num_nodes, num_nodes), dtype=dtype, device=device )  )
        self.log_sigmoid = torch.nn.LogSigmoid()
    def get_delta_h(self, state:torch.Tensor):
        J_no_diag = self.J - torch.diag_embed( torch.diagonal(self.J, dim1=self.time_dim, dim2=self.node_dim), dim1=self.time_dim, dim2=self.node_dim )
        return self.beta * 2.0 * state * ( self.h + torch.matmul(state, J_no_diag) )
    def get_prob_accept(self, state:torch.Tensor):
        return torch.exp( -self.get_delta_h(state) ).clamp(min=0.0, max=0.99)
    def forward(self, data_ts:torch.Tensor):
        return -torch.mean(  self.log_sigmoid( self.get_delta_h(data_ts) )  )

def get_num_nan(mat:torch.Tensor):
    return torch.count_nonzero( torch.isnan(mat) )

def print_best(vals:torch.Tensor, learning_rates:list, batch_sizes:list, find_max:bool, val_name:str):
    big_val = 10e14
    mean_vals = vals.nanmean(dim=-1)
    is_nan_val = torch.isnan(mean_vals)
    num_bss = mean_vals.size(dim=-1)
    if find_max:
        mean_vals[is_nan_val] = -big_val
        best_index = torch.argmax(mean_vals)
    else:
        mean_vals[is_nan_val] = big_val
        best_index = torch.argmin(mean_vals)
    row_index = best_index//num_bss
    col_index = best_index % num_bss
    best_lr = learning_rates[row_index]
    best_bs = batch_sizes[col_index]
    best_mean = mean_vals[row_index, col_index]
    print(f'learning rate {best_lr:.3g}, batch size {best_bs:.3g} had best mean {val_name}, {best_mean:.3g}')

code_start_time = time.time()
int_type = torch.int
float_type = torch.float
device = torch.device('cuda')
subject_ids = hcp.load_training_subjects(directory_path=data_directory) + hcp.load_validation_subjects(directory_path=data_directory) + hcp.load_testing_subjects(directory_path=data_directory)
num_subjects = len(subject_ids)
# learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
learning_rates = [0.1, 0.001, 0.00001]
num_learning_rates = len(learning_rates)
# batch_sizes = [4800, 2400, 1600, 1200, 960, 800, 600, 480, 400, 320, 300, 240, 200, 160, 150, 100, 75, 64, 32, 30, 25, 20, 16, 15, 12, 10]
batch_sizes = [4800, 1200, 600, 75]
num_batch_sizes = len(batch_sizes)
fc_rmse_set = torch.zeros( (num_learning_rates, num_batch_sizes, num_reps), dtype=float_type, device=device )
fc_corr_set = torch.zeros( (num_learning_rates, num_batch_sizes, num_reps), dtype=float_type, device=device )
for subject_index in range(num_subjects):
    subject_id = subject_ids[subject_index]
    # Load, normalize, binarize, and flatten the fMRI time series data.
    print('loading fMRI data...')
    data_ts = ising.standardize_and_binarize_ts_data( ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, dtype=float_type, device=device), threshold=threshold, time_dim=1 ).flatten(start_dim=0, end_dim=-2)[:,:num_nodes]# .unsqueeze(dim=0).repeat( repeats=(num_reps, 1, 1) )
    print( 'data ts size: ', data_ts.size() )
    num_data_time_points = data_ts.size(dim=0)
    for lr_index in range(num_learning_rates):
        learning_rate = learning_rates[lr_index]
        for bs_index in range(num_batch_sizes):
            batch_size = batch_sizes[bs_index]
            num_batches = num_data_time_points//batch_size
            print('training models...')
            imnlpl_fn = IsingModelNegativeLogPseudoLikelihood(num_models=num_reps, num_nodes=num_nodes, dtype=float_type, device=device, beta=beta)
            optimizer = torch.optim.Adam( imnlpl_fn.parameters(), lr=learning_rate )
            min_loss = imnlpl_fn( data_ts.unsqueeze(dim=0) ).item()
            min_loss_epoch = -1
            for epoch in range(num_epochs):
                time_point_order = torch.randperm(num_data_time_points, dtype=int_type, device=device)
                data_ts_shuffled = data_ts[time_point_order,:]
                loss_sum = 0
                for batch in range(num_batches):
                    batch_start = batch*batch_size
                    batch_end = batch_start+batch_size
                    data_ts_batch = data_ts_shuffled[batch_start:batch_end,:].unsqueeze(dim=0)
                    optimizer.zero_grad()
                    loss = imnlpl_fn(data_ts_batch)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                    # data_ts_batch_num_pos = (data_ts_batch > 0.0).count_nonzero(dim=-2)
                    # data_ts_batch_num_all_neg = (data_ts_batch_num_pos == 0).count_nonzero()
                    # data_ts_batch_num_all_pos = (data_ts_batch_num_pos == batch_size).count_nonzero()
                    # if data_ts_batch_num_all_neg > 0:
                    #     print(f'epoch: {epoch}\tbatch: {batch}\t(region, subject) pairs with only -1s in batch: {data_ts_batch_num_all_neg}')
                    # if data_ts_batch_num_all_pos > 0:
                    #     print(f'epoch: {epoch}\tbatch: {batch}\t(region, subject) pairs with only +1s in batch: {data_ts_batch_num_all_pos}')
                    # num_J_nans = get_num_nan(imnlpl_fn.J)
                    # if num_J_nans > 0:
                    #     print(f'epoch: {epoch}\tbatch: {batch}\tNaNs in J: {num_J_nans}')
                    # num_h_nans = get_num_nan(imnlpl_fn.h)
                    # if num_h_nans > 0:
                    #     print(f'epoch: {epoch}\tbatch: {batch}\tNaNs in h: {num_h_nans}')
                    # time_elapsed = time.time() - code_start_time
                    # print(f'subject {subject_id} ({subject_index} of {num_subjects}), epoch {epoch} of {num_epochs}, batch {batch} of {num_batches}, loss {loss.item():.3g}, time {time_elapsed:.3f}')
                loss_mean = loss_sum/num_batches
                if loss_mean < min_loss:
                    min_loss = loss_mean
                    min_loss_epoch = epoch
            h = imnlpl_fn.h.squeeze()
            J = imnlpl_fn.J
            time_elapsed = time.time() - code_start_time
            print(f'done, min loss {min_loss:.3g} at epoch {min_loss_epoch}, time {time_elapsed:.3f}')
            with torch.no_grad():
                print('simulating with trained h and J...')
                s = ising.get_random_state(batch_size=num_reps, num_nodes=num_nodes, dtype=float_type, device=device)
                sim_ts = torch.zeros( (num_reps, num_data_time_points, num_nodes), dtype=float_type, device=device )
                sim_ts, s = ising.run_batched_balanced_metropolis_sim(sim_ts=sim_ts, J=J, h=h, s=s, num_steps=num_data_time_points, beta=beta)
                data_fc = hcp.get_fc(data_ts).unsqueeze(dim=0).repeat( (num_reps, 1, 1) )
                sim_fc = hcp.get_fc_batch(sim_ts)
                fc_rmse = hcp.get_triu_rmse_batch(data_fc, sim_fc)
                fc_corr = hcp.get_triu_corr_batch(data_fc, sim_fc)
                fc_rmse_set[lr_index, bs_index, :] = fc_rmse
                fc_corr_set[lr_index, bs_index, :] = fc_corr
                rmse_min = fc_rmse.min()
                rmse_mean = fc_rmse.mean()
                rmse_max = fc_rmse.max()
                corr_min = fc_corr.min()
                corr_mean = fc_corr.mean()
                corr_max = fc_corr.max()
                time_elapsed = time.time() - code_start_time
                print(f'done, learning rate {learning_rate} batch size {batch_size}, (min, mean, max) RMSE ({rmse_min:.3g}, {rmse_mean:.3g}, {rmse_max:.3g}), correlation ({corr_min:.3g}, {corr_mean:.3g}, {corr_max:.3g}) time: {time_elapsed:.3f}')
            # print('saving Ising model J and h...')
            # file_suffix = f'pl_nodes_{num_nodes}_epochs_{num_epochs}_bs_{batch_size}_lr_{learning_rate:.3f}_thresh_{threshold:.3f}_subject_{subject_id}'
            # J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
            # torch.save(J, J_file_name)
            # print(J_file_name)
            # h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
            # torch.save(h, h_file_name)
            # print(h_file_name)
            # time_elapsed = time.time() - code_start_time
            # print(f'done, time {time_elapsed:.3f}')
    print_best(vals=fc_rmse_set, learning_rates=learning_rates, batch_sizes=batch_sizes, find_max=False, val_name='FC RMSE')
    print_best(vals=fc_corr_set, learning_rates=learning_rates, batch_sizes=batch_sizes, find_max=True, val_name='FC correlation')
    print('saving FC RMSEs and correlations...')
    file_suffix = f'hyperparam_test_pl_nodes_{num_nodes}_epochs_{num_epochs}_thresh_{threshold:.3f}_subject_{subject_id}'
    rmse_file_name = os.path.join(output_directory, f'rmse_{file_suffix}.pt')
    torch.save(fc_rmse_set, rmse_file_name)
    print(rmse_file_name)
    corr_file_name = os.path.join(output_directory, f'corr_{file_suffix}.pt')
    torch.save(fc_corr_set, corr_file_name)
    print(corr_file_name)
time_elapsed = time.time() - code_start_time
print(f'done, time: {time_elapsed:.3f}')
