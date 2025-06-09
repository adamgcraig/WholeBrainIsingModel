import os
import torch
import time
import argparse
import hcpdatautils as hcp

parser = argparse.ArgumentParser(description="Train a bunch of Ising models on fMRI time series data.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-a", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate by which we multiply updates to weights and biases at each step")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-p", "--prob_update", type=float, default=0.02, help="probability of updating the model parameters on any given step")
parser.add_argument("-n", "--num_nodes", type=int, default=360, help="number of nodes to model")
parser.add_argument("-e", "--num_epochs", type=int, default=100, help="number of times to repeat the training time series")
parser.add_argument("-r", "--num_reps", type=int, default=5, help="number of models to train for each subject")
parser.add_argument("-d", "--data_subset", type=str, default='training', help="which data subset to use, either training, validation or testing")
parser.add_argument("-o", "--subjects_start", type=int, default=0, help="index of first subject in slice on which to train")
parser.add_argument("-x", "--subjects_end", type=int, default=10, help="index one past last subject in slice on which to train")
args = parser.parse_args()
data_directory = args.data_directory
output_directory = args.output_directory
learning_rate = args.learning_rate
beta = args.beta
threshold = args.threshold
prob_update = args.prob_update
num_nodes = args.num_nodes
num_epochs = args.num_epochs
num_reps = args.num_reps
data_subset = args.data_subset
subjects_start = args.subjects_start
subjects_end = args.subjects_end

def binarize_ts_data(data_ts:torch.Tensor, data_threshold:float):
    data_std, data_mean = torch.std_mean(data_ts, dim=-2, keepdim=True)
    data_ts -= data_mean
    data_ts /= data_std
    data_ts = data_ts.flatten(start_dim=1, end_dim=-2)
    data_ts = torch.sign(data_ts - data_threshold)
    return data_ts

def get_batched_ising_models(batch_size:int, num_nodes:int):
    J = torch.randn( (batch_size, num_nodes, num_nodes), dtype=float_type, device=device )
    J = 0.5*( J + J.transpose(dim0=-2, dim1=-1) )
    J = J - torch.diag_embed( torch.diagonal(J,dim1=-2,dim2=-1) )
    h = torch.zeros( (batch_size, num_nodes), dtype=float_type, device=device )
    s = 2.0*torch.randint( 2, (batch_size, num_nodes), dtype=float_type, device=device ) - 1.0
    return J, h, s
    
def print_if_nan(mat:torch.Tensor):
    mat_is_nan = torch.isnan(mat)
    num_nan = torch.count_nonzero(mat_is_nan)
    if num_nan > 0:
        print( mat.size(), f'matrix contains {num_nan} NaNs.' )

with torch.no_grad():
    code_start_time = time.time()
    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    # torch.bernoulli() needs this to be a Tensor.
    prob_update = torch.tensor([prob_update], dtype=float_type, device=device)

    # Load, normalize, binarize, and flatten the fMRI time series data.
    print('loading fMRI data...')
    if data_subset == 'validation':
        subject_ids = hcp.load_validation_subjects(directory_path=data_directory)
    else:
        subject_ids = hcp.load_training_subjects(directory_path=data_directory)
    subject_ids = subject_ids[subjects_start:subjects_end]
    data_ts = hcp.load_all_time_series_for_subjects(directory_path=data_directory, subject_ids=subject_ids, dtype=float_type, device=device)
    data_ts = binarize_ts_data(data_ts=data_ts, data_threshold=threshold)
    data_ts = data_ts[:,:,:num_nodes]
    num_subjects, num_time_points, _ = data_ts.size()
    print('loaded with subjects x time points x nodes: ', num_subjects, num_time_points, num_nodes)
    data_fc = hcp.get_fc_batch(data_ts)

    # Use -2.0 as a placeholder value, since neither the correlation (restricted to [-1, +1]) nor the RMSE (restricted to [0, 2]) can take on this value.
    fc_corr_history = torch.full( (num_epochs, num_subjects), -2.0, dtype=float_type, device=device )
    fc_rmse_history = torch.full( (num_epochs, num_subjects), -2.0, dtype=float_type, device=device )

    # Pre-allocate space for our time series.
    sim_ts = torch.zeros_like(data_ts)
    # Pre-allocate space for running totals of mean and covariance.
    data_mu_sum = torch.zeros( (num_subjects, num_nodes), dtype=float_type, device=device )
    data_cov_sum = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    sim_mu_sum = torch.zeros( (num_subjects, num_nodes), dtype=float_type, device=device )
    sim_cov_sum = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    num_steps = 0.0

    # Train the model by
    # running it for a few steps at a time,
    # finding the means and covariances of individual regions in the simulation,
    # comparing them to the means and covariances in a window of the fMRI data,
    # adjusting h in proportion to the difference in means,
    # adjusting J in proportion to the difference in covariances,
    # repeating for all windows in the time series,
    # and repeating the full time series for a set number of epochs.
    print('training Ising model...')
    print('rep\tepoch\tnum flips\tworst corr.\tworst RMSE\ttime')
    for rep in range(num_reps):    # Initialize the model and run a sim prior to training to establish a baseline for improvement.
        # print('initializing Ising model for rep', rep, '...')
        J, h, s = get_batched_ising_models(batch_size=num_subjects, num_nodes=num_nodes)
        num_steps = 0.0
        num_flips = 0
        data_mu_sum[:,:] = 0.0
        sim_mu_sum[:,:] = 0.0
        data_cov_sum[:,:,:] = 0.0
        sim_cov_sum[:,:,:] = 0.0
        for epoch in range(num_epochs):
            for t in range(num_time_points):
                node_order = torch.randperm(n=num_nodes, dtype=int_type, device=device)
                for i in range(num_nodes):
                    node_index = node_order[i]
                    deltaH = 2*(  torch.sum( J[:,:,node_index] * s, dim=-1 ) + h[:,node_index]  )*s[:,node_index]
                    prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
                    flip = torch.bernoulli(prob_accept)
                    s[:,node_index] *= ( 1.0 - 2.0*flip )
                    num_flips += torch.count_nonzero(flip)
                sim_ts[:,t,:] = s
                d = data_ts[:,t,:]
                data_mu_sum += d
                sim_mu_sum += s
                data_cov_sum += d[:,:,None] * d[:,None,:]
                sim_cov_sum += s[:,:,None] * s[:,None,:]
                num_steps += 1.0
                if torch.bernoulli(prob_update) > 0.0:
                    learning_rate_scaled = learning_rate/num_steps
                    num_steps = 0.0
                    h += learning_rate_scaled * (data_mu_sum - sim_mu_sum)
                    data_mu_sum[:,:] = 0.0
                    sim_mu_sum[:,:] = 0.0
                    J += learning_rate_scaled * (data_cov_sum - sim_cov_sum)
                    data_cov_sum[:,:,:] = 0.0
                    sim_cov_sum[:,:,:] = 0.0
            for t in range(num_time_points):
                node_order = torch.randperm(n=num_nodes, dtype=int_type, device=device)
                for i in range(num_nodes):
                    node_index = node_order[i]
                    deltaH = 2*(  torch.sum( J[:,:,node_index] * s, dim=-1 ) + h[:,node_index]  )*s[:,node_index]
                    prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
                    flip = torch.bernoulli(prob_accept)
                    s[:,node_index] *= ( 1.0 - 2.0*flip )
                    num_flips += torch.count_nonzero(flip)
                sim_ts[:,t,:] = s
            sim_fc = hcp.get_fc_batch(sim_ts)
            print_if_nan(sim_fc)
            fc_corr = hcp.get_triu_corr_batch(sim_fc, data_fc)
            print_if_nan(fc_corr)
            fc_corr_history[epoch,:] = fc_corr
            fc_rmse = hcp.get_triu_rmse_batch(sim_fc, data_fc)
            print_if_nan(fc_rmse)
            fc_rmse_history[epoch,:] = fc_rmse
            time_since_start = time.time() - code_start_time
            print(f'{rep:.3g}\t{epoch}\t{num_flips}\t{fc_corr.min():.3g}\t{fc_rmse.max():.3g}\t{time_since_start:.3g}')
        # Save the trained model for the subject as two files, one for J and one for h.
        file_suffix = f'data_{data_subset}_nodes_{num_nodes}_rep_{rep}_epochs_{num_epochs}_p_{prob_update.item()}_lr_{learning_rate}_threshold_{threshold}_start_{subjects_start}_end_{subjects_end}'
        print('saving Ising model J and h...')
        J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
        torch.save(J, J_file_name)
        h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
        torch.save(h, h_file_name)
        # Save the FC correlation histories and FC RMSE histories from the model training sessions.
        print('saving FC correlation and RMSE matrices...')
        fc_corr_file_name = os.path.join(output_directory, f'fc_corr_{file_suffix}.pt')
        torch.save(fc_corr_history, fc_corr_file_name)
        fc_rmse_file_name = os.path.join(output_directory, f'fc_rmse_{file_suffix}.pt')
        torch.save(fc_rmse_history, fc_rmse_file_name)
    code_end_time = time.time()
    print('done, time', code_end_time-code_start_time)