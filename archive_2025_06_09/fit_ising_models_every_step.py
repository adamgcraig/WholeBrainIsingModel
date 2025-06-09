import os
import torch
import time
import argparse
import hcpdatautils as hcp

parser = argparse.ArgumentParser(description="Train a bunch of Ising models on fMRI time series data.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate by which we multiply updates to weights and biases at each step")
parser.add_argument("-a", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes to model")
parser.add_argument("-e", "--num_epochs", type=int, default=10, help="number of times to repeat the training time series")
parser.add_argument("-r", "--num_reps", type=int, default=5, help="number of models to train for each subject")
parser.add_argument("-d", "--data_subset", type=str, default='training', help="which data subset to use, either training, validation or testing")
parser.add_argument("-o", "--subjects_start", type=int, default=0, help="index of first subject in slice on which to train")
parser.add_argument("-x", "--subjects_end", type=int, default=10, help="index one past last subject in slice on which to train")
args = parser.parse_args()
data_directory = args.data_directory
output_directory = args.output_directory
beta = args.beta
learning_rate = args.learning_rate
beta = args.beta
threshold = args.threshold
num_nodes = args.num_nodes
num_epochs = args.num_epochs
num_reps = args.num_reps
data_subset = args.data_subset
subjects_start = args.subjects_start
subjects_end = args.subjects_end

with torch.no_grad():
    code_start_time = time.time()
    int_type = torch.int
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')

    def get_fc(ts:torch.Tensor):
        # We want to take the correlations between pairs of areas over time.
        # torch.corrcoef() assumes observations are in columns and variables in rows.
        # As such, we have to take the transpose.
        return torch.corrcoef( ts.transpose(dim0=0, dim1=1) )

    def get_fc_batch(ts_batch:torch.Tensor):
        # Take the FC of each individual time series.
        batch_size = ts_batch.size(dim=0)
        num_rois = ts_batch.size(dim=-1)
        fc = torch.zeros( (batch_size, num_rois, num_rois), dtype=ts_batch.dtype, device=ts_batch.device )
        for ts_index in range(batch_size):
            fc[ts_index,:,:] = get_fc(ts_batch[ts_index,:,:])
        return fc

    # only assumes the tensors are the same shape
    def get_rmse(tensor1:torch.Tensor, tensor2:torch.Tensor):
        return torch.sqrt(  torch.mean( torch.square(tensor2 - tensor1) )  )

    # takes the mean over all but the first dimension
    # For our 3D batch-first tensors, this gives us a 1D tensor of RMSE values, one for each batch.
    def get_rmse_batch(tensor1:torch.Tensor, tensor2:torch.Tensor):
        dim_indices = tuple(   range(  len( tensor1.size() )  )   )
        return torch.sqrt(  torch.mean( torch.square(tensor2 - tensor1), dim=dim_indices[1:] )  )

    # In several cases, we have symmetric square matrices with fixed values on the diagonal.
    # In particular, this is true of the functional connectivity and structural connectivity matrices.
    # For such matrices, it is more meaningful to only calculate the RMSE of the elements above the diagonal.
    def get_triu_rmse(tensor1:torch.Tensor, tensor2:torch.Tensor):
        indices = torch.triu_indices( row=tensor1.size(0), col=tensor1.size(1), offset=1, device=tensor1.device )
        indices_r = indices[0]
        indices_c = indices[1]
        return get_rmse( tensor1[indices_r,indices_c], tensor2[indices_r,indices_c] )

    def get_triu_rmse_batch(tensor1:torch.Tensor, tensor2:torch.Tensor):
        indices = torch.triu_indices( tensor1.size(-2), tensor1.size(-1), 1 )
        indices_r = indices[0]
        indices_c = indices[1]
        batch_size = tensor1.size(dim=0)
        num_triu_elements = indices.size(dim=1)
        dtype = tensor1.dtype
        device = tensor1.device
        triu1 = torch.zeros( (batch_size, num_triu_elements), dtype=dtype, device=device )
        triu2 = torch.zeros_like(triu1)
        for b in range(batch_size):
            triu1[b,:] = tensor1[b,indices_r,indices_c]
            triu2[b,:] = tensor2[b,indices_r,indices_c]
        return get_rmse_batch(triu1, triu2)

    def get_triu_corr(tensor1:torch.Tensor, tensor2:torch.Tensor):
        indices = torch.triu_indices( row=tensor1.size(0), col=tensor1.size(1), offset=1, device=tensor1.device )
        indices_r = indices[0]
        indices_c = indices[1]
        tensor1_triu = tensor1[indices_r,indices_c].unsqueeze(dim=0)
        # print( tensor1_triu.size() )
        tensor2_triu = tensor2[indices_r,indices_c].unsqueeze(dim=0)
        # print( tensor2_triu.size() )
        tensor_pair_triu = torch.cat( (tensor1_triu,tensor2_triu), dim=0 )
        # print( tensor_pair_triu.size() )
        corr = torch.corrcoef(tensor_pair_triu)
        # print(corr)
        return corr[0,1]

    def get_triu_corr_batch(tensor1:torch.Tensor, tensor2:torch.Tensor):
        indices = torch.triu_indices( tensor1.size(-2), tensor1.size(-1), 1 )
        indices_r = indices[0]
        indices_c = indices[1]
        batch_size = tensor1.size(dim=0)
        dtype = tensor1.dtype
        device = tensor1.device
        corr_batch = torch.zeros( (batch_size,), dtype=dtype, device=device )
        for b in range(batch_size):
            triu_pair = torch.stack( (tensor1[b,indices_r,indices_c], tensor2[b,indices_r,indices_c]) )
            pair_corr = torch.corrcoef(triu_pair)
            corr_batch[b] = pair_corr[0,1]
        return corr_batch
    
    def get_data_time_series_from_single_file(data_directory:str, data_subset:str, subjects_start:int, subjects_end:int, num_nodes:int, data_threshold:float):
        ts_file = os.path.join( data_directory, f'sc_fmri_ts_{data_subset}.pt' )
        data_ts = torch.load(ts_file)[subjects_start:subjects_end,:,:,:num_nodes]
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

    # Load, normalize, binarize, and flatten the fMRI time series data.
    print('loading fMRI data...')
    # training_subjects = load_training_subjects(directory_path=data_directory)
    # data_fc, data_cov, data_mu = get_data_time_series_stats(data_directory=data_directory, subjects=training_subjects, time_series_per_subject=time_series_per_subject, steps_per_time_series=num_time_points, num_nodes=num_nodes, data_threshold=threshold, window_length=num_steps)
    data_ts = get_data_time_series_from_single_file(data_directory=data_directory, data_subset=data_subset, subjects_start=subjects_start, subjects_end=subjects_end, num_nodes=num_nodes, data_threshold=threshold)
    num_subjects, num_time_points, num_nodes = data_ts.size()
    print('loaded with subjects x time points x nodes: ', num_subjects, num_time_points, num_nodes)
    data_fc = get_fc_batch(data_ts)

    # Use -2.0 as a placeholder value, since neither the correlation (restricted to [-1, +1]) nor the RMSE (restricted to [0, 2]) can take on this value.
    fc_corr_history = torch.full( (num_epochs, num_subjects), -2.0, dtype=float_type, device=device )
    fc_rmse_history = torch.full( (num_epochs, num_subjects), -2.0, dtype=float_type, device=device )

    # Pre-allocate space for our time series.
    sim_ts = torch.zeros_like(data_ts)
    # Pre-allocate space for running totals of mean and covariance.

    # Train the model by
    # running it for a few steps at a time,
    # finding the means and covariances of individual regions in the simulation,
    # comparing them to the means and covariances in a window of the fMRI data,
    # adjusting h in proportion to the difference in means,
    # adjusting J in proportion to the difference in covariances,
    # repeating for all windows in the time series,
    # and repeating the full time series for a set number of epochs.
    print('training Ising model...')
    print('rep\tepoch\tworst corr.\tworst RMSE\ttime')
    for rep in range(num_reps):    # Initialize the model and run a sim prior to training to establish a baseline for improvement.
        # print('initializing Ising model for rep', rep, '...')
        J, h, s = get_batched_ising_models(batch_size=num_subjects, num_nodes=num_nodes)
        for epoch in range(num_epochs):
            for t in range(num_time_points):
                node_order = torch.randperm(n=num_nodes, dtype=int_type, device=device)
                for i in range(num_nodes):
                    node_index = node_order[i]
                    deltaH = 2*(  torch.sum( J[:,:,node_index] * s, dim=-1 ) + h[:,node_index]  )*s[:,node_index]
                    prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
                    s[:,node_index] *= ( 1.0 - 2.0*torch.bernoulli(prob_accept) )
                sim_ts[:,t,:] = s
                d = data_ts[:,t,:]
                h += learning_rate * (d - s)
                J += learning_rate * (d[:,:,None] * d[:,None,:] - s[:,:,None] * s[:,None,:])
            for t in range(num_time_points):
                node_order = torch.randperm(n=num_nodes, dtype=int_type, device=device)
                for i in range(num_nodes):
                    node_index = node_order[i]
                    deltaH = 2*(  torch.sum( J[:,:,node_index] * s, dim=-1 ) + h[:,node_index]  )*s[:,node_index]
                    prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
                    s[:,node_index] *= ( 1.0 - 2.0*torch.bernoulli(prob_accept) )
                sim_ts[:,t,:] = s
            sim_fc = get_fc_batch(sim_ts)
            print_if_nan(sim_fc)
            fc_corr = get_triu_corr_batch(sim_fc, data_fc)
            print_if_nan(fc_corr)
            fc_corr_history[epoch,:] = fc_corr
            fc_rmse = get_triu_rmse_batch(sim_fc, data_fc)
            print_if_nan(fc_rmse)
            fc_rmse_history[epoch,:] = fc_rmse
            time_since_start = time.time() - code_start_time
            print(f'{rep:.3g}\t{epoch}\t{fc_corr.min():.3g}\t{fc_rmse.max():.3g}\t{time_since_start:.3g}')
        # Save the trained model for the subject as two files, one for J and one for h.
        file_suffix = f'data_{data_subset}_nodes_{num_nodes}_rep_{rep}_epochs_{num_epochs}_every_step_lr_{learning_rate}_threshold_{threshold}_start_{subjects_start}_end_{subjects_end}'
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