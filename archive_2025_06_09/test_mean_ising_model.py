import os
import torch
import time
import argparse

with torch.no_grad():

    start_time = time.time()
    last_time = start_time

    parser = argparse.ArgumentParser(description="Compare the per-subject means of trained Ising models to the originals.")

    # directories
    parser.add_argument("-d", "--fmri_data_dir", type=str, default='E:\\HCP_data', help="directory containing the fMRI time series data file")
    parser.add_argument("-i", "--ising_model_dir", type=str, default='E:\\Ising_model_results_daai', help="directory containing the fitted Ising model h parameter file")
    parser.add_argument("-v", "--stats_dir", type=str, default="E:\\Ising_model_results_daai", help="directory to which to write the calculated correlations and RMSEs")

    # hyperparameters of the Ising model, used for looking up which h files to load
    parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes in Ising model")
    parser.add_argument("-m", "--num_reps", type=int, default=10, help="number of Ising models trained for each subject")
    parser.add_argument("-o", "--num_epochs_ising", type=int, default=200, help="number of epochs for which we trained the Ising model")
    parser.add_argument("-p", "--prob_update", type=str, default='0.019999999552965164', help="probability of updating the model parameters on any given step used when training Ising model")
    parser.add_argument("-j", "--learning_rate_ising", type=str, default='0.001', help="learning rate used when training Ising model")
    parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean, in the Ising model")
    parser.add_argument("-a", "--beta", type=float, default=0.5, help="conversion factor between delta-H and log probability of flipping in the Ising model")

    args = parser.parse_args()

    anti_nan_offset = float(10**-10)

    fmri_data_dir = args.fmri_data_dir
    ising_model_dir = args.ising_model_dir
    stats_dir = args.stats_dir

    num_nodes = args.num_nodes
    num_reps = args.num_reps
    num_epochs_ising = args.num_epochs_ising
    prob_update = args.prob_update
    learning_rate_ising = args.learning_rate_ising
    threshold = args.threshold
    beta = args.beta

    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')

    ising_model_string = f'nodes_{num_nodes}_reps_{num_reps}_epochs_{num_epochs_ising}_p_{prob_update}_lr_{learning_rate_ising}_threshold_{threshold}'

    def load_ising_models(subset:str):
        if subset == 'training':
            num_subjects = 699
        else:
            num_subjects = 83
        ising_h_file = os.path.join(ising_model_dir, f'h_data_{subset}_{ising_model_string}_start_0_end_{num_subjects}.pt')
        h = torch.load(ising_h_file)
        num_reps_h, num_subjects_h, num_nodes_h = h.size()
        print(f'h: {num_reps_h} x {num_subjects_h} x {num_nodes_h}')
        ising_J_file = os.path.join(ising_model_dir, f'J_data_{subset}_{ising_model_string}_start_0_end_{num_subjects}.pt')
        J = torch.load(ising_J_file)
        num_reps_J, num_subjects_J, num_source_nodes_J, num_target_nodes_J = J.size()
        print(f'J: {num_reps_J} x {num_subjects_J} x {num_source_nodes_J} x {num_target_nodes_J}')
        return h, J
    
    def get_data_time_series_from_single_file(data_directory:str, data_subset:str, num_nodes:int, data_threshold:float=0.1, subjects_start:int=0, subjects_end:int=None):
        if not subjects_end:
            if data_subset == 'training':
                subjects_end = 699
            else:
                subjects_end = 83
        ts_file = os.path.join( data_directory, f'sc_fmri_ts_{data_subset}.pt' )
        data_ts = torch.load(ts_file)[subjects_start:subjects_end,:,:,:num_nodes]
        data_std, data_mean = torch.std_mean(data_ts, dim=-2, keepdim=True)
        data_ts -= data_mean
        data_ts /= data_std
        data_ts = data_ts.flatten(start_dim=1, end_dim=-2)
        data_ts = torch.sign(data_ts - data_threshold)
        num_subjects, num_time_points, num_nodes = data_ts.size()
        print(f'data_ts: {num_subjects} x {num_time_points} x {num_nodes}')
        return data_ts

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

    def get_fc_batch_batch(ts_batch_batch:torch.Tensor):
        # Take the FC of each individual time series.
        batch_batch_size, batch_size, _, num_rois = ts_batch_batch.size()
        fc = torch.zeros( (batch_batch_size, batch_size, num_rois, num_rois), dtype=ts_batch_batch.dtype, device=ts_batch_batch.device )
        for batch_index in range(batch_batch_size):
            fc[batch_index,:,:,:] = get_fc_batch(ts_batch_batch[batch_index,:,:,:])
        return fc

    def run_ising_model(J:torch.Tensor, h:torch.Tensor, num_time_points:int):
        num_reps, num_subjects, num_nodes = h.size()
        sim_ts = torch.zeros( (num_reps, num_subjects, num_time_points, num_nodes), dtype=float_type, device=device )
        s = 2.0*torch.randint_like( h, high=2 ) - 1.0
        for t in range(num_time_points):
            node_order = torch.randperm(n=num_nodes, dtype=int_type, device=device)
            for i in range(num_nodes):
                node_index = node_order[i]
                deltaH = 2*(  torch.sum( J[:,:,:,node_index] * s, dim=-1 ) + h[:,:,node_index]  )*s[:,:,node_index]
                prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
                s[:,:,node_index] *= ( 1.0 - 2.0*torch.bernoulli(prob_accept) )
                sim_ts[:,:,t,:] = s
        return sim_ts

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
    
    def get_triu_corr_batch_batch_vs_batch(tensor1_batch_batch:torch.Tensor, tensor2_batch:torch.Tensor):
        num_batch_batches, num_batches, _, _ = tensor1_batch_batch.size()
        corr_batch_batch = torch.zeros( (num_batch_batches, num_batches), dtype=tensor1_batch_batch.dtype, device=tensor1_batch_batch.device )
        for batch_index in range(num_batch_batches):
            for mat_index in range(num_batches):
                corr_batch_batch[batch_index, mat_index] = get_triu_corr(tensor1_batch_batch[batch_index, mat_index, :, :], tensor2_batch[mat_index, :, :]) 
        return corr_batch_batch

    # only assumes the tensors are the same shape
    def get_rmse(tensor1:torch.Tensor, tensor2:torch.Tensor):
        return torch.sqrt(  torch.mean( torch.square(tensor2 - tensor1) )  )

    # In several cases, we have symmetric square matrices with fixed values on the diagonal.
    # In particular, this is true of the functional connectivity and structural connectivity matrices.
    # For such matrices, it is more meaningful to only calculate the RMSE of the elements above the diagonal.
    def get_triu_rmse(tensor1:torch.Tensor, tensor2:torch.Tensor):
        indices = torch.triu_indices( row=tensor1.size(0), col=tensor1.size(1), offset=1, device=tensor1.device )
        indices_r = indices[0]
        indices_c = indices[1]
        return get_rmse( tensor1[indices_r,indices_c], tensor2[indices_r,indices_c] )
    
    def get_triu_rmse_batch_batch_vs_batch(tensor1_batch_batch:torch.Tensor, tensor2_batch:torch.Tensor):
        num_batch_batches, num_batches, _, _ = tensor1_batch_batch.size()
        rmse_batch_batch = torch.zeros( (num_batch_batches, num_batches), dtype=tensor1_batch_batch.dtype, device=tensor1_batch_batch.device )
        for batch_index in range(num_batch_batches):
            for mat_index in range(num_batches):
                rmse_batch_batch[batch_index, mat_index] = get_triu_rmse(tensor1_batch_batch[batch_index, mat_index, :, :], tensor2_batch[mat_index, :, :]) 
        return rmse_batch_batch
    
    def print_if_nan(mat:torch.Tensor):
        mat_is_nan = torch.isnan(mat)
        num_nan = torch.count_nonzero(mat_is_nan)
        if num_nan > 0:
            print( mat.size(), f'matrix contains {num_nan} NaNs.' )
    
    def print_min_median_max(values:torch.Tensor):
        print(f'min: {values.min().item():.3g}, median: {values.median().item():.3g}, max: {values.max().item():.3g}')
    
    def run_and_compare(subset:str):
        print('loading Ising models...')
        last_time = time.time()
        h, J = load_ising_models(subset=subset)
        current_time = time.time()

        h_mean = h.mean(dim=0, keepdim=True)
        J_mean = J.mean(dim=0, keepdim=True)

        # Load, normalize, binarize, and flatten the fMRI time series data.
        print('loading fMRI data...')
        last_time = time.time()
        data_ts = get_data_time_series_from_single_file(data_directory=fmri_data_dir, data_subset=subset, num_nodes=num_nodes, data_threshold=threshold)
        num_time_points = data_ts.size(dim=1)
        current_time = time.time()
        print('time:', current_time-last_time)
        
        data_fc = get_fc_batch(data_ts)

        print('running trained Ising models of training subjects...')
        last_time = time.time()
        sim_ts = run_ising_model(J=J, h=h, num_time_points=num_time_points)
        sim_fc = get_fc_batch_batch(sim_ts)
        print_if_nan(sim_fc)
        current_time = time.time()
        print('time:', current_time-last_time)
        
        print('running per-subject means of trained Ising models of training subjects...')
        last_time = time.time()
        sim_ts_mean = run_ising_model(J=J_mean, h=h_mean, num_time_points=num_time_points)
        sim_fc_mean = get_fc_batch_batch(sim_ts_mean)
        print_if_nan(sim_fc_mean)
        current_time = time.time()
        print('time:', current_time-last_time)
        
        print('computing correlations between sim and data FC matrices...')
        last_time = time.time()
        fc_corr = get_triu_corr_batch_batch_vs_batch(sim_fc, data_fc)
        print_if_nan(fc_corr)
        print_min_median_max(fc_corr)
        print('computing correlations between meaned model sim and data FC matrices...')
        fc_corr_mean = get_triu_corr_batch_batch_vs_batch(sim_fc_mean, data_fc)
        print_if_nan(fc_corr_mean)
        print_min_median_max(fc_corr_mean)
        print('computing RMSEs between sim and data FC matrices...')
        fc_rmse = get_triu_rmse_batch_batch_vs_batch(sim_fc, data_fc)
        print_if_nan(fc_rmse)
        print_min_median_max(fc_rmse)
        print('computing RMSEs between meaned model sim and data FC matrices...')
        fc_rmse_mean = get_triu_rmse_batch_batch_vs_batch(sim_fc_mean, data_fc)
        print_if_nan(fc_rmse_mean)
        print_min_median_max(fc_rmse_mean)
        current_time = time.time()
        print('time:', current_time-last_time)

        print('saving results...')
        file_suffix = f'data_{subset}_{ising_model_string}_mean_testing'
        fc_corr_file_name = os.path.join(stats_dir, f'fc_corr_{file_suffix}.pt')
        torch.save(fc_corr, fc_corr_file_name)
        fc_corr_mean_file_name = os.path.join(stats_dir, f'fc_corr_mean_{file_suffix}.pt')
        torch.save(fc_corr_mean, fc_corr_mean_file_name)
        fc_rmse_file_name = os.path.join(stats_dir, f'fc_rmse_{file_suffix}.pt')
        torch.save(fc_rmse, fc_rmse_file_name)
        fc_rmse_mean_file_name = os.path.join(stats_dir, f'fc_rmse_mean_{file_suffix}.pt')
        torch.save(fc_rmse_mean, fc_rmse_mean_file_name)
    
    run_and_compare('training')
    run_and_compare('validation')
    end_time = time.time()
    print('done, time', end_time-start_time)