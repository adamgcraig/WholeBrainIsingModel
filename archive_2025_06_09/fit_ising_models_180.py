import os
import numpy as np
import torch
import time

with torch.no_grad():
    code_start_time = time.time()
    int_type = torch.int
    float_type = torch.float
    # device = torch.device('cuda')
    device = torch.device('cpu')

    # There are always 360 brain areas and 4 features per area.
    # There are usually 1200 time points in a data time series.
    # There are usually 4 time series per subject.
    # Some are missing 2_LR and 2_RL.
    # For the "get_..._file_path()" functions,
    # we rely on the consistent naming convention we set up
    # and assume directory_path ends in the correct path separator character, '/' or '\'.
    num_brain_areas = 360
    num_time_points = 1200
    num_nodes = 180# Choose this many brain regions at a time to model.
    time_series_suffixes = ['1_LR', '1_RL', '2_LR', '2_RL']
    time_series_per_subject = len(time_series_suffixes)
    data_directory = 'E:\HCP_data'
    output_directory = f'E:\Ising_model_results_{num_nodes}'
    beta = 0.5# Factor we can use to adjust the chance of accepting a flip that increases the energy.
    num_node_combinations = 10# Choose this many combinations of num_nodes brain regions to try successively.
    num_starting_conditions = 10# For each combination, try this many independent runs to check how variable the results are.
    anti_nan_offset = float(10**-10)
    num_steps = 50
    max_num_epochs = 500# maximum number of epochs for which to train a given model
    max_epochs_since_best = 10# Stop early if we go this many epochs without a new minimum FC RMSE.
    learning_rate = 0.001
    data_threshold = 0.1# threshold to use for binarizing fMRI data

    def load_subject_list(file_path:str):
        with open(file_path, 'r', encoding='utf-8') as id_file:
            subject_list = list(  map( int, id_file.read().split() )  )
            return subject_list

    def load_training_subjects(directory_path:str):
        return load_subject_list( os.path.join(directory_path, 'training_subject_ids.txt') )

    def load_validation_subjects(directory_path:str):
        return load_subject_list( os.path.join(directory_path, 'validation_subject_ids.txt') )

    def load_testing_subjects(directory_path:str):
        return load_subject_list( os.path.join(directory_path, 'testing_subject_ids.txt') )

    def get_structural_connectivity_file_path(directory_path:str, subject_id:int):
        return os.path.join(directory_path, 'dtMRI_binaries', f"sc_{subject_id}.bin")

    def get_time_series_file_path(directory_path:str, subject_id:int, time_series_suffix:str):
        return os.path.join(directory_path, 'fMRI_ts_binaries', f"ts_{subject_id}_{time_series_suffix}.bin")

    def get_has_sc_subject_list(directory_path:str, subject_list:list):
        return list(  filter(lambda subject_id: os.path.isfile( get_structural_connectivity_file_path(directory_path, subject_id) ), subject_list)  )

    # Load the files in the more typical Python way, where the index of the last (column) dimension changes fastest.
    # We store data matrices as sequences of 64-bit floating point numbers.
    # Specifically, each consists of some number of 360-number blocks where consecutive elements are for different ROIs.
    # As such, we need to specify the number of columns as 360 in order to convert back to a 2D matrix.
    # If device is None, return a numpy array.
    # Otherwise, return a PyTorch tensor.
    def load_matrix_from_binary(file_path:str, dtype=float_type, device='cpu', num_cols:int=num_brain_areas):
        data_matrix = np.fromfile(file_path, np.float64).reshape( (-1, num_cols), order='C' )
        return torch.from_numpy(data_matrix).to(device, dtype=dtype)

    def load_all_time_series_for_subject(directory_path:str, subject_id:int, dtype=float_type, device='cpu', num_cols:int=num_brain_areas):
        time_series = torch.zeros( (time_series_per_subject, num_time_points, num_cols), dtype=dtype, device=device )
        for ts_index in range(time_series_per_subject):
            ts_suffix = time_series_suffixes[ts_index]
            file_path = get_time_series_file_path(directory_path=directory_path, subject_id=subject_id, time_series_suffix=ts_suffix)
            ts = load_matrix_from_binary(file_path=file_path, dtype=dtype, device=device, num_cols=num_cols)
            # Some time series have fewer than the standard number of time points, but none have more.
            # All of the ones in our selected training, validation, and testing sets have the standard number.
            actual_num_time_points = ts.size(dim=0)
            time_series[ts_index,:actual_num_time_points,:] = ts
        return time_series

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

    def get_fcd(ts:torch.Tensor, window_length:int=90, window_step:int=1):
        # Calculate functional connectivity dynamics.
        # Based on Melozzi, Francesca, et al. "Individual structural features constrain the mouse functional connectome." Proceedings of the National Academy of Sciences 116.52 (2019): 26961-26969.
        dtype = ts.dtype
        device = ts.device
        ts_size = ts.size()
        T = ts_size[0]
        R = ts_size[1]
        triu_index = torch.triu_indices(row=R, col=R, offset=1, dtype=torch.int, device=device)
        triu_row = triu_index[0]
        triu_col = triu_index[1]
        num_indices = triu_index.size(dim=1)
        left_window_margin = window_length//2 + 1
        right_window_margin = window_length - left_window_margin
        window_centers = torch.arange(start=left_window_margin, end=T-left_window_margin+1, step=window_step, dtype=torch.int, device=device)
        window_offsets = torch.arange(start=-right_window_margin, end=left_window_margin, dtype=torch.int, device=device)
        num_windows = window_centers.size(dim=0)
        window_fc = torch.zeros( (num_windows, num_indices), dtype=dtype, device=device )
        for c in range(num_windows):
            fc = get_fc(ts[window_offsets+window_centers[c],:])
            window_fc[c,:] = fc[triu_row, triu_col]
        return torch.corrcoef(window_fc)

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

    def init_state(num_nodes:int_type=num_brain_areas):
        s = torch.randint( 2, (num_nodes,), dtype=float_type, device=device )
        s = 2*s - 1
        return s

    def metropolis_sim(J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, num_steps:int_type):
        num_nodes = s.size(dim=0)
        sim_ts = torch.zeros( (num_steps+1, num_nodes), dtype=float_type, device=device )
        # Pre-allocate our random numbers.
        choice_rand = torch.rand( (num_steps, num_nodes), dtype=float_type, device=device )
        J_no_diag = J - torch.diag_embed( torch.diagonal(J) )
        for t in range(num_steps):
            node_order = torch.randperm(n=num_nodes, dtype=int_type, device=device)
            for i in range(num_nodes):
                node_index = node_order[i]
                deltaH = 2*(  torch.dot( J_no_diag[node_index,:], s ) + h[node_index]  )*s[node_index]
                p_accept = torch.exp(-beta*deltaH)
                cr = choice_rand[t,node_index]
                s[node_index] *= torch.sign( cr - p_accept - (cr==p_accept).float() )# If it is equal, subtract 1 so that we accept instead of multiplying by 0.
            sim_ts[t,:] = s
        sim_ts[-1,:] = -1*sim_ts[-2,:]
        return sim_ts

    def replace_diag(M:torch.Tensor, new_diagonal_elements:torch.Tensor):
        return M - torch.diag_embed( torch.diagonal(M) ) + torch.diag_embed(new_diagonal_elements)

    # Generate the target model means and covariances.
    def get_time_series_cov_mu(time_series:torch.Tensor):
        cov = torch.mean( time_series[:,:,None] * time_series[:,None,:], dim=0 )
        mu = torch.mean(time_series, dim=0)
        return cov, mu

    training_subjects = load_training_subjects(directory_path=data_directory)
    num_subjects = len(training_subjects)
    # Use -2.0 as a placeholder value, since neither the correlation (restricted to [-1, +1]) nor the RMSE (restricted to [0, 2]) can take on this value.
    fc_corr_history = torch.full( (num_subjects, num_node_combinations, num_starting_conditions, max_num_epochs), -2.0, dtype=float_type, device=device )
    fc_rmse_history = torch.full( (num_subjects, num_node_combinations, num_starting_conditions, max_num_epochs), -2.0, dtype=float_type, device=device )
    for subject_index in range(num_subjects):

        subject_id = training_subjects[subject_index]
        print(f'subject {subject_id}, {subject_index+1} of {num_subjects}')

        # Load, normalize, binarize, and flatten the fMRI time series data.
        print('loading fMRI data...')
        data_ts = load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject_id, device=device)
        # Normalize over each time series separately, since they may be on different scales or have different baselines.
        data_std, data_mean = torch.std_mean(data_ts,dim=1,keepdim=True)
        data_ts_norm = (data_ts - data_mean)/data_std
        data_ts_binary = torch.sign(data_ts_norm - data_threshold)
        data_ts_flat = data_ts_binary.flatten(start_dim=0, end_dim=-2)
        data_length = data_ts_flat.size(dim=0)

        # Find the means and covariances for individual windows of the data.
        print('finding fMRI window means and covariances...')
        data_fc = get_fc(data_ts_flat)
        num_windows = data_length//num_steps
        data_cov = torch.zeros( (num_windows, num_brain_areas, num_brain_areas), dtype=float_type, device=device )
        data_mu = torch.zeros( (num_windows, num_brain_areas), dtype=float_type, device=device )
        for window in range(num_windows):
            data_start = window*num_steps
            data_end = min(data_start + num_steps, data_length)
            dc, dm = get_time_series_cov_mu(data_ts_flat[data_start:data_end,:])
            data_cov[window,:,:] = dc
            data_mu[window,:] = dm
        
        for node_combination_index in range(num_node_combinations):
            # Randomly select num_nodes regions, and extract the relevant FC, mean, and covariance values from the data.
            node_order = torch.randperm(n=num_brain_areas, dtype=int_type, device=device)
            selected_nodes = node_order[:num_nodes]
            data_fc_nodes = data_fc[selected_nodes,:]
            data_fc_nodes = data_fc_nodes[:,selected_nodes]
            data_cov_nodes = data_cov[:,selected_nodes,:]
            data_cov_nodes = data_cov_nodes[:,:,selected_nodes]
            data_mu_nodes = data_mu[:,selected_nodes]

            for starting_condition_index in range(num_starting_conditions):
                # Initialize the model and run a sim prior to training to establish a baseline for improvement.
                print('initializing Ising model...')
                J = torch.randn( (num_nodes, num_nodes), dtype=float_type, device=device )
                J = 0.5*( J + J.transpose(dim0=0, dim1=1) )
                h = torch.zeros( (num_nodes,), dtype=float_type, device=device )
                s = init_state(num_nodes=num_nodes)

                print('doing a test run...')
                s_init = s.clone()
                sim_ts = metropolis_sim(J, h, s, data_length)
                # num_changed = torch.count_nonzero(s_init != s).item()
                # print('num different values between current and initial state =', num_changed)
                sim_fc = get_fc(sim_ts)
                fc_corr = get_triu_corr(sim_fc, data_fc).item()
                fc_rmse = get_triu_rmse(sim_fc, data_fc).item()
                print(f'initial FC correlation = {fc_corr:.3g}, RMSE = {fc_rmse:.3g}')
                
                # Train the model by
                # running it for a few steps at a time,
                # finding the means and covariances of individual regions in the simulation,
                # comparing them to the means and covariances in a window of the fMRI data,
                # adjusting h in proportion to the difference in means,
                # adjusting J in proportion to the difference in covariances,
                # repeating for all windows in the time series,
                # and repeating the full time series for a set number of epochs.
                print('training Ising model...')
                print('time\tsubject\tnode combo\tstart cond.\tepoch\tcorr.\tRMSE')
                epochs_since_best = 0
                best_rmse = 2.0# maximum possible value, since individual FC elements are in [-1, +1].
                for epoch in range(max_num_epochs):
                    for window in range(num_windows):
                        sim_ts = metropolis_sim(J, h, s, num_steps)
                        sim_cov, sim_mu = get_time_series_cov_mu(sim_ts)
                        J -= learning_rate * (sim_cov - data_cov_nodes[window,:,:])
                        h -= learning_rate * (sim_mu - data_mu_nodes[window,:])
                    sim_ts = metropolis_sim(J, h, s, data_length)
                    sim_fc = get_fc(sim_ts)
                    fc_corr = get_triu_corr(sim_fc, data_fc_nodes)
                    fc_corr_history[subject_index, node_combination_index, starting_condition_index, epoch] = fc_corr
                    fc_rmse = get_triu_rmse(sim_fc, data_fc_nodes)
                    fc_rmse_history[subject_index, node_combination_index, starting_condition_index, epoch] = fc_rmse
                    time_since_start = time.time() - code_start_time
                    print(f'{time_since_start:.3g}\t{subject_index}\t{node_combination_index}\t{starting_condition_index}\t{epoch}\t{fc_corr.item():.3g}\t{fc_rmse.item():.3g}')
                    if fc_rmse < best_rmse:
                        epochs_since_best = 0
                        best_rmse = fc_rmse
                    else:
                        epochs_since_best += 1
                    if epochs_since_best > max_epochs_since_best:
                        print(f'It has been {epochs_since_best} epochs since the best FC RMSE so far {best_rmse:.3g}, so quitting training.')
                        break
                
                # Save the trained model for the subject as two files, one for J and one for h.
                print('saving Ising model J and h and node indices...')
                J_file_name = os.path.join(output_directory, 'J', f'J_s_{subject_id}_nc_{node_combination_index}_sc_{starting_condition_index}.pt')
                torch.save(J, J_file_name)
                h_file_name = os.path.join(output_directory, 'h', f'h_s_{subject_id}_nc_{node_combination_index}_sc_{starting_condition_index}.pt')
                torch.save(h, h_file_name)
                nodes_file_name = os.path.join(output_directory, 'node_indices', f'nodes_s_{subject_id}_nc_{node_combination_index}_sc_{starting_condition_index}.pt')
                torch.save(selected_nodes, nodes_file_name)
    # Save the FC correlation histories and FC RMSE histories from the model training sessions.
    print('saving FC correlation and RMSE matrices...')
    fc_corr_file_name = os.path.join(output_directory, 'fc_corr.pt')
    torch.save(fc_corr_history, fc_corr_file_name)
    fc_rmse_file_name = os.path.join(output_directory, 'fc_rmse.pt')
    torch.save(fc_rmse_history, fc_rmse_file_name)
    code_end_time = time.time()
    print('done, time', code_end_time-code_start_time)