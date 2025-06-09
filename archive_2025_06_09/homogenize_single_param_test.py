import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_mean_std_1', help="part of the data mean state and state product file names between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-e", "--model_file_fragment", type=str, default='all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
    parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
    parser.add_argument("-g", "--combine_scans", action='store_true', default=False, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-i", "--homogenize_h", action='store_true', default=False, help="Set this flag to homogenize h of one region at a time.")
    parser.add_argument("-j", "--homogenize_J", action='store_true', default=False, help="Set this flag to homogenize J of one region pair at a time.")
    parser.add_argument("-k", "--use_abs", action='store_true', default=False, help="Set this flag to take the absolute value of the priority Tensor.")
    parser.add_argument("-l", "--descend", action='store_true', default=False, help="Set this flag to test the values in descending instead of ascending order.")
    parser.add_argument("-m", "--h_priorities_file", type=str, default='h_myelination_corr_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000.pt', help="file name for values to use for prioritizing h to test")
    parser.add_argument("-n", "--J_priorities_file", type=str, default='J_sc_corr_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000.pt', help="file name for values to use for prioritizing J to test")
    parser.add_argument("-o", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-p", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    model_file_fragment = args.model_file_fragment
    print(f'model_file_fragment={model_file_fragment}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')
    homogenize_h = args.homogenize_h
    print(f'homogenize_h={homogenize_h}')
    homogenize_J = args.homogenize_J
    print(f'homogenize_J={homogenize_J}')
    use_abs = args.use_abs
    print(f'use_abs={use_abs}')
    descend = args.descend
    print(f'descend={descend}')
    h_priorities_file = args.h_priorities_file
    print(f'h_priorities_file={h_priorities_file}')
    J_priorities_file = args.J_priorities_file
    print(f'J_priorities_file={J_priorities_file}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')

    def load_data_means(data_file_name_part:str):
        print('loading data time series state and state product means')
        state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
        target_state_mean = torch.load(state_mean_file)
        # On load, the dimensions of target_state_mean should be subject x node or scan x subject x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded state_mean with size', target_state_mean.size() )
        state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
        target_state_product_mean = torch.load(state_product_mean_file)
        # On load, the dimensions of target_state_product_mean should be subject x node-pair, subject x node x node, scan x subject x node-pair, or scan x subject x node x node.
        print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', target_state_product_mean.size() )
        # Assume that either both target Tensors have a scan dimension, or neither does.
        # If they have a scan dimension, then first remove it, either by averaging over scans or flattening together the subject and scan dimensions.
        state_mean_size = target_state_mean.size()
        # target_state_product_mean_size = target_state_product_mean.size()
        num_batch_dims = len(state_mean_size) - 1
        if num_batch_dims > 1:
            if combine_scans:
                extra_dim_range = tuple( range(num_batch_dims-1) )# average over all extra batch dimensions
                print( 'averaging over extra batch dimensions', extra_dim_range )
                target_state_mean = torch.mean(target_state_mean, dim=extra_dim_range, keepdim=False)
                target_state_product_mean = torch.mean(target_state_product_mean, dim=extra_dim_range, keepdim=False)
            else:
                print('flattening extra batch dimensions')
                target_state_mean = torch.flatten(target_state_mean, start_dim=0, end_dim=-2)
                target_state_product_mean = torch.flatten(target_state_product_mean, start_dim=0, end_dim=-3)
        # Regardless of whether the data originally had a scan dimension, we add in a singleton model replica dimension so that we can broadcast with h and J.
        print('prepending singleton model replica dimension')
        target_state_mean = target_state_mean.unsqueeze(dim=0)
        target_state_product_mean = target_state_product_mean.unsqueeze(dim=0)
        target_cov = isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
        target_fc = isingmodellight.get_fc_binary(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
        # This was getting too complicated.
        # Just assume we are reading in batches of square matrices.
        # # We want to work with the mean state products as square matrices, not upper triangular part vectors.
        # if len( state_product_mean.size() ) < 4:
        #     print('converting mean state products from upper triangular parts to square matrices')
        #     state_product_mean = isingmodellight.triu_to_square_pairs(triu_pairs=state_product_mean, diag_fill=0)
        return target_state_mean, target_state_product_mean, target_cov, target_fc
    
    def test_model_and_save(model:IsingModelLight, sim_file_fragment:str, target_state_mean:torch.Tensor, target_state_product_mean:torch.Tensor, target_cov:torch.Tensor, target_fc:torch.Tensor):
        sim_state_mean, sim_state_product_mean, flip_rate = model.simulate_and_record_means_and_flip_rate_pmb(num_steps=sim_length)
        print( 'sim_state_mean size', sim_state_mean.size() )
        print( 'sim_state_product_mean size', sim_state_product_mean.size() )
        print( 'flip_rate size', flip_rate.size() )
        print( f'time {time.time()-code_start_time:.3f}, done simulating {sim_length} steps' )
        print(f'flip rate  min {flip_rate.min():.3g}, mean {flip_rate.mean():.3g}, max {flip_rate.max():.3g}')
        flip_rate_file = os.path.join(output_directory, f'flip_rate_{sim_file_fragment}.pt')
        torch.save(obj=flip_rate, f=flip_rate_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file}')
        state_mean_rmse = isingmodellight.get_pairwise_rmse(mat1=sim_state_mean, mat2=target_state_mean)
        print(f'state mean RMSE  min {state_mean_rmse.min():.3g}, mean {state_mean_rmse.mean():.3g}, max {state_mean_rmse.max():.3g}')
        state_mean_rmse_file = os.path.join(output_directory, f'state_mean_rmse_{sim_file_fragment}.pt')
        torch.save(obj=state_mean_rmse, f=state_mean_rmse_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {state_mean_rmse_file}')
        state_product_mean_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_state_product_mean, mat2=target_state_product_mean)
        print(f'state product mean RMSE  min {state_product_mean_rmse.min():.3g}, mean {state_product_mean_rmse.mean():.3g}, max {state_product_mean_rmse.max():.3g}')
        state_product_mean_rmse_file = os.path.join(output_directory, f'state_product_mean_rmse_{sim_file_fragment}.pt')
        torch.save(obj=state_product_mean_rmse, f=state_product_mean_rmse_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {state_product_mean_rmse_file}')
        sim_cov = isingmodellight.get_cov(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
        # sim_cov_file = os.path.join(output_directory, f'sim_cov_{sim_file_fragment}.pt')
        # torch.save(obj=sim_cov, f=sim_cov_file)
        # print(f'time {time.time()-code_start_time:.3f}, saved {sim_cov_file}')
        cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_cov, mat2=target_cov)
        print(f'covariance RMSE  min {cov_rmse.min():.3g}, mean {cov_rmse.mean():.3g}, max {cov_rmse.max():.3g}')
        cov_rmse_file = os.path.join(output_directory, f'cov_rmse_{sim_file_fragment}.pt')
        torch.save(obj=cov_rmse, f=cov_rmse_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {cov_rmse_file}')
        sim_fc = isingmodellight.get_fc_binary(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
        fc_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_fc, mat2=target_fc)
        print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}')
        fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{sim_file_fragment}.pt')
        torch.save(obj=fc_rmse, f=fc_rmse_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file}')
        fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc)
        print(f'FC correlation min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}')
        fc_corr_file = os.path.join(output_directory, f'fc_corr_{sim_file_fragment}.pt')
        torch.save(obj=fc_corr, f=fc_corr_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {fc_corr_file}')
        return sim_state_mean, sim_state_product_mean, flip_rate
    
    original_sim_file_fragment = f'{model_file_fragment}_test_length_{sim_length}'

    print('loading target data time series state and state product means')
    target_state_mean, target_state_product_mean, target_cov, target_fc = load_data_means(data_file_name_part=data_file_name_part)
    print( f'time {time.time()-code_start_time:.3f}, target_state_mean size', target_state_mean.size() )
    print( f'time {time.time()-code_start_time:.3f}, target_state_product_mean size', target_state_product_mean.size() )
    model_file = os.path.join(data_directory, f'ising_model_light_{model_file_fragment}.pt')
    model = torch.load(f=model_file, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file}' )

    print(f'time {time.time()-code_start_time:.3f}, testing model as-is...')
    sim_state_mean_original, sim_state_product_mean_original, flip_rate_original = test_model_and_save(model=model, sim_file_fragment=original_sim_file_fragment, target_state_mean=target_state_mean, target_state_product_mean=target_state_product_mean, target_cov=target_cov, target_fc=target_fc)
    
    if homogenize_h:
        original_h = model.h.clone()
        h_priorities_file_path = os.path.join(output_directory, h_priorities_file)
        h_priorities = torch.load(h_priorities_file_path, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {h_priorities_file_path}, size', h_priorities.size() )
        if use_abs:
            h_priorities = h_priorities.abs()
        h_priorities_sorted, h_node_order = torch.sort(input=h_priorities, descending=descend)
        for node_index in h_node_order:
            model.h[:,:,:] = original_h[:,:,:]
            std_h, mean_h = torch.std_mean(model.h[:,training_subject_start:training_subject_end,node_index])
            model.h[:,:,node_index] = mean_h
            modified_sim_file_fragment = f'{original_sim_file_fragment}_hom_h_node_{node_index}'
            print(f'time {time.time()-code_start_time:.3f}, testing model with homogenized node {node_index}, priority {h_priorities[node_index]:.3g}, h mean {mean_h:.3g}, SD {std_h:.3g}...')
            test_model_and_save(model=model, sim_file_fragment=modified_sim_file_fragment, target_state_mean=target_state_mean, target_state_product_mean=target_state_product_mean, target_cov=target_cov, target_fc=target_fc)
    
    if homogenize_J:
        original_J = model.J.clone()
        J_priorities_file_path = os.path.join(output_directory, J_priorities_file)
        J_priorities = torch.load(J_priorities_file_path, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {J_priorities_file_path}, size', J_priorities.size() )
        if use_abs:
            J_priorities = J_priorities.abs()
        J_priorities_sorted, J_node_order = torch.sort(input=J_priorities, descending=descend)
        num_nodes = model.h.size(dim=-1)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products(num_nodes=num_nodes, device=model.J.device)
        for pair_index in J_node_order:
            node_i = triu_rows[pair_index]
            node_j = triu_cols[pair_index]
            model.J[:,:,:,:] = original_J[:,:,:,:]
            std_J, mean_J = torch.std_mean(model.J[:,training_subject_start:training_subject_end,node_i,node_j])
            model.J[:,:,node_i,node_j] = mean_J
            model.J[:,:,node_j,node_i] = mean_J
            modified_sim_file_fragment = f'{original_sim_file_fragment}_hom_J_pair_{pair_index}'
            print(f'time {time.time()-code_start_time:.3f}, testing model with homogenized node pair {pair_index} ({node_i}, {node_j}) priority {J_priorities[pair_index]:.3g}, J mean {mean_J:.3g}, SD {std_J:.3g}...')
            test_model_and_save(model=model, sim_file_fragment=modified_sim_file_fragment, target_state_mean=target_state_mean, target_state_product_mean=target_state_product_mean, target_cov=target_cov, target_fc=target_fc)
    
    print(f'time {time.time()-code_start_time:.3f}, done')