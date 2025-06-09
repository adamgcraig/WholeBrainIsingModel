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

    parser = argparse.ArgumentParser(description="Simulate one Ising model with several different beta values while tracking flip rate and covariance RMSE.")
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_as_is', help="part of the output file name between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--output_file_name_part", type=str, default='short', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
    parser.add_argument("-s", "--subject_index", type=int, default=0, help="individual subject to select")
    parser.add_argument("-e", "--models_per_subject", type=int, default=10000, help="number of separate models of each subject")
    parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
    parser.add_argument("-m", "--center_cov", action='store_true', default=False, help="Set this flag in order to initialize J to the centered covariance instead of the uncentered covariance (state mean product).")
    parser.add_argument("-n", "--use_inverse_cov", action='store_true', default=False, help="Set this flag in order to initialize J to the inverse covariance instead of the covariance.")
    parser.add_argument("-o", "--combine_scans", action='store_true', default=False, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-p", "--min_beta", type=float, default=10e-10, help="low end of initial beta search interval")
    parser.add_argument("-q", "--max_beta", type=float, default=1.0, help="high end of initial beta search interval")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    output_file_name_part = args.output_file_name_part
    print(f'output_file_name_part={output_file_name_part}')
    subject_index = args.subject_index
    print(f'subject_index={subject_index}')
    models_per_subject = args.models_per_subject
    print(f'models_per_subject={models_per_subject}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
    center_cov = args.center_cov
    print(f'center_cov={center_cov}')
    use_inverse_cov = args.use_inverse_cov
    print(f'use_inverse_cov={use_inverse_cov}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')
    min_beta = args.min_beta
    print(f'min_beta={min_beta}')
    max_beta = args.max_beta
    print(f'min_beta={max_beta}')

    print('loading data time series state and state product means')
    target_state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
    target_state_mean = torch.load(target_state_mean_file)
    # On load, the dimensions of target_state_mean should be scan x subject x node.
    print( f'time {time.time()-code_start_time:.3f}, loaded target_state_mean with size', target_state_mean.size() )
    target_state_mean = target_state_mean[:,subject_index,:]
    if combine_scans:
        target_state_mean = torch.mean(target_state_mean, dim=0, keepdim=True)
    print( f'time {time.time()-code_start_time:.3f}, selected one subject, so now', target_state_mean.size() )
    target_state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
    target_state_product_mean = torch.load(target_state_product_mean_file)
    # On load, the dimensions of target_state_product_mean should be scan x subject x node x node or scan x subject x node-pair.
    print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', target_state_product_mean.size() )
    if len( target_state_product_mean.size() ) < 4:
        target_state_product_mean = isingmodellight.triu_to_square_pairs( triu_pairs=target_state_product_mean, diag_fill=0 )
    target_state_product_mean = target_state_product_mean[:,subject_index,:,:]
    if combine_scans:
        target_state_product_mean = torch.mean(target_state_product_mean, dim=0, keepdim=True)
    print( f'time {time.time()-code_start_time:.3f}, converted to square format and selected subject so now', target_state_product_mean.size() )

    print('initializing Ising model...')
    num_targets, num_nodes = target_state_mean.size()
    beta = isingmodellight.get_linspace_beta(models_per_subject=models_per_subject, num_subjects=num_targets, dtype=float_type, device=device)
    s = isingmodellight.get_neg_state(models_per_subject=models_per_subject, num_subjects=num_targets, num_nodes=num_nodes, dtype=float_type, device=device)
    h = target_state_mean.unsqueeze(dim=0).repeat( (models_per_subject,1,1) )
    init_J = target_state_product_mean.clone()
    if center_cov:
        init_J -= target_state_mean.unsqueeze(dim=-1) * target_state_mean.unsqueeze(dim=-2)
    if use_inverse_cov:
        init_J = torch.linalg.inv(init_J)
    # 0 out the diagonal.
    # init_J -= torch.diag_embed( torch.diagonal(init_J, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
    # J = init_J.unsqueeze(dim=0).repeat( (models_per_subject, 1, 1, 1) )
    J = isingmodellight.get_J_from_means(models_per_subject=models_per_subject, mean_state_product=init_J)
    model = IsingModelLight(beta=beta, J=J, h=h, s=s)
    print( f'time {time.time()-code_start_time:.3f}, initialized model with h of size', model.h.size(), ' J of size', model.J.size(), 'and state of size', model.s.size() )

    if center_cov and use_inverse_cov:
        init_str = 'inv_centered'
    elif use_inverse_cov:
        init_str = 'inv_uncentered'
    elif center_cov:
        init_str = 'centered'
    else:
        init_str = 'uncentered'
    model_file_fragment = f'{data_file_name_part}_{output_file_name_part}_subject_{subject_index}_init_{init_str}_beta_num_{models_per_subject}_min_{min_beta:.3g}_max_{max_beta:.3g}_steps_{sim_length}'
    print('simulating...')
    sim_state_mean, sim_state_product_mean, flip_rate = model.simulate_and_record_means_and_flip_rate_faster(num_steps=sim_length)
    sim_state_mean_file = os.path.join(output_directory, f'sim_state_mean_{model_file_fragment}.pt')
    torch.save(obj=sim_state_mean, f=sim_state_mean_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_state_mean_file}')
    sim_state_product_mean_file = os.path.join(output_directory, f'sim_state_product_mean_{model_file_fragment}.pt')
    torch.save(obj=sim_state_product_mean, f=sim_state_product_mean_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_state_product_mean_file}')
    flip_rate_file = os.path.join(output_directory, f'flip_rate_{model_file_fragment}.pt')
    torch.save(obj=flip_rate, f=flip_rate_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file}')
    target_cov = isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
    sim_cov = isingmodellight.get_cov(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
    cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_cov, mat2=target_cov)
    cov_rmse_file = os.path.join(output_directory, f'cov_rmse_{model_file_fragment}.pt')
    torch.save(obj=cov_rmse, f=cov_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {cov_rmse_file}')
    cov_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_cov, mat2=target_cov)
    cov_corr_file = os.path.join(output_directory, f'cov_corr_{model_file_fragment}.pt')
    torch.save(obj=cov_corr, f=cov_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {cov_corr_file}')
    target_fc = isingmodellight.get_fc_binary(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
    sim_fc = isingmodellight.get_fc_binary(state_mean=sim_state_mean, state_product_mean=sim_state_product_mean)
    fc_rmse = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc)
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{model_file_fragment}.pt')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file}')
    fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc)
    fc_corr_file = os.path.join(output_directory, f'fc_corr_{model_file_fragment}.pt')
    torch.save(obj=fc_corr, f=fc_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_corr_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')