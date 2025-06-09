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
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_mean_std_0', help="part of the output file name between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--output_file_name_part", type=str, default='short', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
    parser.add_argument("-s", "--subject_index", type=int, default=0, help="individual subject to select")
    parser.add_argument("-f", "--sim_length", type=int, default=1200, help="number of simulation steps between updates")
    parser.add_argument("-m", "--center_cov", action='store_true', default=False, help="Set this flag in order to initialize J to the centered covariance instead of the uncentered covariance (state mean product).")
    parser.add_argument("-n", "--use_inverse_cov", action='store_true', default=False, help="Set this flag in order to initialize J to the inverse covariance instead of the covariance.")
    parser.add_argument("-o", "--combine_scans", action='store_true', default=True, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
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
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
    center_cov = args.center_cov
    print(f'center_cov={center_cov}')
    use_inverse_cov = args.use_inverse_cov
    print(f'use_inverse_cov={use_inverse_cov}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')

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
    selected_beta = [0.0, 0.01, 0.027, 0.029102912172675133, 0.035, 0.06]
    models_per_subject = len(selected_beta)
    beta = torch.tensor(data=selected_beta, dtype=float_type, device=device).unsqueeze(dim=-1).repeat( repeats=(1,num_targets) )
    s = isingmodellight.get_neg_state(models_per_subject=models_per_subject, num_subjects=num_targets, num_nodes=num_nodes, dtype=float_type, device=device)
    h = target_state_mean.unsqueeze(dim=0).repeat( (models_per_subject,1,1) )
    init_J = target_state_product_mean.clone()
    if center_cov:
        init_J -= target_state_mean.unsqueeze(dim=-1) * target_state_mean.unsqueeze(dim=-2)
    if use_inverse_cov:
        init_J = torch.linalg.inv(init_J)
    # 0 out the diagonal.
    init_J -= torch.diag_embed( torch.diagonal(init_J, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
    J = init_J.unsqueeze(dim=0).repeat( (models_per_subject, 1, 1, 1) )
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
    model_file_fragment = f'{data_file_name_part}_{output_file_name_part}_subject_{subject_index}_init_{init_str}_beta_num_{models_per_subject}_selected_beta_steps_{sim_length}'
    print('simulating for time series and covariance...')
    time_series, sim_cov = model.simulate_and_record_time_series_and_cov(num_steps=sim_length)
    time_series_file = os.path.join(output_directory, f'time_series_{model_file_fragment}.pt')
    torch.save(obj=time_series, f=time_series_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {time_series_file}')
    sim_cov_file = os.path.join(output_directory, f'sim_cov_{model_file_fragment}.pt')
    torch.save(obj=sim_cov, f=sim_cov_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_cov_file}')
    target_cov = isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
    target_cov_file = os.path.join(output_directory, f'target_cov_{model_file_fragment}.pt')
    torch.save(obj=target_cov, f=target_cov_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {target_cov_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')