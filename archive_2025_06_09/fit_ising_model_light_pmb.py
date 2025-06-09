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
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_as_is', help="part of the output file name between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--output_file_name_part", type=str, default='short', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
    parser.add_argument("-e", "--models_per_subject", type=int, default=3, help="number of separate models of each subject")
    parser.add_argument("-s", "--beta_sim_length", type=int, default=1200, help="number of simulation steps between updates in the beta optimization loop")
    parser.add_argument("-f", "--param_sim_length", type=int, default=1200, help="number of simulation steps between updates in the main parameter-fitting loop")
    parser.add_argument("-g", "--num_updates_beta", type=int, default=1000000, help="maximum number of updates within which to find the optimal inverse temperature beta (We stop if we find it to within machine precision.)")
    parser.add_argument("-i", "--updates_per_save", type=int, default=1000, help="number of fitting updates of individual parameters between re-optimizations of beta (In practice, these never perfectly converge, so we do not set any stopping criterion.)")
    parser.add_argument("-j", "--saves_per_beta_opt", type=int, default=1000, help="number of times we save a model to a file")
    parser.add_argument("-k", "--num_beta_opts", type=int, default=1, help="number of times we re-optimize beta between saves of the")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.01, help="amount by which to multiply updates to the model parameters during the Euler step")
    parser.add_argument("-m", "--center_cov", action='store_true', default=False, help="Set this flag in order to initialize J to the centered covariance instead of the uncentered covariance (state mean product).")
    parser.add_argument("-n", "--use_inverse_cov", action='store_true', default=False, help="Set this flag in order to initialize J to the inverse covariance instead of the covariance.")
    parser.add_argument("-o", "--combine_scans", action='store_true', default=False, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-p", "--min_beta", type=float, default=10e-10, help="low end of initial beta search interval")
    parser.add_argument("-q", "--max_beta", type=float, default=1.0, help="high end of initial beta search interval")
    parser.add_argument("-r", "--target_flip_rate", type=float, default=None, help="target flip rate to use when optimizing beta")
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
    models_per_subject = args.models_per_subject
    print(f'models_per_subject={models_per_subject}')
    beta_sim_length = args.beta_sim_length
    print(f'beta_sim_length={beta_sim_length}')
    param_sim_length = args.param_sim_length
    print(f'param_sim_length={param_sim_length}')
    num_updates_beta = args.num_updates_beta
    print(f'num_updates_scaling={num_updates_beta}')
    updates_per_save = args.updates_per_save
    print(f'updates_per_save={updates_per_save}')
    saves_per_beta_opt = args.saves_per_beta_opt
    print(f'saves_per_beta_opt={saves_per_beta_opt}')
    num_beta_opts = args.num_beta_opts
    print(f'num_beta_opts={num_beta_opts}')
    learning_rate = args.learning_rate
    print(f'learning_rate={learning_rate}')
    center_cov = args.center_cov
    print(f'center_cov={center_cov}')
    use_inverse_cov = args.use_inverse_cov
    print(f'use_inverse_cov={use_inverse_cov}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')
    min_beta = args.min_beta
    print(f'min_beta={min_beta}')
    max_beta = args.max_beta
    print(f'max_beta={max_beta}')
    target_flip_rate = args.target_flip_rate
    print(f'target_flip_rate={target_flip_rate}')
    
    print('loading data time series state and state product means')
    target_state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
    target_state_mean = torch.load(target_state_mean_file)
    # On load, the dimensions of target_state_mean should be scan x subject x node or scan x subject x node-pair.
    print( f'time {time.time()-code_start_time:.3f}, loaded target_state_mean with size', target_state_mean.size() )
    if combine_scans:
        target_state_mean = torch.mean(target_state_mean, dim=0, keepdim=False)
    else:
        target_state_mean = torch.flatten(target_state_mean, start_dim=0, end_dim=1)
    print( f'time {time.time()-code_start_time:.3f}, flattened scan and subject dimensions', target_state_mean.size() )
    target_state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
    target_state_product_mean = torch.load(target_state_product_mean_file)
    # On load, the dimensions of target_state_product_mean should be scan x subject x node x node or scan x subject x node-pair.
    print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', target_state_product_mean.size() )
    if len( target_state_product_mean.size() ) < 4:
        target_state_product_mean = isingmodellight.triu_to_square_pairs( triu_pairs=target_state_product_mean, diag_fill=0 )
    if combine_scans:
        target_state_product_mean = torch.mean(target_state_product_mean, dim=0, keepdim=False)
    else:
        target_state_product_mean = torch.flatten(target_state_product_mean, start_dim=0, end_dim=1)
    print( f'time {time.time()-code_start_time:.3f}, converted to square format and flattened scan and subject dimensions', target_state_product_mean.size() )

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
    init_J -= torch.diag_embed( torch.diagonal(init_J, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
    J = init_J.unsqueeze(dim=0).repeat( (models_per_subject, 1, 1, 1) )
    model = IsingModelLight(beta=beta, J=J, h=h, s=s)
    print( f'time {time.time()-code_start_time:.3f}, initialized model with h of size', model.h.size(), ' J of size', model.J.size(), 'and state of size', model.s.size() )

    target_cov = isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
    
    if center_cov and use_inverse_cov:
        init_str = 'inv_centered'
    elif use_inverse_cov:
        init_str = 'inv_uncentered'
    elif center_cov:
        init_str = 'centered'
    else:
        init_str = 'uncentered'
    base_model_file_fragment = f'{data_file_name_part}_{output_file_name_part}_init_{init_str}_reps_{models_per_subject}_beta_min_{min_beta:.3g}_max_{max_beta:.3g}_steps_{beta_sim_length}_lr_{learning_rate}_steps_{param_sim_length}_pupd_per_bopt_{updates_per_save}'

    num_updates_beta_total = 0
    num_updates_params_total = 0
    for beta_opt_index in range(num_beta_opts):
        print('optimizing beta...')
        num_beta_updates_completed = model.optimize_beta_pmb(target_cov=target_cov, num_updates=num_updates_beta, num_steps=beta_sim_length, verbose=True, min_beta=min_beta, max_beta=max_beta)
        num_updates_beta_total += num_beta_updates_completed
        print( f'time {time.time()-code_start_time:.3f}, done optimizing beta after {num_beta_updates_completed} iterations' )
        for save_index in range(saves_per_beta_opt):
            print('fitting parameters h and J...')
            model.fit_by_simulation_pmb(target_state_mean=target_state_mean, target_state_product_mean=target_state_product_mean, num_updates=updates_per_save, steps_per_update=param_sim_length, learning_rate=learning_rate, verbose=True)
            num_updates_params_total += updates_per_save
            model_file_fragment = f'{base_model_file_fragment}_num_opt_{beta_opt_index+1}_bopt_steps_{num_updates_beta_total}_popt_steps_{num_updates_params_total}'
            model_file = os.path.join(output_directory, f'ising_model_light_{model_file_fragment}.pt')
            torch.save(obj=model, f=model_file)
            print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')