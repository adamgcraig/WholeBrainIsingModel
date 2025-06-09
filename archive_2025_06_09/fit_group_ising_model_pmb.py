import os
import torch
import time
import argparse
import hcpdatautilsnopandas as hcp
import isingmodellight
from isingmodellight import IsingModelLight

def get_binarized_info(data_ts:torch.Tensor, threshold_z:float):
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    binarized_ts = 2*( data_ts > (data_ts_mean + threshold_z*data_ts_std) ).float() - 1
    # We want to average over all scans and all subjects but then the model expects a singleton subject batch dimension.
    return binarized_ts.mean(dim=-1).mean( dim=(0,1) ).unsqueeze(dim=0), torch.matmul( binarized_ts, binarized_ts.transpose(dim0=-2, dim1=-1) ).mean( dim=(0,1) ).unsqueeze(dim=0)/binarized_ts.size(dim=-1)

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Test the effect of binarization threshold on our ability to fit a group Ising model.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\HCP_data', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-c", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-e", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    parser.add_argument("-f", "--threshold_z", type=float, default=1.0, help="binarization threshold in std. dev.s above the mean")
    parser.add_argument("-j", "--models_per_subject", type=int, default=5, help="number of instances of the group Ising model to train for each subject")
    parser.add_argument("-d", "--beta_sim_length", type=int, default=12000, help="number of simulation steps between beta updates")
    parser.add_argument("-k", "--param_sim_length", type=int, default=1200, help="number of simulation steps between parameter updates")
    parser.add_argument("-l", "--num_updates_beta", type=int, default=1000000, help="maximum number of updates within which to find the optimal inverse temperature beta (We stop if we find it to within machine precision.)")
    parser.add_argument("-m", "--updates_per_save", type=int, default=1000, help="number of fitting updates of individual parameters between re-optimizations of beta (In practice, these never perfectly converge, so we do not set any stopping criterion.)")
    parser.add_argument("-n", "--num_saves", type=int, default=1000, help="number of times we save a model after doing updates_per_save parameter updates")
    parser.add_argument("-o", "--learning_rate", type=float, default=0.01, help="amount by which to multiply updates to the model parameters during the Euler step")
    parser.add_argument("-p", "--min_beta", type=float, default=1e-10, help="low end of initial beta search interval")
    parser.add_argument("-q", "--max_beta", type=float, default=1.0, help="high end of initial beta search interval")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')
    threshold_z = args.threshold_z
    print(f'threshold_z={threshold_z:.3g}')
    models_per_subject = args.models_per_subject
    print(f'models_per_subject={models_per_subject}')
    beta_sim_length = args.beta_sim_length
    print(f'beta_sim_length={beta_sim_length}')
    param_sim_length = args.param_sim_length
    print(f'param_sim_length={param_sim_length}')
    num_updates_beta = args.num_updates_beta
    print(f'num_updates_beta={num_updates_beta}')
    updates_per_save = args.updates_per_save
    print(f'updates_per_save={updates_per_save}')
    num_saves = args.num_saves
    print(f'num_saves={num_saves}')
    learning_rate = args.learning_rate
    print(f'learning_rate={learning_rate}')
    min_beta = args.min_beta
    print(f'min_beta={min_beta}')
    max_beta = args.max_beta
    print(f'min_beta={max_beta}')

    data_ts_file = os.path.join(output_directory, f'data_ts_all_as_is.pt')
    data_ts = torch.load(data_ts_file)
    num_ts, num_subjects, num_nodes, num_steps = data_ts.size()
    print(f'time {time.time()-code_start_time:.3f}, computing binarized mean states and state products...')
    # Use only the training subjects to get the group means.
    target_state_mean, target_state_product_mean = get_binarized_info(data_ts=data_ts[:,training_subject_start:training_subject_end,:,:], threshold_z=threshold_z)
    
    print('initializing Ising model...')
    beta = isingmodellight.get_linspace_beta(models_per_subject=models_per_subject, num_subjects=num_subjects, dtype=float_type, device=device)
    s = isingmodellight.get_neg_state(models_per_subject=models_per_subject, num_subjects=num_subjects, num_nodes=num_nodes, dtype=float_type, device=device)
    h = target_state_mean.unsqueeze(dim=0).repeat( (models_per_subject, num_subjects, 1) )
    # 0 out the diagonal.
    init_J = target_state_product_mean - torch.diag_embed( torch.diagonal(target_state_product_mean, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
    J = init_J.unsqueeze(dim=0).repeat( (models_per_subject, num_subjects, 1, 1) )
    model = IsingModelLight(beta=beta, J=J, h=h, s=s)
    print( f'time {time.time()-code_start_time:.3f}, initialized model with h of size', model.h.size(), ' J of size', model.J.size(), 'and state of size', model.s.size() )
    
    print('optimizing beta...')
    num_beta_updates_completed = model.optimize_beta_pmb( target_cov=isingmodellight.get_cov(state_mean=target_state_mean, state_product_mean=target_state_product_mean), num_updates=num_updates_beta, num_steps=beta_sim_length, verbose=True, min_beta=min_beta, max_beta=max_beta )
    print( f'time {time.time()-code_start_time:.3f}, done optimizing beta after {num_beta_updates_completed} iterations' )
    model_file_fragment = f'ising_model_light_group_threshold_{threshold_z:.3g}_betas_{models_per_subject}_min_{min_beta:.3g}_max_{max_beta:.3g}_beta_steps_{beta_sim_length}_param_steps_{param_sim_length}_lr_{learning_rate:.3g}_beta_updates_{num_beta_updates_completed}'
    model_file = os.path.join(output_directory, f'{model_file_fragment}.pt')
    torch.save(obj=model, f=model_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
    print('fitting parameters h and J...')
    num_param_updates_total = 0
    for save_index in range(num_saves):
        model.fit_by_simulation_pmb(target_state_mean=target_state_mean, target_state_product_mean=target_state_product_mean, num_updates=updates_per_save, steps_per_update=param_sim_length, learning_rate=learning_rate, verbose=True)
        num_param_updates_total += updates_per_save
        model_file = os.path.join(output_directory, f'{model_file_fragment}_param_updates_{num_param_updates_total}.pt')
        torch.save(obj=model, f=model_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')
