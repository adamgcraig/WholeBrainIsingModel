import os
import torch
import time
import argparse
import isingmodelshort
from isingmodelshort import IsingModel

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Fit a batch of Ising models (h: replica x subject x ROI, J: replica x subject x ROI x ROI) to a batch of target means (subject x ROI) and uncentered covariances (subject x ROI x ROI) via Boltzmann learning.")
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\aal90_short_pytorch', help="directory where we can find the target mean state and mean state product files")
    parser.add_argument("-b", "--model_directory", type=str, default='E:\\aal90_short_models', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--mean_state_file_name", type=str, default='data_mean_all_mean_std_1.pt', help="name of the target mean state file")
    parser.add_argument("-d", "--mean_state_product_file_name", type=str, default='data_mean_product_all_mean_std_1.pt', help="name of the target mean state product file")
    parser.add_argument("-e", "--model_file_name_prefix", type=str, default='ising_model_all_mean_std_1', help="prefix with which to start the names of the saved Ising model files")
    parser.add_argument("-f", "--models_per_subject", type=int, default=101, help="number of models to fit to each data target")
    parser.add_argument("-g", "--beta_sim_length", type=int, default=1200, help="number of simulation steps between updates in the beta optimization loop")
    parser.add_argument("-i", "--param_sim_length", type=int, default=1200, help="number of simulation steps between updates in the main parameter-fitting loop")
    parser.add_argument("-j", "--max_num_beta_updates", type=int, default=1000000, help="maximum number of updates within which to find the optimal inverse temperature beta (We stop if we find it to within machine precision.)")
    parser.add_argument("-k", "--updates_per_save", type=int, default=1000, help="number of updates to parameters h and J to perform between snapshots (These almost never perfectly converge, so we never stop early.)")
    parser.add_argument("-l", "--num_saves", type=int, default=1000, help="number of times we save a model snapshot to a file")
    parser.add_argument("-m", "--learning_rate", type=float, default=0.01, help="amount by which to scale down updates to the model parameters during parameter fitting")
    parser.add_argument("-n", "--min_beta", type=float, default=10e-10, help="low end of initial beta search interval")
    parser.add_argument("-o", "--max_beta", type=float, default=1.0, help="high end of initial beta search interval")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'model_directory={data_directory}')
    model_directory = args.model_directory
    print(f'output_directory={model_directory}')
    mean_state_file_name = args.mean_state_file_name
    print(f'mean_state_file_name={mean_state_file_name}')
    mean_state_product_file_name = args.mean_state_product_file_name
    print(f'mean_state_product_file_name={mean_state_product_file_name}')
    model_file_name_prefix = args.model_file_name_prefix
    print(f'model_file_name_prefix={model_file_name_prefix}')
    models_per_subject = args.models_per_subject
    print(f'models_per_subject={models_per_subject}')
    beta_sim_length = args.beta_sim_length
    print(f'beta_sim_length={beta_sim_length}')
    param_sim_length = args.param_sim_length
    print(f'param_sim_length={param_sim_length}')
    max_num_beta_updates = args.max_num_beta_updates
    print(f'max_num_beta_updates={max_num_beta_updates}')
    updates_per_save = args.updates_per_save
    print(f'updates_per_save={updates_per_save}')
    num_saves = args.num_saves
    print(f'num_saves={num_saves}')
    learning_rate = args.learning_rate
    print(f'learning_rate={learning_rate}')
    min_beta = args.min_beta
    print(f'min_beta={min_beta}')
    max_beta = args.max_beta
    print(f'max_beta={max_beta}')
    
    print('loading data time series state and state product means')
    target_state_mean_file = os.path.join(data_directory, mean_state_file_name)
    target_mean_state = torch.load(target_state_mean_file, weights_only=False)
    # On load, the dimensions of target_mean_state should be subject x node.
    print( f'time {time.time()-code_start_time:.3f}, loaded target_mean_state with size', target_mean_state.size() )
    target_mean_state_product_file = os.path.join(data_directory, mean_state_product_file_name)
    target_mean_state_product = torch.load(target_mean_state_product_file, weights_only=False)
    # On load, the dimensions of target_mean_state_product should be subject x node x node.
    print( f'time {time.time()-code_start_time:.3f}, loaded target_mean_state_product with size', target_mean_state_product.size() )

    print('initializing Ising model...')
    num_targets, num_nodes = target_mean_state.size()
    beta = isingmodelshort.get_linspace_beta(models_per_subject=models_per_subject, num_subjects=num_targets, dtype=float_type, device=device)
    J = isingmodelshort.get_J_from_means(models_per_subject=models_per_subject, mean_state_product=target_mean_state_product)
    h = isingmodelshort.get_h_from_means(models_per_subject=models_per_subject, mean_state=target_mean_state)
    s = isingmodelshort.get_neg_state_like(input=h)
    model = IsingModel(beta=beta, J=J, h=h, s=s)
    print( f'time {time.time()-code_start_time:.3f}, initialized model with beta of size ', model.beta.size(), ' h of size', model.h.size(), ' J of size', model.J.size(), 'and state of size', model.s.size() )

    target_cov = isingmodelshort.get_cov(state_mean=target_mean_state, state_product_mean=target_mean_state_product)
    
    num_beta_updates = 0
    total_param_updates = 0
    print('optimizing beta...')
    num_beta_updates = model.optimize_beta(target_cov=target_cov, num_updates=max_num_beta_updates, num_steps=beta_sim_length, min_beta=min_beta, max_beta=max_beta, verbose=True)
    print( f'time {time.time()-code_start_time:.3f}, done optimizing beta after {num_beta_updates} iterations' )
    model_file_name_prefix_with_beta = f'{model_file_name_prefix}_beta_updates_{num_beta_updates}'
    model_file = os.path.join(model_directory, f'{model_file_name_prefix_with_beta}.pt')
    torch.save(obj=model, f=model_file)
    for _ in range(num_saves):
        print('fitting parameters h and J...')
        model.fit_by_simulation(target_mean_state=target_mean_state, target_mean_state_product=target_mean_state_product, num_updates=updates_per_save, steps_per_update=param_sim_length, learning_rate=learning_rate, verbose=True)
        total_param_updates += updates_per_save
        model_file = os.path.join(model_directory, f'{model_file_name_prefix_with_beta}_param_updates_{total_param_updates}.pt')
        torch.save(obj=model, f=model_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')