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
    parser.add_argument("-d", "--model_file_fragment", type=str, default='pseudolikelihood_mean_std_1_models_5_rand_min_-100_max_100_lr_0.001_steps_1000000_num_beta_837_min_1e-10_max_1_updates_1000_sim_120000', help="name of the model file to load")
    parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
    parser.add_argument("-p", "--min_beta", type=float, default=10e-10, help="low end of initial beta search interval")
    parser.add_argument("-q", "--max_beta", type=float, default=1.0, help="high end of initial beta search interval")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    model_file_fragment = args.model_file_fragment
    print(f'model_file_fragment={model_file_fragment}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
    min_beta = args.min_beta
    print(f'min_beta={min_beta}')
    max_beta = args.max_beta
    print(f'min_beta={max_beta}')
    epsilon = 0.0

    model_file = os.path.join(output_directory, f'ising_model_{model_file_fragment}.pt')
    model = torch.load(f=model_file)
    print(f'time {time.time()-code_start_time:.3f}, loaded {model_file}')

    model.beta = isingmodellight.get_linspace_beta( models_per_subject=model.beta.size(dim=0), num_subjects=model.beta.size(dim=1), dtype=model.beta.dtype, device=model.beta.device, min_beta=min_beta, max_beta=max_beta )
    print(f'time {time.time()-code_start_time:.3f}, starting simulation...')
    sim_state_mean, sim_state_product_mean, flip_rate = model.simulate_and_record_means_and_flip_rate_pmb(num_steps=sim_length)
    print(f'time {time.time()-code_start_time:.3f}, simulation complete')

    output_file_fragment = f'{model_file_fragment}_test_beta_min_{min_beta:.3g}_max_{max_beta:.3g}_sim_steps_{sim_length}'
    sim_state_mean_file = os.path.join(output_directory, f'sim_state_mean_{output_file_fragment}.pt')
    torch.save(obj=sim_state_mean, f=sim_state_mean_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_state_mean_file}')
    sim_state_product_mean_file = os.path.join(output_directory, f'sim_state_product_mean_{output_file_fragment}.pt')
    torch.save(obj=sim_state_product_mean, f=sim_state_product_mean_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_state_product_mean_file}')
    flip_rate_file = os.path.join(output_directory, f'flip_rate_{output_file_fragment}.pt')
    torch.save(obj=flip_rate, f=flip_rate_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')