import os
import torch
import time
import argparse
import isingmodel
from isingmodel import IsingModel

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-f", "--data_subset", type=str, default='validation', help="the subset of subjects over which to search for unique states")
parser.add_argument("-m", "--model_type", type=str, default='individual', help="either 'group' for a group model or 'individual' for an individual model")
parser.add_argument("-b", "--num_folds", type=int, default=1, help="number of unconnected parts into which to partition the brain")
parser.add_argument("-l", "--sim_length", type=int, default=12000, help="number of simulation steps between Euler updates")
parser.add_argument("-s", "--sims_per_save", type=int, default=100, help="determines how frequently to save snapshots of the model")
parser.add_argument("-d", "--num_updates_beta", type=int, default=100, help="number of updates used when optimizing beta")
parser.add_argument("-u", "--num_updates_boltzmann", type=int, default=1000000, help="number of Euler/Boltzmann updates to do")
parser.add_argument("-p", "--num_parallel", type=int, default=10, help="number of parallel simulations to run")
parser.add_argument("-r", "--learning_rate", type=float, default=0.1, help="amount by which to multiply updates to the model parameters during Boltzmann learning")
parser.add_argument("-z", "--resume_from_sim", type=int, default=-1, help="If None or < 0, start with a new model. If 0, load beta-optimized model. If 1, load model saved after first Boltzmann learning step, etc.")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
num_folds = args.num_folds
print(f'num_folds={num_folds}')
model_type = args.model_type
print(f'model_type={model_type}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
sims_per_save = args.sims_per_save
print(f'sims_per_save={sims_per_save}')
num_updates_beta = args.num_updates_beta
print(f'num_updates_beta={num_updates_beta}')
num_updates_boltzmann = args.num_updates_boltzmann
print(f'num_updates={num_updates_boltzmann}')
num_parallel = args.num_parallel
print(f'num_parallel={num_parallel}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')
resume_from_sim = args.resume_from_sim
print(f'resume_from_sim={resume_from_sim}')
if type(resume_from_sim) == type(None):
    resume_from_sim = -1

def load_data_means_and_fc(output_directory:str, model_type:str, data_subset:str, num_folds:int, num_parallel:int):
    file_suffix = f'{model_type}_{data_subset}.pt'
    state_mean_file = os.path.join(output_directory, f'mean_state_{file_suffix}')
    mean_state = torch.load(state_mean_file)
    batch_size, num_nodes = mean_state.size()
    mean_state = isingmodel.fold_nodes(node_params=mean_state, num_folds=num_folds)
    folded_batch_size, folded_num_nodes = mean_state.size()
    print(f'loaded {state_mean_file} and folded from {batch_size} x {num_nodes} to {folded_batch_size} x {folded_num_nodes}, time {time.time() - code_start_time:.3f}')
    product_mean_file = os.path.join(output_directory, f'mean_state_product_{file_suffix}')
    mean_state_product = torch.load(product_mean_file)
    batch_size, num_pairs = mean_state_product.size()
    mean_state_product = isingmodel.fold_node_pairs(pair_params=mean_state_product, num_nodes=num_nodes, num_folds=num_folds)
    folded_batch_size, folded_num_pairs = mean_state_product.size()
    print(f'loaded {product_mean_file} and folded from {batch_size} x {num_pairs} to {folded_batch_size} x {folded_num_pairs}, time {time.time() - code_start_time:.3f}')
    # Load the FC of the data.
    fc_file = os.path.join(output_directory, f'fc_{file_suffix}')
    data_fc = torch.load(fc_file)
    batch_size, num_pairs = data_fc.size()
    data_fc = isingmodel.fold_node_pairs(pair_params=data_fc, num_nodes=num_nodes, num_folds=num_folds)
    folded_batch_size, folded_num_pairs = data_fc.size()
    print(f'loaded {fc_file} and folded from {batch_size} x {num_pairs} to {folded_batch_size} x {folded_num_pairs}, time {time.time() - code_start_time:.3f}')
    # Now replicate once for each beta we want to try at a time.
    mean_state = mean_state.unsqueeze(dim=0).repeat( (num_parallel,1,1) ).flatten(start_dim=0, end_dim=1)
    mean_state_product = mean_state_product.unsqueeze(dim=0).repeat( (num_parallel,1,1) ).flatten(start_dim=0, end_dim=1)
    data_fc = data_fc.unsqueeze(dim=0).repeat( (num_parallel,1,1) ).flatten(start_dim=0, end_dim=1)
    return mean_state, mean_state_product, data_fc

def init_and_optimize_beta(output_directory:str, file_suffix:str, mean_state:torch.Tensor, mean_state_product:torch.Tensor, data_fc:torch.Tensor, folded_num_nodes:int, folded_batch_size:int, num_parallel:int, num_updates_beta:int, sim_length:int):
    # Initialize the model.
    beta = torch.linspace(start=0.0, end=1.0, steps=num_parallel, dtype=float_type, device=device).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat( (1,folded_batch_size,1) )
    total_num_models = num_parallel * folded_batch_size
    model = IsingModel( batch_size=total_num_models, num_nodes=folded_num_nodes, beta=beta.flatten(start_dim=0,end_dim=1), dtype=float_type, device=device )
    model.set_params(h=mean_state, J_triu=mean_state_product)
    model.set_target_means(target_node_means=mean_state, target_node_product_means=mean_state_product)
    # Optimize beta.
    for sim in range(num_updates_beta):
        sim_fc = model.simulate_and_record_fc(num_steps=sim_length)
        fc_rmse = isingmodel.get_pairwise_rmse(mat1=data_fc, mat2=sim_fc).unflatten( dim=0, sizes=(num_parallel, folded_batch_size) )
        fc_correlation = isingmodel.get_pairwise_correlation(mat1=sim_fc, mat2=data_fc)
        best_beta_index = fc_rmse.argmin(dim=0, keepdim=False)
        for model_index in range(folded_batch_size):
            best_beta_index_for_model = best_beta_index[model_index]
            betas_for_model = beta[:,model_index,0]
            best_beta_for_model = betas_for_model[best_beta_index_for_model]
            if best_beta_index_for_model == 0:
                beta_start = 0.0
                if best_beta_for_model == 0.0:
                    beta_end = betas_for_model[1]
                else:
                    beta_end = best_beta_for_model
            elif best_beta_index_for_model == num_parallel-1:
                beta_start = best_beta_for_model
                beta_end = 2.0 * best_beta_for_model
            else:
                beta_start = betas_for_model[best_beta_index_for_model-1]
                beta_end = betas_for_model[best_beta_index_for_model+1]
            beta[:,model_index,0] = torch.linspace(start=beta_start, end=beta_end, steps=num_parallel, dtype=float_type, device=device)
        model.beta = beta.flatten(start_dim=0, end_dim=1)
        print(f'beta optimization step {sim+1}\tbeta min {beta.min():.3g},\tmean {beta.mean():.3g},\tmax {beta.max():.3g},\tRMSE min {fc_rmse.min():.3g},\tmean {fc_rmse.mean():.3g},\tmax {fc_rmse.max():.3g},\tcorrelation min {fc_correlation.min():.3g},\tmean {fc_correlation.mean():.3g},\tmax {fc_correlation.max():.3g},\ttime {time.time() - code_start_time:.3f}')
    model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    torch.save(obj=model, f=model_file)
    print(f'saved {model_file}')
    sim_fc_file = os.path.join(output_directory, f'sim_fc_{file_suffix}')
    torch.save(obj=sim_fc, f=sim_fc_file)
    print(f'saved {sim_fc_file}')
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{file_suffix}')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'saved {fc_rmse_file}')
    return model

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    # Load the mean values and FC of the data.
    mean_state, mean_state_product, data_fc = load_data_means_and_fc(output_directory=output_directory, model_type=model_type, data_subset=data_subset, num_folds=num_folds, num_parallel=num_parallel)
    total_num_models, folded_num_nodes = mean_state.size()
    folded_batch_size = total_num_models//num_parallel
    
    beta_opt_file_suffix = f'{model_type}_{data_subset}_fold_{num_folds}_parallel_{num_parallel}_steps_{sim_length}_beta_sims_{num_updates_beta}.pt'
    if resume_from_sim < 0:
        model = init_and_optimize_beta(output_directory=output_directory, file_suffix=beta_opt_file_suffix, mean_state=mean_state, mean_state_product=mean_state_product, data_fc=data_fc, folded_num_nodes=folded_num_nodes, folded_batch_size=folded_batch_size, num_parallel=num_parallel, num_updates_beta=num_updates_beta, sim_length=sim_length)
        start_sim = 0
    elif resume_from_sim == 0:
        model_file = os.path.join(output_directory, f'ising_model_{beta_opt_file_suffix}')
        model = torch.load(f=model_file)
        start_sim = 0
    else:
        model_file = os.path.join(output_directory, f'ising_model_{model_type}_{data_subset}_fold_{num_folds}_parallel_{num_parallel}_steps_{sim_length}_beta_sims_{num_updates_beta}_boltzmann_sims_{resume_from_sim}_lr_{learning_rate:.3g}.pt')
        model = torch.load(f=model_file)
        start_sim = resume_from_sim

    # Second, do Boltzmann learning.
    for sim in range(start_sim, num_updates_boltzmann):
        model.fit_to_stored_means_log_update(num_updates=1, steps_per_update=sim_length, learning_rate=learning_rate)
        if ( (sim+1) % sims_per_save ) == 0:
            sim_fc = model.simulate_and_record_fc(num_steps=sim_length)
            fc_rmse = isingmodel.get_pairwise_rmse(mat1=sim_fc, mat2=data_fc)
            fc_correlation = isingmodel.get_pairwise_correlation(mat1=sim_fc, mat2=data_fc)
            print(f'Boltzmann learning step {sim+1},\tRMSE min {fc_rmse.min():.3g},\tmean {fc_rmse.mean():.3g},\tmax {fc_rmse.max():.3g},\tcorrelation min {fc_correlation.min():.3g},\tmean {fc_correlation.mean():.3g},\tmax {fc_correlation.max():.3g},\ttime {time.time() - code_start_time:.3f}')
            file_suffix = f'{model_type}_{data_subset}_fold_{num_folds}_parallel_{num_parallel}_steps_{sim_length}_beta_sims_{num_updates_beta}_boltzmann_sims_{sim+1}_lr_{learning_rate:.3g}.pt'
            model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
            torch.save(obj=model, f=model_file)
            print(f'saved {model_file}')
            sim_fc_file = os.path.join(output_directory, f'sim_fc_{file_suffix}')
            torch.save(obj=sim_fc, f=sim_fc_file)
            print(f'saved {sim_fc_file}')
            fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{file_suffix}')
            torch.save(obj=fc_rmse, f=fc_rmse_file)
            print(f'saved {fc_rmse_file}')
    sim_fc = model.simulate_and_record_fc(num_steps=sim_length)
    fc_rmse = isingmodel.get_pairwise_rmse(mat1=sim_fc, mat2=data_fc)
    fc_correlation = isingmodel.get_pairwise_correlation(mat1=sim_fc, mat2=data_fc)
    print(f'Boltzmann learning step {sim+1},\tRMSE min {fc_rmse.min():.3g},\tmean {fc_rmse.mean():.3g},\tmax {fc_rmse.max():.3g},\tcorrelation min {fc_correlation.min():.3g},\tmean {fc_correlation.mean():.3g},\tmax {fc_correlation.max():.3g},\ttime {time.time() - code_start_time:.3f}')
    file_suffix = f'{model_type}_{data_subset}_fold_{num_folds}_parallel_{num_parallel}_steps_{sim_length}_beta_sims_{num_updates_beta}_boltzmann_sims_{num_updates_boltzmann}_lr_{learning_rate:.3g}.pt'
    model_file = os.path.join(output_directory, f'ising_model_{file_suffix}')
    torch.save(obj=model, f=model_file)
    print(f'saved {model_file}')
    sim_fc_file = os.path.join(output_directory, f'sim_fc_{file_suffix}')
    torch.save(obj=sim_fc, f=sim_fc_file)
    print(f'saved {sim_fc_file}')
    fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{file_suffix}')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'saved {fc_rmse_file}')
print(f'done, time {time.time() - code_start_time:.3f}')