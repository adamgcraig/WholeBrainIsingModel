import os
import torch
import hcpdatautils as hcp
import isingutils as ising
import time
import argparse

start_time = time.time()
int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="Run a single of Ising model multiple times, and compare the time series FC matrices to the real data FC.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-m", "--model_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the model")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory to which to write the FC RMSE and correlation files")
parser.add_argument("-a", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-w", "--batch_size", type=int, default=10, help="number of models to simulate in parallel at a time; more gets you done faster but risks running out of CUDA memory")
parser.add_argument("-l", "--sim_length", type=int, default=None, help="number of steps for which to run the simulation, defaults to the same number as the number of timepoints in the data")
parser.add_argument("-p", "--ising_param_string", type=str, default='window_length_test_nodes_360_epochs_1800_max_window_2400_lr_0.001_threshold_0.100', help="parameter string of the Ising model file, the characters between h_/J_ and _[subject ID]")
parser.add_argument("-s", "--subject_id", type=int, default=516742, help="if we want one subject, set it to the ID of the subject of the model")
parser.add_argument("-d", "--data_subset", type=str, default=None, help="if we want multiple subjects, set it to training or validation")
parser.add_argument("-e", "--num_sims", type=int, default=1000, help="number of sims to run in parallel")
parser.add_argument("-n", "--model_index", type=int, default=0, help="index of model to run in batch of models loaded from files")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
model_directory = args.model_directory
print(f'model_directory={model_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
beta = args.beta
print(f'beta={beta:.3g}')
threshold = args.threshold
print(f'threshold={threshold:.3g}')
batch_size = args.batch_size
print(f'batch_size={batch_size}')
sim_length = args.sim_length
print(f'sim_length={sim_length}')
ising_param_string = args.ising_param_string
print(f'ising_param_string={ising_param_string}')
subject_id = args.subject_id
print(f'subject_id={subject_id}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
num_sims = args.num_sims
print(f'num_sims={num_sims}')
model_index = args.model_index
print(f'model_index={model_index}')
num_batches = num_sims//batch_size + int(num_sims % batch_size > 0)

with torch.no_grad():
    subject_string = data_subset
    if data_subset == 'training':
        subject_list = hcp.load_training_subjects(data_directory)
    elif data_subset == 'validation':
        subject_list = hcp.load_validation_subjects(data_directory)
    elif data_subset == 'testing':
        subject_list = hcp.load_testing_subjects(data_directory)
    elif data_subset == 'all':
        subject_list = hcp.load_training_subjects(data_directory) + hcp.load_validation_subjects(data_directory) + hcp.load_testing_subjects(data_directory)
    else:
        subject_list = [subject_id]
        subject_string = str(subject_id)
    num_subjects = len(subject_list)
    fc_rmse_set = torch.zeros( (num_subjects, num_sims), dtype=float_type, device=device )
    fc_corr_set = torch.zeros( (num_subjects, num_sims), dtype=float_type, device=device )
    for subject_index in range(num_subjects):
        subject = subject_list[subject_index]
        print(f'subject {subject} ({subject_index} of {num_subjects})')
        print('loading data...')
        data_ts = ising.standardize_and_binarize_ts_data(  ts=hcp.load_all_time_series_for_subject(directory_path=data_directory, subject_id=subject, dtype=float_type, device=device), threshold=threshold  ).flatten(start_dim=0, end_dim=1)
        num_time_points, num_nodes_data = data_ts.size()
        if not sim_length:
            sim_length = num_time_points
            print(f'set sim_length={sim_length}')
        print(f'data ts size {num_time_points} x {num_nodes_data}')
        print('loading models...')
        h = torch.load( os.path.join(model_directory, f'h_{ising_param_string}_subject_{subject}.pt') )[model_index,:].unsqueeze(dim=0)
        print( 'h size', h.size() )
        J = torch.load( os.path.join(model_directory, f'J_{ising_param_string}_subject_{subject}.pt') )[model_index,:,:].unsqueeze(dim=0)
        print( 'J size', J.size() )
        num_nodes = h.size(dim=-1)
        print('allocating time series Tensor...')
        sim_ts = torch.zeros( (batch_size, sim_length+2, num_nodes), dtype=float_type, device=device )
        sim_ts[:,-2,:] = -1.0
        sim_ts[:,-1,:] = 1.0
        for batch in range(num_batches):
            batch_start = batch*batch_size
            batch_end = min(batch_start + batch_size, num_sims)
            current_batch_size = batch_end - batch_start
            current_sim_ts = sim_ts[:current_batch_size,:,:]
            print(f'batch {batch} of {num_batches}, {batch_start} to {batch_end}')
            print('initializing state...')
            s = ising.get_random_state(batch_size=current_batch_size, num_nodes=num_nodes, dtype=float_type, device=device)
            print('running sim...')
            current_sim_ts, s = ising.run_batched_balanced_metropolis_sim(sim_ts=current_sim_ts, J=J, h=h, s=s, num_steps=sim_length, beta=beta)
            sim_fc = hcp.get_fc_batch(current_sim_ts)
            data_fc = hcp.get_fc(data_ts).unsqueeze(dim=0).repeat( (current_batch_size, 1, 1) )
            fc_rmse_set[subject_index,batch_start:batch_end] = hcp.get_triu_rmse_batch(sim_fc, data_fc)
            fc_corr_set[subject_index,batch_start:batch_end] = hcp.get_triu_corr_batch(sim_fc, data_fc)
            elapsed_time = time.time() - start_time
            rmse_so_far = fc_rmse_set[subject_index,:batch_end]
            best_rmse = torch.min(rmse_so_far)
            mean_rmse = torch.mean(rmse_so_far)
            worst_rmse = torch.max(rmse_so_far)
            corr_so_far = fc_corr_set[subject_index,:batch_end]
            best_corr = torch.max(corr_so_far)
            mean_corr = torch.mean(corr_so_far)
            worst_corr = torch.min(corr_so_far)
            print(f'worst, mean, best RMSE ({worst_rmse:.3g},{mean_rmse:.3g},{best_rmse:.3g}), correlation ({worst_corr:.3g},{mean_corr:.3g},{best_corr:.3g}) time {elapsed_time:.3f}')
        did_not_flip = torch.var(sim_ts[:,:-2,:], dim=1) == 0.0
        num_no_flips = torch.count_nonzero(did_not_flip)
        num_state = torch.numel(did_not_flip)
        print(f'for {num_time_points} time points, regions that did not flip at any point in the simulation: {num_no_flips} of {num_state}, or {100*num_no_flips/num_state:.3g} percent')
    results_file_suffix = f'{ising_param_string}_model_{model_index}_sims_{num_sims}_length_{sim_length}_subject_{subject_string}'
    rmse_file = os.path.join(output_directory, f'fc_rmse_{results_file_suffix}.pt')
    torch.save(fc_rmse_set, rmse_file)
    print(rmse_file)
    corr_file = os.path.join(output_directory, f'fc_corr_{results_file_suffix}.pt')
    print(corr_file)
    torch.save(fc_corr_set, corr_file)
    elapsed_time = time.time() - start_time
    best_rmse = torch.min(fc_rmse_set)
    mean_rmse = torch.mean(fc_rmse_set)
    worst_rmse = torch.max(fc_rmse_set)
    best_corr = torch.max(fc_corr_set)
    mean_corr = torch.mean(fc_corr_set)
    worst_corr = torch.min(fc_corr_set)
    print(f'done, (worst, mean, best) correlation ({worst_corr:.3g}, {mean_corr:.3g}, {best_corr:.3g}) RMSE ({worst_rmse:.3g}, {mean_rmse:.3g}, {best_rmse:.3g}) time {elapsed_time:.3f}')