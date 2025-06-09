import os
import torch
import time
import argparse
from isingutilsslow import IsingModel
from isingutilsslow import get_fc_binarized
from isingutilsslow import get_triu_flattened

parser = argparse.ArgumentParser(description="Fit multiple Ising models to the concatenated fMRI data of all subjects.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-s", "--model_set", type=str, default='group_training', help="'group_training', 'subjects_training', or 'subjects_validation'")
parser.add_argument("-b", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate by which we multiply updates to weights and biases at each step")
parser.add_argument("-t", "--threshold", type=str, default='0.1', help="threshold at which to binarize the fMRI data, in standard deviations above the mean, or the string 'median'")
parser.add_argument("-n", "--num_nodes", type=int, default=21, help="number of nodes to model")
parser.add_argument("-r", "--reps_per_batch", type=int, default=1000, help="number of replicas of each subject to train in a single batch")
parser.add_argument("-c", "--num_batches", type=int, default=1, help="number of batches to train serially and save to separate files")
parser.add_argument("-w", "--window_length", type=int, default=75, help="number of time points between model parameter updates")
parser.add_argument("-e", "--num_epochs", type=int, default=1, help="number of epochs for which we fitted the Ising model")
parser.add_argument("-z", "--test_length", type=int, default=48000, help="number of sim steps in test run")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
model_set = args.model_set
print(f'model_set={model_set}')
beta = args.beta
print(f'beta={beta:.3g}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate:.3g}')
threshold_str = args.threshold
print(f'threshold={threshold_str}')
num_nodes = args.num_nodes
print(f'num_nodes={num_nodes}')
reps_per_batch = args.reps_per_batch
print(f'reps_per_batch={reps_per_batch}')
num_batches = args.num_batches
print(f'num_batches={num_batches}')
window_length = args.window_length
print(f'window_length={window_length}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
test_length = args.test_length
print(f'test_length={test_length}')

def get_num_nan(mat:torch.Tensor):
    return torch.count_nonzero( torch.isnan(mat) )

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    device = torch.device('cuda')
    # device = torch.device('cpu')
    # Make a string of the parts of the file name that are constant across all output files.
    file_const_params = f'nodes_{num_nodes}_window_{window_length}_lr_{learning_rate:.3f}_threshold_{threshold_str}_beta_{beta:.3f}_{model_set}_reps_{reps_per_batch}'
    max_diff_fim = torch.zeros( (test_length,), dtype=float_type, device=device )
    max_diff_fc = torch.zeros( (test_length,), dtype=float_type, device=device )
    for batch_index in range(num_batches):
        file_suffix = f'{file_const_params}_batch_{batch_index}_epoch_{num_epochs}'
        ising_model_file = os.path.join(output_directory, f'ising_model_{file_suffix}.pt')
        ising_model = torch.load(ising_model_file)
        print(ising_model)
        ising_model.inv_betaneg2 = 1.0/(-2.0*beta)
        # fc, fim = ising_model.simulate_and_record_fc_and_fim_faster(num_steps=test_length)
        state_sum = torch.zeros_like(ising_model.s)
        cross_sum = state_sum[:,:,:,None] * state_sum[:,:,None,:]
        reps_per_subject, num_subjects, num_nodes = ising_model.s.size()
        num_params = num_nodes + ( num_nodes*(num_nodes-1) )//2
        params = torch.zeros( (reps_per_subject, num_subjects, num_params), dtype=float_type, device=device )
        param_sum = torch.zeros_like(params)
        param_cross_sum = param_sum[:,:,:,None] * param_sum[:,:,None,:]
        fc_old = torch.zeros_like(cross_sum)
        fim_old = torch.zeros_like(param_cross_sum)
        for step in range(test_length):
            num_steps = step+1
            # s_pre = self.s.clone()
            ising_model.do_ising_model_step()
            state = ising_model.s
            ising_model.s_mean += state# B x N x 1
            cross_state = state[:,:,:,None] * state[:,:,None,:]# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            cross_sum += cross_state
            params[:,:,:num_nodes] = state
            params[:,:,num_nodes:] = get_triu_flattened(cross_state)
            param_sum += params
            param_cross_sum += params[:,:,:,None] * params[:,:,None,:]
            state_mean = state_sum/num_steps
            state_cov = cross_sum/num_steps
            param_mean = param_sum/num_steps
            param_cov = param_cross_sum/num_steps
            fc = get_fc_binarized(s_mean=state_mean, s_cov=state_cov)
            fim = param_cov - param_mean[:,:,:,None] * param_mean[:,:,None,:]
            max_diff_fc[step] = torch.maximum( max_diff_fc[step], (fc - fc_old).abs().max() )
            max_diff_fim[step] = torch.maximum( max_diff_fim[step], (fim - fim_old).abs().max() )
            fc_old = fc.clone()
            fim_old = fim.clone()
        # Save the final FC and FIM for this batch.
        fc_file = os.path.join(output_directory, f'fc_{file_suffix}_steps_{test_length}.pt')
        torch.save(obj=fc, f=fc_file)
        print(f'saved {fc_file}, time {time.time()-code_start_time:.3f}')
        fim_file = os.path.join(output_directory, f'fim_{file_suffix}_steps_{test_length}.pt')
        torch.save(obj=fim, f=fim_file)
        print(f'saved {fim_file}, time {time.time()-code_start_time:.3f}')
    # Save the maximum changes in FC and FIM over all batches at each step of the simulation.
    max_diff_fc_file = os.path.join(output_directory, f'max_diff_fc_{file_const_params}_epoch_{num_epochs}_steps_{test_length}.pt')
    torch.save(obj=max_diff_fc, f=max_diff_fc_file)
    max_diff_fim_file = os.path.join(output_directory, f'max_diff_fim_{file_const_params}_epoch_{num_epochs}_steps_{test_length}.pt')
    torch.save(obj=max_diff_fim, f=max_diff_fim_file)
    print(f'done, time {time.time() - code_start_time:.3f}')