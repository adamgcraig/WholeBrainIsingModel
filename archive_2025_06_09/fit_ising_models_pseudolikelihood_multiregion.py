# Based on the definitions of pseudolikelihood for Ising model parameters in
# H. Chau Nguyen, Riccardo Zecchina & Johannes Berg (2017)
# Inverse statistical problems: from the inverse Ising problem to data science,
# Advances in Physics, 66:3, 197-261, DOI: 10.1080/00018732.2017.1341604
# and
# Aurell, E., & Ekeberg, M. (2012).
# Inverse Ising inference using all the data.
# Physical review letters, 108(9), 090201.

import os
import torch
import time
import argparse
import hcpdatautils as hcp

code_start_time = time.time()
int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="Train a bunch of Ising models on fMRI time series data using pseudolikelihood maximization.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_batch', help="directory where we can find the time series files")
parser.add_argument("-l", "--learning_rate", type=float, default=0.00001, help="learning rate by which we multiply updates to weights and biases at each step")
parser.add_argument("-a", "--beta", type=float, default=0.5, help="factor we can use to adjust the chance of accepting a flip that increases the energy")
parser.add_argument("-t", "--threshold", type=float, default=0.1, help="threshold at which to binarize the fMRI data, in standard deviations above the mean")
parser.add_argument("-n", "--num_nodes", type=int, default=360, help="number of nodes to model")
parser.add_argument("-e", "--num_epochs", type=int, default=100000, help="number of times to repeat the training time series")
parser.add_argument("-r", "--num_reps", type=int, default=30, help="number of models to train for each subject")
parser.add_argument("-d", "--data_subset", type=str, default='validation', help="which data subset to use, either training, validation or testing")
parser.add_argument("-o", "--subjects_start", type=int, default=0, help="index of first subject in slice on which to train")
parser.add_argument("-x", "--subjects_end", type=int, default=83, help="index one past last subject in slice on which to train")
parser.add_argument("-v", "--print_every_seconds", type=int, default=10, help="minimum number of seconds between printouts of training status")
parser.add_argument("-c", "--batch_size", type=int, default=30, help="number of time points per batch on which to train in parallel")
args = parser.parse_args()
data_directory = args.data_directory
output_directory = args.output_directory
learning_rate = args.learning_rate
beta = args.beta
threshold = args.threshold
num_nodes = args.num_nodes
num_epochs = args.num_epochs
num_reps = args.num_reps
data_subset = args.data_subset
subjects_start = args.subjects_start
subjects_end = args.subjects_end
print_every_seconds = args.print_every_seconds
batch_size = args.batch_size
    
def binarize_ts_data(data_ts:torch.Tensor, data_threshold:float):
    data_std, data_mean = torch.std_mean(data_ts, dim=-2, keepdim=True)
    data_ts -= data_mean
    data_ts /= data_std
    data_ts = data_ts.flatten(start_dim=1, end_dim=-2)
    data_ts = torch.sign(data_ts - data_threshold)
    return data_ts

def get_negative_log_pseudolikelihood_loss(h:torch.Tensor, J:torch.Tensor, data_ts:torch.Tensor, node_index:int, beta:float):
    not_node_indices = torch.arange( data_ts.size(dim=-1), dtype=int_type, device=device ) != node_index
    nlpll = torch.mean(  torch.log( 1+torch.exp(-2*beta*data_ts[:,:,node_index]*(h[:,node_index] + torch.sum( data_ts[:,:,not_node_indices] * J[:,not_node_indices,node_index]))) )  )
    return nlpll

# Use this model to train the bias and input weights to a single node of the Ising model.
# The input to either get_prob_accept() or forward() should be a 3D stack of system states
# with dimensions (num_subjects,T,num_nodes).
# num_subjects and num_nodes are pre-specified when we create the model.
# T can be any value.
# beta is a fixed, scalar hyperparameter of the Ising model.
# The return value is a scalar, the mean negative log pseudolikelihood
# of getting the observed time series/state given model parameters h and J.
# The stack in the 3rd dimension is a stack of separate models, one for each subject.
# Consequently, h has size (num_subjects, 1, num_nodes),
# and J has size (num_subjects, num_nodes, num_nodes).
class IsingModelNegativeLogPseudoLikelihood(torch.nn.Module):
    def __init__(self, num_subjects:int, num_nodes:int, dtype=float_type, device=device, beta:float=beta):
        super(IsingModelNegativeLogPseudoLikelihood, self).__init__()
        self.beta = beta
        self.subjects_dim = -3
        self.time_dim = -2
        self.node_dim = -1
        self.h = torch.nn.Parameter( torch.randn( (num_subjects, 1, num_nodes), dtype=dtype, device=device ) )
        self.J = torch.nn.Parameter(  torch.randn( (num_subjects, num_nodes, num_nodes), dtype=dtype, device=device )  )
        self.log_sigmoid = torch.nn.LogSigmoid()
    def get_delta_h(self, state:torch.Tensor):
        J_no_diag = self.J - torch.diag_embed( torch.diagonal(self.J, dim1=self.time_dim, dim2=self.node_dim), dim1=self.time_dim, dim2=self.node_dim )
        return self.beta * 2.0 * state * ( self.h + torch.matmul(state, J_no_diag) )
    def get_prob_accept(self, state:torch.Tensor):
        return torch.exp( -self.get_delta_h(state) ).clamp(min=0.0, max=0.99)
    def forward(self, data_ts:torch.Tensor):
        return -torch.mean(  self.log_sigmoid( self.get_delta_h(data_ts) )  )

def run_ising_model_sim(ising_model:torch.nn.Module, num_time_points:int):
    print( f'NaNs in h: {ising_model.h.isnan().count_nonzero()}' )
    print( f'NaNs in J: {ising_model.J.isnan().count_nonzero()}' )
    num_subjects, _, num_nodes = ising_model.h.size()
    sim_ts = torch.zeros( (num_subjects, num_time_points, num_nodes), dtype=float_type, device=device )
    num_flips = 0
    state = 2.0*torch.randint( low=0, high=1, size=(num_subjects, 1, num_nodes), dtype=float_type, device=device ) - 1.0
    for t in range(num_time_points):
        prob_accept = ising_model.get_prob_accept(state)
        prob_accept[prob_accept.isnan()] = 0.0
        flip = torch.bernoulli(prob_accept)
        num_flips += flip.count_nonzero()
        state *= ( 1.0 - 2.0 * flip.float() )
        sim_ts[:,t,:] = state.squeeze()
    return sim_ts, num_flips

print('loading fMRI data...')
if data_subset == 'validation':
    subject_ids = hcp.load_validation_subjects(directory_path=data_directory)
else:
    subject_ids = hcp.load_training_subjects(directory_path=data_directory)
subject_ids = subject_ids[subjects_start:subjects_end]
data_ts = hcp.load_all_time_series_for_subjects(directory_path=data_directory, subject_ids=subject_ids, dtype=float_type, device=device)
data_ts = binarize_ts_data(data_ts=data_ts, data_threshold=threshold)
num_subjects, num_time_points, num_nodes = data_ts.size()
num_batches = num_time_points//batch_size

print('fitting...')
for rep in range(num_reps):
    last_time = time.time()
    imnlpl_fn = IsingModelNegativeLogPseudoLikelihood(num_subjects=num_subjects, num_nodes=num_nodes, dtype=float_type, device=device, beta=beta)
    optimizer = torch.optim.Adam( imnlpl_fn.parameters(), lr=learning_rate )
    for epoch in range(num_epochs):
        time_point_order = torch.randperm(num_time_points, dtype=int_type, device=device)
        data_ts_shuffled = data_ts[:,time_point_order,:]
        for batch in range(num_batches):
            batch_start = batch*batch_size
            batch_end = batch_start+batch_size
            data_ts_batch = data_ts_shuffled[:,batch_start:batch_end,:]
            data_ts_batch_num_pos = (data_ts_batch > 0.0).count_nonzero(dim=-2)
            data_ts_batch_num_all_neg = (data_ts_batch_num_pos == 0).count_nonzero()
            data_ts_batch_num_all_pos = (data_ts_batch_num_pos == batch_size).count_nonzero()
            if data_ts_batch_num_all_neg > 0:
                print(f'epoch: {epoch}\tbatch: {batch}\t(region, subject) pairs with only -1s in batch: {data_ts_batch_num_all_neg}')
            if data_ts_batch_num_all_pos > 0:
                print(f'epoch: {epoch}\tbatch: {batch}\t(region, subject) pairs with only +1s in batch: {data_ts_batch_num_all_pos}')
            optimizer.zero_grad()
            loss = imnlpl_fn(data_ts_batch)
            num_loss_nans = loss.isnan().count_nonzero()
            if num_loss_nans > 0:
                print(f'epoch: {epoch}\tbatch: {batch}\tNaNs in loss: {num_loss_nans}')
            loss.backward()
            optimizer.step()
            num_J_nans = imnlpl_fn.J.isnan().count_nonzero()
            if num_J_nans > 0:
                print(f'epoch: {epoch}\tbatch: {batch}\tNaNs in J: {num_J_nans}')
            num_h_nans = imnlpl_fn.h.isnan().count_nonzero()
            if num_h_nans > 0:
                print(f'epoch: {epoch}\tbatch: {batch}\tNaNs in h: {num_h_nans}')
            current_time = time.time()
            if current_time - last_time >= print_every_seconds:
                loss = imnlpl_fn(data_ts)
                print( f'epoch: {epoch}\tbatch: {batch}\tloss: {loss.item():.3g}\ttime: {current_time-code_start_time:.3f}' )
                last_time = current_time
    loss = imnlpl_fn(data_ts)
    print( f'epoch: {epoch}\tloss: {loss.item():.3g}\ttime: {current_time-code_start_time:.3f}' )
    # Save the trained model for the subject as two files, one for J and one for h.
    file_suffix = f'pl_data_{data_subset}_nodes_{num_nodes}_rep_{rep}_epochs_{num_epochs}_lr_{learning_rate}_threshold_{threshold}_beta_{beta}_start_{subjects_start}_end_{subjects_end}'
    print('saving Ising model J and h...')
    J_file_name = os.path.join(output_directory, f'J_{file_suffix}.pt')
    torch.save(imnlpl_fn.J, J_file_name)
    h_file_name = os.path.join(output_directory, f'h_{file_suffix}.pt')
    torch.save(imnlpl_fn.h, h_file_name)
    print('simulating...')
    with torch.no_grad():
        sim_ts, num_flips = run_ising_model_sim(ising_model=imnlpl_fn, num_time_points=num_time_points)
        data_fc = hcp.get_fc_batch(data_ts)
        sim_fc = hcp.get_fc_batch(sim_ts)
        fc_corr = hcp.get_triu_corr_batch(sim_fc, data_fc)
        fc_rmse = hcp.get_triu_corr_batch(sim_fc, data_fc)
        current_time = time.time()
        print(f'rep: {rep:.3g}\tflips: {num_flips}\tFC corr: {fc_corr.min():.3g}\tFC RMSE: {fc_rmse.max():.3g}\ttime: {current_time - code_start_time:.3g}')
current_time = time.time()
print(f'done, time {current_time - code_start_time}')