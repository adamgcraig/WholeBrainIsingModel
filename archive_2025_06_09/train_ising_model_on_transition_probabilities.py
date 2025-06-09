import os
import torch
import time
import argparse
# import hcpdatautils as hcp

parser = argparse.ArgumentParser(description="Train an Ising model to predict the probability of a transition based on empirical transition counts.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-s", "--data_subset", type=str, default='validation', help="the subset of subjects over which to search for unique states")
parser.add_argument("-e", "--num_epochs", type=int, default=100, help="number of times to iterate over all starting states")
parser.add_argument("-v", "--save_interval", type=int, default=10, help="number of epochs between saves")
parser.add_argument("-b", "--batch_size", type=int, default=4006, help="number of starting states to use in a single training step")
parser.add_argument("-l", "--learning_rate", type=float, default=0.000000001, help="learning rate to use when training")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
data_subset = args.data_subset
print(f'data_subset={data_subset}')
num_epochs = args.num_epochs
print(f'num_epochs={num_epochs}')
save_interval = args.save_interval
print(f'save_interval={save_interval}')
batch_size = args.batch_size
print(f'batch_size={batch_size}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
file_suffix = f'{data_subset}.pt'
unique_states_file = os.path.join(output_directory, f'unique_states_{file_suffix}')
unique_states = torch.load(unique_states_file)
print(f'loaded {unique_states_file}, time {time.time() - code_start_time:.3f}')
unique_state_counts_file = os.path.join(output_directory, f'unique_state_counts_{file_suffix}')
unique_state_counts = torch.load(unique_state_counts_file)
print(f'loaded {unique_state_counts_file}, time {time.time() - code_start_time:.3f}')
state_transition_counts_file = os.path.join(output_directory, f'state_transition_counts_{file_suffix}')
state_transition_counts = torch.load(state_transition_counts_file)
print(f'loaded {state_transition_counts_file}, time {time.time() - code_start_time:.3f}')
flip_number_counts_file = os.path.join(output_directory, f'flip_number_counts_{file_suffix}')
flip_number_counts = torch.load(flip_number_counts_file)
print(f'loaded {flip_number_counts_file}, time {time.time() - code_start_time:.3f}')
num_states, num_nodes = unique_states.size()
print(f'found {num_states} unique states for {num_nodes} nodes')
num_batches = num_states//batch_size
triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, dtype=int_type, device=device)
triu_rows = triu_indices[0]
triu_cols = triu_indices[1]
num_edges = triu_rows.numel()
num_params = num_nodes + num_edges
ising_model = torch.nn.Linear(in_features=num_params, out_features=1, bias=False, dtype=float_type, device=device)
optimizer = torch.optim.Adam( params=ising_model.parameters(), lr=learning_rate )
start_states = torch.zeros( size=(batch_size, num_params), dtype=float_type, device=device )
end_states = torch.zeros_like(start_states)
for epoch in range(num_epochs):
    state_order = torch.randperm(n=num_states, dtype=int_type, device=device)
    loss_min = 10e10
    loss_sum = 0
    loss_max = 0
    for batch in range(num_batches):
        batch_start = batch_size*batch
        batch_end = batch_start + batch_size
        state_indices = state_order[batch_start:batch_end]
        start_states[:,:num_nodes] = 2.0 * unique_states[state_indices,:].float() - 1.0
        torch.multiply( input=start_states[:,triu_rows], other=start_states[:,triu_cols], out=start_states[:,num_nodes:] )
        start_state_counts = unique_state_counts[state_indices].float()
        state_counts_total = torch.sum(start_state_counts)
        for flip_node_index in range(num_nodes):
            node_flip_counts = state_transition_counts[state_indices,flip_node_index].float()
            empirical_probabilities = torch.clamp_max( input=num_nodes*node_flip_counts/start_state_counts, max=1.0 ).unsqueeze(dim=1)
            end_states[:,:num_nodes] = start_states[:,:num_nodes]
            end_states[:,flip_node_index] *= -1
            torch.multiply( input=end_states[:,triu_rows], other=end_states[:,triu_cols], out=end_states[:,num_nodes:] )
            states_diff = end_states - start_states
            model_probabilities = torch.clamp_max( input=ising_model(states_diff).exp(), max=1.0 )
            batch_loss = torch.sum(  start_state_counts*torch.square( (model_probabilities - empirical_probabilities) )  )/state_counts_total
            batch_loss.backward()
            optimizer.step()
            batch_loss_item = batch_loss.sqrt().item()
            loss_min = min(loss_min, batch_loss_item)
            loss_sum += batch_loss_item
            loss_max = max(loss_max, batch_loss_item)
            # print(f'{epoch},{batch},{flip_node_index},{batch_loss:.3g}')
    print(f'{epoch},\tloss min {loss_min:.3g},\tmean {loss_sum/(num_batches*num_nodes):.3g},\tmax {loss_max:.3g},\ttime {time.time() - code_start_time:.3f}')
    if ( (epoch+1) % save_interval == 0 ) or (epoch+1 == num_epochs):
        ising_model_file = os.path.join(output_directory, f'ising_model_group_{data_subset}_epochs_{epoch+1}_batch_{batch_size}_lr_{learning_rate:.3g}.pt')
        torch.save( obj=ising_model.weight.flatten(), f=ising_model_file )
        print(f'saved {ising_model_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')