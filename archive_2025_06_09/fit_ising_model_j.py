import os
import torch
import time
import argparse
import isingmodelj
from isingmodelj import IsingModelJ

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
parser.add_argument("-f", "--data_file_name_part", type=str, default='group_training_and_individual_all', help="part of the file name between mean_state_product_ and .pt")
parser.add_argument("-w", "--output_file_name_part", type=str, default='short', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
parser.add_argument("-p", "--models_per_subject", type=int, default=2, help="number of separate models of each subject")
parser.add_argument("-l", "--sim_length", type=int, default=1200, help="number of simulation steps between updates")
parser.add_argument("-d", "--num_updates_scaling", type=int, default=1000000, help="maximum number of updates within which to find the optimal scaling factor, beta (We stop if we find it to within machine precision.)")
parser.add_argument("-u", "--updates_per_save", type=int, default=10, help="number of fitting updates of individual parameters between saves of an output file (In practice, these never perfectly converge, so we do not set any stopping criterion.)")
parser.add_argument("-s", "--num_saves", type=int, default=10, help="number of times we save a model to a file")
parser.add_argument("-r", "--learning_rate", type=float, default=0.0001, help="amount by which to multiply updates to the model parameters during the Euler step")
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
sim_length = args.sim_length
print(f'sim_length={sim_length}')
num_updates_scaling = args.num_updates_scaling
print(f'num_updates_scaling={num_updates_scaling}')
updates_per_save = args.updates_per_save
print(f'updates_per_save={updates_per_save}')
num_saves = args.num_saves
print(f'num_saves={num_saves}')
models_per_subject = args.models_per_subject
print(f'models_per_subject={models_per_subject}')
learning_rate = args.learning_rate
print(f'learning_rate={learning_rate}')

target_state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
target_state_product_mean = torch.load(target_state_product_mean_file)
print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', target_state_product_mean.size() )
target_state_product_mean = isingmodelj.triu_to_square_pairs( triu_pairs=torch.unsqueeze(target_state_product_mean, dim=0), diag_fill=0 )
print( f'time {time.time()-code_start_time:.3f}, converted to square format', target_state_product_mean.size() )
_, num_subjects, num_nodes, _ = target_state_product_mean.size()
model = IsingModelJ(  initial_J=target_state_product_mean.repeat( (models_per_subject, 1, 1, 1) ), initial_s=isingmodelj.get_neg_state(models_per_subject=models_per_subject, num_subjects=num_subjects, num_nodes=num_nodes, dtype=float_type, device=device)  )
print( f'time {time.time()-code_start_time:.3f}, initialized model with J of size', model.J.size(), 'and state of size', model.s.size() )
print('optimizing scale factor beta...')
beta, num_beta_updates_completed = model.rescale_J(target_state_product_means=target_state_product_mean, num_updates=num_updates_scaling, num_steps=sim_length, verbose=True)
print( f'time {time.time()-code_start_time:.3f}, done optimizing beta after {num_beta_updates_completed} iterations' )
model_file_fragment_beta_only = f'{data_file_name_part}_{output_file_name_part}_reps_{models_per_subject}_steps_{sim_length}_beta_updates_{num_beta_updates_completed}'
beta_file = os.path.join(output_directory, f'beta_for_j_{model_file_fragment_beta_only}.pt')
torch.save(obj=beta, f=beta_file)
print( f'time {time.time()-code_start_time:.3f}, saved {beta_file} with size', beta.size() )
model_file = os.path.join(output_directory, f'ising_model_j_{model_file_fragment_beta_only}.pt')
torch.save(obj=model, f=model_file)
print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
model_file_fragment = f'{model_file_fragment_beta_only}_lr_{learning_rate}'
for save_index in range(num_saves):
    model.fit_by_simulation(target_state_product_mean=target_state_product_mean, num_updates=updates_per_save, steps_per_update=sim_length, learning_rate=learning_rate, verbose=True)
    num_updates = (save_index+1)*updates_per_save
    model_file = os.path.join(output_directory, f'ising_model_j_{model_file_fragment}_J_updates_{num_updates}.pt')
    torch.save(obj=model, f=model_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {model_file}')
print(f'time {time.time()-code_start_time:.3f}, done')