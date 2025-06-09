import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Simulate several Ising models of a single subject with different values of beta and compare flip rate and RMSE between simulated and target FC.")
parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
parser.add_argument("-c", "--data_file_name_part", type=str, default='all_unit_scale', help="part of the output file name between mean_state_ or mean_state_product_ and .pt")
parser.add_argument("-d", "--output_file_name_part", type=str, default='short', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
parser.add_argument("-e", "--models_per_subject", type=int, default=1, help="number of separate models of each subject")
parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
parser.add_argument("-g", "--num_updates_beta", type=int, default=1000000, help="maximum number of updates within which to find the optimal inverse temperature beta (We stop if we find it to within machine precision.)")
parser.add_argument("-i", "--updates_per_save", type=int, default=1000, help="number of fitting updates of individual parameters between saves of an output file (In practice, these never perfectly converge, so we do not set any stopping criterion.)")
parser.add_argument("-j", "--num_saves", type=int, default=1000, help="number of times we save a model to a file")
parser.add_argument("-l", "--center_cov", action='store_true', default=False, help="Set this flag in order to initialize J to the centered covariance instead of the uncentered covariance (state mean product).")
parser.add_argument("-m", "--use_inverse_cov", action='store_true', default=False, help="Set this flag in order to initialize J to the inverse covariance instead of the covariance.")
parser.add_argument("-o", "--beta", type=float, default=0.0001, help="value of beta to use for all models")
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
num_updates_beta = args.num_updates_beta
print(f'num_updates_scaling={num_updates_beta}')
updates_per_save = args.updates_per_save
print(f'updates_per_save={updates_per_save}')
num_saves = args.num_saves
print(f'num_saves={num_saves}')
models_per_subject = args.models_per_subject
print(f'models_per_subject={models_per_subject}')
center_cov = args.center_cov
print(f'center_cov={center_cov}')
use_inverse_cov = args.use_inverse_cov
print(f'use_inverse_cov={use_inverse_cov}')
beta = args.beta
print(f'min_beta={beta}')

target_state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
target_state_mean = torch.load(target_state_mean_file)
print( f'time {time.time()-code_start_time:.3f}, loaded target_state_mean with size', target_state_mean.size() )
target_state_mean = torch.flatten(target_state_mean, start_dim=0, end_dim=1)
print( f'time {time.time()-code_start_time:.3f}, flattened scan and subject dimensions', target_state_mean.size() )
target_state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
target_state_product_mean = torch.load(target_state_product_mean_file)
print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', target_state_product_mean.size() )
target_state_product_mean = torch.flatten(target_state_product_mean, start_dim=0, end_dim=1)
print( f'time {time.time()-code_start_time:.3f}, flattened scan and subject dimensions', target_state_product_mean.size() )
num_targets, num_nodes = target_state_mean.size()
copy_beta = beta * isingmodellight.get_unit_beta(models_per_subject=models_per_subject, num_subjects=num_targets, dtype=float_type, device=device)
s = isingmodellight.get_neg_state(models_per_subject=models_per_subject, num_subjects=num_targets, num_nodes=num_nodes, dtype=float_type, device=device)
h = target_state_mean.unsqueeze(dim=0).repeat( (models_per_subject, 1, 1) )
init_J = target_state_product_mean
if center_cov:
    init_J = init_J - target_state_mean.unsqueeze(dim=-1) * target_state_mean.unsqueeze(dim=-2)
if use_inverse_cov:
    init_J = torch.linalg.inv(init_J)
init_J -= torch.diag_embed( torch.diagonal(init_J, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
J = init_J.unsqueeze(dim=0).repeat( (models_per_subject, 1, 1, 1) )
model = IsingModelLight(beta=copy_beta, J=J, h=h, s=s)
print( f'time {time.time()-code_start_time:.3f}, initialized model with beta of size', model.beta.size(), ', h of size', model.h.size(), ', J of size', model.J.size(), ', state of size', model.s.size() )
print('running simulations...')
target_fc = isingmodellight.get_fc(state_mean=target_state_mean, state_product_mean=target_state_product_mean)
sim_fc, flip_rate = model.simulate_and_record_fc_and_flip_rate(num_steps=sim_length)
fc_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=target_fc, mat2=sim_fc)
print( f'time {time.time()-code_start_time:.3f}, done with simulation' )
if center_cov:
    center_cov_str = 'centered'
else:
    center_cov_str = 'uncentered'
if use_inverse_cov:
    cov_str = 'inverse_cov'
else:
    cov_str = 'cov'
model_file_fragment_beta_only = f'{data_file_name_part}_{output_file_name_part}_reps_{models_per_subject}_steps_{sim_length}_{cov_str}_{center_cov_str}_beta_{beta:.3g}'
fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{model_file_fragment_beta_only}.pt')
torch.save( obj=fc_rmse, f=fc_rmse_file )
print( f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file} with size', fc_rmse.size() )
flip_rate_file = os.path.join(output_directory, f'flip_rate_{model_file_fragment_beta_only}.pt')
torch.save( obj=flip_rate, f=flip_rate_file )
print( f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file} with size', flip_rate.size() )
print(f'time {time.time()-code_start_time:.3f}, done')