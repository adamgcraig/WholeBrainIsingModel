import os
import torch
import time
import argparse
import isingmodel
from isingmodel import IsingModel
from graph2graphcnn import UniformMultiLayerPerceptron

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Group (structural feature, Ising model parameter) pairs with similar feature values into bins, and train an MLP to predict mean and variance of the param bin given those of the feature bin. Try it with different bin sizes.")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--ts_file_name", type=str, default='binary_data_ts_all.pt', help="name of binarized data time series file")
parser.add_argument("-d", "--training_index_start", type=int, default=0, help="first index of training subjects")
parser.add_argument("-e", "--training_index_end", type=int, default=669, help="last index of training subjects + 1")
parser.add_argument("-p", "--validation_index_start", type=int, default=669, help="first index of validation subjects")
parser.add_argument("-q", "--validation_index_end", type=int, default=753, help="last index of validation subjects + 1")
parser.add_argument("-v", "--variance_file_name", type=str, default='data_fc_intra_subject_variances.pt', help="file to which to write the intra-subject, intra-node-pair variances of FC")
parser.add_argument("-r", "--range_file_name", type=str, default='data_fc_intra_subject_ranges.pt', help="file to which to write the intra-subject, intra-node-pair ranges of FC")
args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
ts_file_name = args.ts_file_name
print(f'ts_file_name={ts_file_name}')
training_index_start = args.training_index_start
print(f'training_index_start={training_index_start}')
training_index_end = args.training_index_end
print(f'training_index_end={training_index_end}')
validation_index_start = args.validation_index_start
print(f'validation_index_start={validation_index_start}')
validation_index_end = args.validation_index_end
print(f'validation_index_end={validation_index_end}')
variance_file_name = args.variance_file_name
print(f'variance_file_name={variance_file_name}')
range_file_name = args.range_file_name
print(f'range_file_name={range_file_name}')

data_ts_file = os.path.join(input_directory, ts_file_name)
data_ts = torch.load(data_ts_file)
print( f'time {time.time() - code_start_time:.3f}, loaded data_ts with size', data_ts.size() )
num_subjects, num_nodes, total_time_points = data_ts.size()
ts_per_subject = 4
ts_length = total_time_points//ts_per_subject
data_ts = torch.unflatten( data_ts, dim=-1, sizes=(ts_per_subject, ts_length) )
print( f'time {time.time() - code_start_time:.3f}, unflattened to size', data_ts.size() )
data_ts = torch.permute( data_ts, dims=(2, 0, 1, 3) )
print( f'time {time.time() - code_start_time:.3f}, permuted to scans x subjects x nodes x time-points', data_ts.size() )
data_state_product_mean = torch.matmul( data_ts, data_ts.transpose(dim0=-2, dim1=-1) )/ts_length
print( f'time {time.time() - code_start_time:.3f}, computed state product means with size', data_state_product_mean.size() )
data_state_product_mean = isingmodel.square_to_triu_pairs( square_pairs=torch.flatten(data_state_product_mean, start_dim=0, end_dim=1) ).unflatten( dim=0, sizes=(ts_per_subject, num_subjects) )
print( f'time {time.time() - code_start_time:.3f}, computed state product means with size', data_state_product_mean.size() )
data_state_mean = data_ts.mean(dim=-1)
print( f'time {time.time() - code_start_time:.3f}, computed state means with size', data_state_mean.size(), 'min', data_state_mean.min(), 'max', data_state_mean.max() )
data_fc = isingmodel.get_fc_binary( s_mean=data_state_mean.flatten(start_dim=0, end_dim=1), s_product_mean=data_state_product_mean.flatten(start_dim=0, end_dim=1), epsilon=0 ).unflatten( dim=0, sizes=(ts_per_subject, num_subjects) )
print( f'time {time.time() - code_start_time:.3f}, computed FCs with size', data_fc.size() )
data_fc_var = data_fc.var(dim=0)
print( f'time {time.time() - code_start_time:.3f}, computed variance of FC over each individual (subject, node pair) with size', data_fc_var.size() )
data_fc_var_file = os.path.join(output_directory, variance_file_name)
torch.save(obj=data_fc_var, f=data_fc_var_file)
print( f'time {time.time() - code_start_time:.3f}, saved {data_fc_var_file}' )
data_fc_range = data_fc.max(dim=0).values - data_fc.min(dim=0).values
print( f'time {time.time() - code_start_time:.3f}, computed eange of FC over each individual (subject, node pair) with size', data_fc_range.size() )
data_fc_range_file = os.path.join(output_directory, range_file_name)
torch.save(obj=data_fc_range, f=data_fc_range_file)
print( f'time {time.time() - code_start_time:.3f}, saved {data_fc_range_file}' )
print( f'time {time.time() - code_start_time:.3f}, done' )