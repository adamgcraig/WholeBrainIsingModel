import os
import torch
import isingmodel
import time
import argparse

code_start_time = time.time()

int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="Sort feature-param point pairs by feature value. Bin them into groups of an equal number of points. Then calculate the mean and std. dev. of the param")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--file_name_fragment", type=str, default='group_training_and_individual_all_signed_params_rectangular_coords_times_beta', help="part of the input example files between example_[index]_ and _[validation|training]_example.pt, will also be part of the output file names")
parser.add_argument("-d", "--num_bins", type=int, default=1, help="number of bins into which to group the (feature value, param value) pairs")
parser.add_argument("-e", "--log10_features", action='store_true', default=False, help="Set this flag in order to take the base 10 logarithm of feature values.")
args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
file_name_fragment = args.file_name_fragment
print(f'file_name_fragment={file_name_fragment}')
num_bins = args.num_bins
print(f'num_bins={num_bins}')
log10_features = args.log10_features
print(f'log10_features={log10_features}')
if log10_features:
    log_str = 'log10_'
else:
    log_str = ''
def bin_feature_param_std_mean(feature:torch.Tensor, param:torch.Tensor, bin_size:int):
    if len( param.size() ) > 2:
        models_per_subject = param.size(dim=0)
        feature = feature.unsqueeze(dim=0).repeat( (models_per_subject, 1, 1) )
    feature = feature.flatten()
    param = param.flatten()
    feature_sorted, sort_indices = torch.sort(feature, descending=False)
    param_sorted = param[sort_indices]
    total_num_values = feature_sorted.numel()
    num_bins = total_num_values//bin_size
    num_values_in_bins = num_bins*bin_size
    feature_stds, feature_means = torch.std_mean(  feature_sorted[:num_values_in_bins].unflatten( dim=0, sizes=(num_bins, bin_size) ), dim=-1  )
    param_stds, param_means = torch.std_mean(  param_sorted[:num_values_in_bins].unflatten( dim=0, sizes=(num_bins, bin_size) ), dim=-1  )
    return feature_stds, feature_means, param_stds, param_means

node_feature_names = ['radius', 'inclination', 'azimuth'] + ['x', 'y', 'z'] + ['thickness', 'myelination', 'curvature', 'sulcus depth'] + ['mean SC']
edge_feature_names = ['SC'] + ['distance'] + [f'|{pname} difference|' for pname in node_feature_names]
node_feature_names_for_files = ['radius', 'inclination', 'azimuth'] + ['x', 'y', 'z'] + ['thickness', 'myelination', 'curvature', 'sulcus_depth'] + ['mean_SC']
edge_feature_names_for_files = ['SC'] + ['distance'] + [f'{pname}_difference' for pname in node_feature_names_for_files]

feature_types = ['node', 'edge']
param_names = ['h', 'J']
feature_name_lists = [node_feature_names, edge_feature_names]
feature_name_lists_for_files = [node_feature_names_for_files, edge_feature_names_for_files]
for (feature_type, param_name, feature_names, feature_names_for_files) in zip(feature_types, param_names, feature_name_lists, feature_name_lists_for_files):

    features_file = os.path.join(input_directory, f'{feature_type}_features_{file_name_fragment}.pt')
    features = torch.load(features_file)
    if log10_features:
        features = torch.log10( features + 1 - torch.min(features) )
    print('features size')
    print( features.size() )
    num_features = features.size(dim=-1)

    param_file = os.path.join(input_directory, f'{param_name}_{file_name_fragment}.pt')
    param = torch.load(param_file)
    print('param size')
    print( param.size() )

    num_pairs = param.numel()
    bin_size = num_pairs//num_bins
    print(f'using {num_bins} bins of size {bin_size}')
    for feature_index in range(num_features):
        feature_stds, feature_means, param_stds, param_means = bin_feature_param_std_mean(feature=features[:,:,feature_index], param=param, bin_size=bin_size)
        mean_std_correlation = isingmodel.get_pairwise_correlation(mat1=feature_means, mat2=param_stds)
        mean_mean_correlation = isingmodel.get_pairwise_correlation(mat1=feature_means, mat2=param_means)
        std_std_correlation = isingmodel.get_pairwise_correlation(mat1=feature_stds, mat2=param_stds)
        std_mean_correlation = isingmodel.get_pairwise_correlation(mat1=feature_stds, mat2=param_means)
        feature_name = feature_names[feature_index]
        print(f'time\t{time.time() - code_start_time:.3f}\tcorrelation for binned\t{param_name}\tvs\t{feature_name}\tmean-mean\t{mean_mean_correlation:.3g}\tmean-std\t{mean_std_correlation:.3g}\tstd-mean\t{std_mean_correlation:.3g}\tmean-std\t{std_std_correlation:.3g}')
        feature_name_for_file = feature_names_for_files[feature_index]
        std_mean_pairs_file = os.path.join(output_directory, f'binned_{param_name}_{log_str}{feature_name_for_file}_std_mean_{file_name_fragment}_bin_size_{bin_size}.pt')
        torch.save( (feature_stds, feature_means, param_stds, param_means), std_mean_pairs_file )
        # print(f'{time.time() - code_start_time:.3f}, saved {std_mean_pairs_file}')

print(f'{time.time() - code_start_time:.3f}, done')