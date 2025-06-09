import os
import torch
import time
import argparse
import math
import isingmodel
from isingmodel import IsingModel

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')
num_nodes = 360
num_node_features = 4 + 3 + 1# thickness, myelination, curvature, sulcus depth + x, y, z + SC sum
num_edge_features = 1 + 2*num_node_features# SC + node features of first endpoint + node features of second endpoint
num_h_features = 1
num_J_features = 1

parser = argparse.ArgumentParser(description="Measure correlations between structural features and parameters of fitted Ising models.")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--file_name_fragment", type=str, default='group_training_and_individual_all', help="part of the input example files between example_[index]_ and _[validation|training]_example.pt, will also be part of the output file names")
parser.add_argument("-d", "--training_index_start", type=int, default=1, help="first index of training subjects")
parser.add_argument("-e", "--training_index_end", type=int, default=670, help="last index of training subjects + 1")
args = parser.parse_args()
print('getting arguments...')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
file_name_fragment = args.file_name_fragment
print(f'file_name_fragment={file_name_fragment}')
training_index_start = args.training_index_start
print(f'training_index_start={training_index_start}')
training_index_end = args.training_index_end
print(f'training_index_end={training_index_end}')

def save_pairs(node_features:torch.Tensor, edge_features:torch.Tensor, h:torch.Tensor, J:torch.Tensor, file_name_fragment:str):
    models_per_subject, num_training_subjects, num_nodes, num_node_features = node_features.size()
    feature_h_mean_triple = torch.zeros( size=(models_per_subject, num_training_subjects, num_nodes, 3) )
    feature_h_mean_triple[:,:,:,1] = h
    feature_h_mean_triple[:,:,:,2] = target_state_means
    h_flat = h.flatten()
    means_flat = target_state_means.flatten()
    for feature_index in range(num_node_features):
        node_feature = node_features[:,:,:,feature_index]
        feature_h_mean_triple[:,:,:,0] = node_feature
        feature_file = os.path.join(output_directory, f'node_feature_{feature_index}_h_mean_triples_{file_name_fragment}.pt')
        torch.save(obj=feature_h_mean_triple, f=feature_file)
        print(f'saved {feature_file}')
        node_feature_flat = node_feature.flatten()
        feature_h_correlation = isingmodel.get_pairwise_correlation(mat1=h_flat, mat2=node_feature_flat)
        feature_mean_correlation = isingmodel.get_pairwise_correlation(mat1=means_flat, mat2=node_feature_flat)
        print( f'node feature {feature_index} x h correlation {feature_h_correlation:.3g}, node feature {feature_index} x mean state correlation {feature_mean_correlation:.3g}' )

    models_per_subject, num_training_subjects, num_pairs, num_edge_features = edge_features.size()
    feature_J_mean_triple = torch.zeros( size=(models_per_subject, num_training_subjects, num_pairs, 3) )
    feature_J_mean_triple[:,:,:,1] = J
    feature_J_mean_triple[:,:,:,2] = target_state_product_means
    J_flat = J.flatten()
    means_flat = target_state_product_means.flatten()
    for feature_index in range(num_edge_features):
        edge_feature = edge_features[:,:,:,feature_index]
        feature_J_mean_triple[:,:,:,0] = edge_feature
        feature_file = os.path.join(output_directory, f'edge_feature_{feature_index}_J_mean_triples_{file_name_fragment}.pt')
        torch.save(obj=feature_J_mean_triple, f=feature_file)
        print(f'saved {feature_file}')
        edge_feature_flat = edge_feature.flatten()
        feature_J_correlation = isingmodel.get_pairwise_correlation(mat1=J_flat, mat2=edge_feature_flat)
        feature_mean_correlation = isingmodel.get_pairwise_correlation(mat1=means_flat, mat2=edge_feature_flat)
        print( f'edge feature {feature_index} x J correlation {feature_J_correlation:.3g}, edge feature {feature_index} x mean state correlation {feature_mean_correlation:.3g}' )

    feature_pair_J_mean_quad = torch.zeros( size=(models_per_subject, num_training_subjects, num_pairs, 4) )
    feature_pair_J_mean_quad[:,:,:,2] = J
    feature_pair_J_mean_quad[:,:,:,3] = target_state_product_means
    for feature_index in range(num_node_features):
        node_feature_row = node_features[:,:,triu_rows,feature_index]
        node_feature_col = node_features[:,:,triu_cols,feature_index]
        feature_pair_J_mean_quad[:,:,:,0] = node_feature_row
        feature_pair_J_mean_quad[:,:,:,1] = node_feature_col
        feature_file = os.path.join(output_directory, f'node_feature_pair_{feature_index}_J_mean_quads_{file_name_fragment}.pt')
        # torch.save(obj=feature_pair_J_mean_quad, f=feature_file)
        # print(f'saved {feature_file}')
        node_feature_diff = torch.abs(node_feature_col - node_feature_row)
        node_feature_diff_flat = node_feature_diff.flatten()
        feature_diff_J_correlation = isingmodel.get_pairwise_correlation(mat1=J_flat, mat2=node_feature_diff_flat)
        feature_diff_mean_correlation = isingmodel.get_pairwise_correlation(mat1=means_flat, mat2=node_feature_diff_flat)
        print( f'node feature {feature_index} pair diff x J correlation {feature_diff_J_correlation:.3g}, node feature {feature_index} pair diff x mean state correlation {feature_diff_mean_correlation:.3g}' )

ising_model_file = os.path.join(input_directory, f'ising_model_{file_name_fragment}_fold_1_betas_5_steps_1200_lr_0.01.pt')
ising_model = torch.load(f=ising_model_file)
triu_rows, triu_cols = ising_model.get_triu_indices_for_products()
beta = ising_model.beta
print( 'beta size', beta.size() )
h = ising_model.h
print( 'h size', h.size() )
J = ising_model.J[:,triu_rows,triu_cols]
print( 'J size', J.size() )
target_state_means = ising_model.target_state_means
print( 'target_state_mean size', target_state_means.size() )
target_state_product_means = ising_model.target_state_product_means[:,triu_rows,triu_cols]
print( 'target_state_product_mean size', target_state_product_means.size() )
print(f'time {time.time()-code_start_time:.3f}, done')

node_features_file = os.path.join(input_directory, f'node_features_{file_name_fragment}.pt')
node_features = torch.load(f=node_features_file)
print( 'node_features size', node_features.size() )
edge_features_file = os.path.join(input_directory, f'edge_features_{file_name_fragment}.pt')
edge_features = torch.load(f=edge_features_file)
edge_features = edge_features[:,triu_rows,triu_cols,:]
print( 'edge_features size', edge_features.size() )

num_subjects = node_features.size(dim=0)
num_models = h.size(dim=0)
models_per_subject = num_models//num_subjects
node_features = node_features.unsqueeze(dim=0).repeat( (models_per_subject,1,1,1) )
edge_features = edge_features.unsqueeze(dim=0).repeat( (models_per_subject,1,1,1) )
beta = beta.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
h = h.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
J = J.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
target_state_means = target_state_means.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )
target_state_product_means = target_state_product_means.unflatten( dim=0, sizes=(models_per_subject, num_subjects) )

num_training_subjects = training_index_end - training_index_start
node_features = node_features[:,training_index_start:training_index_end,:,:]
edge_features = edge_features[:,training_index_start:training_index_end,:,:]
beta = beta[:,training_index_start:training_index_end,:]
h = h[:,training_index_start:training_index_end,:]
J = J[:,training_index_start:training_index_end,:]
target_state_means = target_state_means[:,training_index_start:training_index_end,:]
target_state_product_means = target_state_product_means[:,training_index_start:training_index_end,:]

print('version without multiplying h or J by beta')
save_pairs(node_features=node_features, edge_features=edge_features, h=h, J=J, file_name_fragment=file_name_fragment)
print('version with multiplying h and J by beta')
save_pairs(node_features=node_features, edge_features=edge_features, h=beta*h, J=beta*J, file_name_fragment=f'beta_multiplied_{file_name_fragment}')

print(f'time {time.time()-code_start_time:.3f}, done')