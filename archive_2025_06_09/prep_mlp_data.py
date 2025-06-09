import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingmodel
from isingmodel import IsingModel

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description="Load the subject ID list, structural features data, and Ising models for all subjects, and place them in separate files.")
parser.add_argument("-a", "--subject_list_directory", type=str, default='E:\\data', help="directory from which we read the lists of subject IDs")
parser.add_argument("-b", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which we read the Ising model and structural features")
parser.add_argument("-c", "--output_directory", type=str, default='E:\\g2gcnn_examples', help="directory to which we write the separate h, J, extended node features, and extended edge features files")
parser.add_argument("-d", "--feature_file_name_fragment", type=str, default='group_training_and_individual_all', help="part of the file name between node_features_ or edge_features_ and .pt")
parser.add_argument("-e", "--model_file_name_fragment", type=str, default='group_training_and_individual_all_fold_1_betas_5_steps_1200_lr_0.01', help="part of the file name between ising_model_ and .pt")
args = parser.parse_args()
print('getting arguments...')
subject_list_directory = args.subject_list_directory
print(f'subject_list_directory={subject_list_directory}')
input_directory = args.input_directory
print(f'input_directory={input_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
feature_file_name_fragment = args.feature_file_name_fragment
print(f'feature_file_name_fragment={feature_file_name_fragment}')
model_file_name_fragment = args.model_file_name_fragment
print(f'model_file_name_fragment={model_file_name_fragment}')

def get_spherical_coords(rectangular_coords:torch.Tensor):
    x = rectangular_coords[:,:,0].unsqueeze(dim=-1)
    y = rectangular_coords[:,:,1].unsqueeze(dim=-1)
    z = rectangular_coords[:,:,2].unsqueeze(dim=-1)
    x_sq_plus_y_sq = x.square() + y.square()
    # radius = sqrt(x^2 + y^2 + z^2)
    radius = torch.sqrt( x_sq_plus_y_sq + z.square() )
    # inclination = arccos(z/radius)
    inclination = torch.arccos(z/radius)
    # azimuth = sgn(y)*arccos( x/sqrt(x^2+y^2) )
    azimuth = torch.sign(y) * torch.arccos( x/torch.sqrt(x_sq_plus_y_sq) )
    return torch.cat( (radius, inclination, azimuth), dim=-1 )

def load_structural_features(input_directory:str, file_name_fragment:str):

    node_features_file = os.path.join(input_directory, f'node_features_{file_name_fragment}.pt')
    node_features = torch.load(f=node_features_file)
    print( 'loaded node_features size', node_features.size() )

    edge_features_file = os.path.join(input_directory, f'edge_features_{file_name_fragment}.pt')
    edge_features = torch.load(f=edge_features_file)
    print( 'loaded edge_features size', edge_features.size() )

    # We want to use the mean SC values of edges to a given node as an additional node feature.
    # It is easier to calculate it here, before we have converted edge_features from square to upper triangular form.
    mean_sc = torch.mean(edge_features, dim=2)
    print( 'mean SC size', mean_sc.size() )

    # Select the part of edge features above the diagonal.
    num_nodes = edge_features.size(dim=2)
    triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
    edge_features = edge_features[:,triu_rows,triu_cols,:]
    print( 'triu edge_features size', edge_features.size() )

    # Extend node features with mean SC and spherical coordinates.
    rectangular_coords = node_features[:,:,:3]
    print( 'rectangular_coords size', rectangular_coords.size() )
    extended_node_features = torch.cat( (node_features, mean_sc), dim=-1 )
    print( 'extended_node_features size', extended_node_features.size() )

    row_node_features = extended_node_features[:,triu_rows,:]
    col_node_features = extended_node_features[:,triu_cols,:]
    distances = (row_node_features[:,:,:3] - col_node_features[:,:,:3]).square().sum(dim=-1, keepdim=True).sqrt()
    # Extend edge features with Euclidian distances between nodes and absolute differences between node features.
    extended_edge_features = torch.cat( (edge_features, distances, row_node_features, col_node_features), dim=-1 )
    print( 'extended_edge_features size', extended_edge_features.size() )

    return extended_node_features, extended_edge_features

def load_ising_model(input_directory:str, file_name_fragment:str):
    ising_model_file = os.path.join(input_directory, f'ising_model_{file_name_fragment}.pt')
    ising_model = torch.load(f=ising_model_file)
    beta = ising_model.beta
    h = ising_model.h
    J = ising_model.J
    print(f'time {time.time()-code_start_time:.3f}, loaded Ising model from {ising_model_file}')
    print( 'beta size', beta.size() )
    print( 'h size', h.size() )
    print( 'J size', J.size() )
    num_models = beta.size(dim=0)
    models_per_subject = ising_model.num_betas_per_target
    num_subjects = num_models//models_per_subject
    beta = torch.unflatten( beta, dim=0, sizes=(models_per_subject, num_subjects) )
    h = torch.unflatten( h, dim=0, sizes=(models_per_subject, num_subjects) )
    J = torch.unflatten( J, dim=0, sizes=(models_per_subject, num_subjects) )
    num_nodes = J.size(dim=-1)
    triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
    J = J[:,:,triu_rows,triu_cols]
    h = beta * h
    J = beta * J
    print('unflattened to')
    print( 'beta size', beta.size() )
    print( 'h size', h.size() )
    print( 'J triu size', J.size() )
    return beta, h, J, ising_model.num_beta_updates, ising_model.num_param_updates

subject_ids = ['group'] + hcp.load_subject_subset(directory_path=subject_list_directory, subject_subset='all', require_sc=True)
extended_node_features, extended_edge_features = load_structural_features(input_directory=input_directory, file_name_fragment=feature_file_name_fragment)
beta, h, J, num_param_updates, num_beta_updates = load_ising_model(input_directory=input_directory, file_name_fragment=model_file_name_fragment)

# We do not use beta for ML training but want to have it in a separate file so that we can load it and plot a histogram of the values in a Jupyter notebook.
# The local machine does not have enough GPU memory to load the entire IsingModel module.
beta_file = os.path.join(output_directory, f'beta_{model_file_name_fragment}_beta_updates_{num_beta_updates}_param_updates_{num_param_updates}.pt')
torch.save(obj=beta, f=beta_file)
print(f'saved {beta_file}')

models_per_subject = beta.size(dim=0)
num_subjects = beta.size(dim=1)
# In the node features, edge features, h, and J Tensors, index 0 is for the mean features and group model.
# Use clone() so that save() saves only the relevant item, not a view plus the full underlying Tensor.
for subject_index in range(num_subjects):
    subject_id = subject_ids[subject_index]
    extended_node_features_file = os.path.join(output_directory, f'node_features_subject_{subject_id}.pt')
    torch.save( obj=extended_node_features[subject_index,:,:].clone(), f=extended_node_features_file )
    print(f'saved {extended_node_features_file}')
    extended_edge_features_file = os.path.join(output_directory, f'edge_features_subject_{subject_id}.pt')
    torch.save( obj=extended_edge_features[subject_index,:,:].clone(), f=extended_edge_features_file )
    print(f'saved {extended_edge_features_file}')
    for model_index in range(models_per_subject):
        h_file = os.path.join(output_directory, f'h_subject_{subject_id}_model_{model_index}.pt')
        torch.save(obj=h[model_index,subject_index,:].clone(), f=h_file)
        print(f'saved {h_file}')
        J_file = os.path.join(output_directory, f'J_subject_{subject_id}_model_{model_index}.pt')
        torch.save( obj=J[model_index,subject_index,:].clone(), f=J_file )
        print(f'saved {J_file}')
print(f'time {time.time()-code_start_time:.3f}, done')