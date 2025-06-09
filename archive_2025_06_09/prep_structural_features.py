import os
import torch
import time
import argparse
import hcpdatautils as hcp
import isingmodel

parser = argparse.ArgumentParser(description="Place the structural MRI and diffusion tensor MRI data into the form we want to use for our machine learning methods.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')

# Assume xyz is a Nx3 2D vector like the first return value of hcp.load_roi_info().
# That is, each row is a set of 3D cartesian coordinates.
# radius_inclination_azimuth is also a Nx3 2D vector
# but with each set of 3D cartesian coordinates converted to a spherical coordinates.
def cartesian_to_spherical(xyz:torch.Tensor):
    radius_inclination_azimuth = torch.zeros_like(xyz)
    square_sum_sqrt = torch.square(xyz).cumsum(dim=-1).sqrt()
    radius = torch.sqrt(square_sum_sqrt[:,2])# radius = sqrt(x^2 + y^2 + z^2)
    radius_inclination_azimuth[:,0] = radius
    radius_inclination_azimuth[:,1] = torch.arccos(xyz[:,2]/radius)# inclination = arccos(z/radius)
    radius_inclination_azimuth[:,2] = torch.sign(xyz[:,1]) * torch.arccos(xyz[:,0]/square_sum_sqrt[:,1])# azimuth = sgn(y)*arccos( x/sqrt(x^2+y^2) )
    return radius_inclination_azimuth

def load_structural_features_data(data_subset:str):
    subject_list = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_list)
    num_nodes = hcp.num_brain_areas
    # We originally stored the SC data as num_nodes x num_nodes symmetric matrices with 0s on the diagonal.
    # For our current purposes, it is more useful to just take the elements above the diagonal.
    triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
    num_pairs = triu_rows.numel()
    structural_features = torch.zeros( (num_subjects, num_nodes, hcp.features_per_area), dtype=float_type, device=device )
    structural_connectivity = torch.zeros( (num_subjects, num_pairs), dtype=float_type, device=device )
    print(f'loading structural data for {data_subset} subjects...')
    for subject in range(num_subjects):
        subject_id = subject_list[subject]
        structural_features_file = hcp.get_area_features_file_path(directory_path=data_directory, subject_id=subject_id)
        structural_features[subject,:,:] = hcp.load_matrix_from_binary(file_path=structural_features_file, dtype=float_type, device=device).transpose(dim0=0, dim1=1)
        structural_connectivity_file = hcp.get_structural_connectivity_file_path(directory_path=data_directory, subject_id=subject_id)
        structural_connectivity[subject,:] = hcp.load_matrix_from_binary(file_path=structural_connectivity_file, dtype=float_type, device=device)[triu_rows,triu_cols]
        print(f'subject {subject+1} of {num_subjects}, ID {subject_id}')
    print( 'structural features size', structural_features.size() )
    print( 'structural connectivity size', structural_connectivity.size() )
    print(f'time {time.time() - code_start_time:.3f}')
    return structural_features, structural_connectivity

def construct_node_features(coords:torch.Tensor, structural_features:torch.Tensor, structural_connectivity:torch.Tensor):
    num_subjects, num_nodes, _ = structural_features.size()
    # Broadcast the coordinates to all subjects.
    coords_for_all = coords.unsqueeze(dim=0).repeat( (num_subjects, 1, 1) )
    # For each region, sum over the structural connectivities of all edges in which it participates.
    # This easier to do if we transform the SC values back into a square matrix.
    triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
    structural_connectivity_square = torch.zeros( size=(num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    structural_connectivity_square[:,triu_rows,triu_cols] = structural_connectivity
    structural_connectivity_square[:,triu_cols,triu_rows] = structural_connectivity
    structural_connectivity_sum = structural_connectivity_square.sum(dim=-1, keepdim=True)
    # Put all these features together to get a 3 + 4 + 1 = 8-dimensional feature vector for each brain region of each individual.
    return torch.cat( (coords_for_all, structural_features, structural_connectivity_sum), dim=-1 )

def construct_edge_features(node_features:torch.Tensor, structural_connectivity:torch.Tensor):
    num_nodes = node_features.size(dim=1)
    triu_rows, triu_cols = isingmodel.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
    return torch.cat( (node_features[:,triu_rows,:], node_features[:,triu_cols,:], structural_connectivity[:,:,None]), dim=-1 )

def construct_all_features(structural_features:torch.Tensor, structural_connectivity:torch.Tensor):
    return torch.cat(  ( structural_features.flatten(start_dim=1, end_dim=2), structural_connectivity ), dim=-1  )

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    # Region names and centroid coordinates are features of the region description from the Atlas and do not vary by individual.
    # z-score the coordinates using the mean and standard deviation of each coordinate over all regions.
    names, coords = hcp.load_roi_info(directory_path=data_directory, dtype=float_type, device=device)
    coords_stds, coords_means = torch.std_mean(input=coords, dim=0, keepdim=True)
    coords = (coords - coords_means)/coords_stds
    # Convert to spherical coordinates after taking the z-scores so that the radius will be the distance from the centroid of the point cloud.
    coords = cartesian_to_spherical(coords)
    structural_feature_means = None
    structural_feature_stds = None
    structural_connectivity_mean = None
    structural_connectivity_std = None
    for data_subset in ['training', 'validation', 'testing']:
        # Load the structural features and structural connectivity for each subset of the data.
        structural_features, structural_connectivity = load_structural_features_data(data_subset)
        # Standardize all of them to z-scores based on the mean and standard deviation from the training data.
        # For the features, take the mean and standard deviation over all subjects and all brain regions to get a separate value for each feature.
        # For the structural connectivity, take the mean and standard deviation over all subjects and region pairs to get a single value.
        if data_subset == 'training':
            structural_feature_stds, structural_feature_means = torch.std_mean( structural_features, dim=(0,1), keepdim=True )
            structural_connectivity_std, structural_connectivity_mean = torch.std_mean(structural_connectivity)
        structural_features = (structural_features - structural_feature_means)/structural_feature_stds
        structural_connectivity = (structural_connectivity - structural_connectivity_mean)/structural_connectivity_std
        # We now construct 3 kinds of feature vectors for input to different kinds of machine learning models.
        # 1. node_features has dimensions subjects x brain regions x features.
        # Use this if we want to learn to transform from the features of one region to a node representation, such as the external field parameter of an Ising model.
        coords_for_all = coords.unsqueeze(dim=0).repeat(  ( structural_features.size(dim=0), 1, 1 )  )
        node_features = construct_node_features(coords=coords, structural_features=structural_features, structural_connectivity=structural_connectivity)
        print( 'node features size', node_features.size() )
        node_features_file = os.path.join(output_directory, f'node_features_{data_subset}.pt')
        torch.save(obj=node_features, f=node_features_file)
        print(f'saved {node_features_file}, time {time.time() - code_start_time:.3f}')
        # 2. edge_features has dimensions subjects x region pair x features.
        # We just concatenate the features of the two individual nodes in a pair and the SC of the pair.
        # Use this if we want to learn to transform from the features of a pair of regions to an edge representation, such as the coupling parameter of an Ising model.
        edge_features = construct_edge_features(node_features=node_features, structural_connectivity=structural_connectivity)
        print( 'edge features size', edge_features.size() )
        edge_features_file = os.path.join(output_directory, f'edge_features_{data_subset}.pt')
        torch.save(obj=edge_features, f=edge_features_file)
        print(f'saved {edge_features_file}, time {time.time() - code_start_time:.3f}')
        # 3. all_features has dimensions subject x features
        # In this version, we flatten out the region structural features and concatenate them with the flattened upper triangular part of the structural connectivity.
        # We do not include the coordinates, as they are invariant with respect to the subject.
        # Use this if we want to learn to transform from one representation of the whole individual brain to another, such as the full set of Ising model parameters.
        all_features = construct_all_features(structural_features=structural_features, structural_connectivity=structural_connectivity)
        print( 'all features size', all_features.size() )
        all_features_file = os.path.join(output_directory, f'all_features_{data_subset}.pt')
        torch.save(obj=all_features, f=all_features_file)
        print(f'saved {all_features_file}, time {time.time() - code_start_time:.3f}')
    # Save the means and standard deviations in case we need to convert back to raw feature values later.
    all_feature_means = torch.cat(  ( coords_means.flatten(), structural_feature_means.flatten(), structural_connectivity_mean.unsqueeze(dim=0) ), dim=0  )
    all_feature_stds = torch.cat(  ( coords_stds.flatten(), structural_feature_stds.flatten(), structural_connectivity_std.unsqueeze(dim=0) ), dim=0  )
    all_feature_std_means = torch.stack( (all_feature_stds, all_feature_means), dim=0 )
    all_feature_std_means_file = os.path.join(output_directory, f'all_feature_std_means_training.pt')
    torch.save(obj=all_feature_std_means, f=all_feature_std_means_file)
    print(f'saved {all_feature_std_means_file}, time {time.time() - code_start_time:.3f}')
print(f'done, time {time.time() - code_start_time:.3f}')