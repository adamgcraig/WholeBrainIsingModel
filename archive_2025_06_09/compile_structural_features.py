import os
import torch
import time
import argparse
import hcpdatautils as hcp
from hcpdatautils import StructuralDataScaler
import isingmodel

parser = argparse.ArgumentParser(description="Place the structural MRI and diffusion tensor MRI data into the form we want to use for our machine learning methods.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
parser.add_argument("-s", "--spherical", action='store_true', default=False, help="Include this flag to convert ROI coordinates from cartesian to spherical after standardization.")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')
spherical = args.spherical
print(f'spherical={spherical}')

def load_structural_features_data(data_subset:str):
    subject_list = hcp.load_subject_subset(directory_path=data_directory, subject_subset=data_subset, require_sc=True)
    num_subjects = len(subject_list)
    num_nodes = hcp.num_brain_areas
    structural_features = torch.zeros( (num_subjects, hcp.features_per_area, num_nodes), dtype=float_type, device=device )
    structural_connectivity = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    print(f'loading structural data for {data_subset} subjects...')
    for subject in range(num_subjects):
        subject_id = subject_list[subject]
        structural_features_file = hcp.get_area_features_file_path(directory_path=data_directory, subject_id=subject_id)
        structural_features[subject,:,:] = hcp.load_matrix_from_binary(file_path=structural_features_file, dtype=float_type, device=device)
        structural_connectivity_file = hcp.get_structural_connectivity_file_path(directory_path=data_directory, subject_id=subject_id)
        structural_connectivity[subject,:,:] = hcp.load_matrix_from_binary(file_path=structural_connectivity_file, dtype=float_type, device=device)
        print(f'subject {subject+1} of {num_subjects}, ID {subject_id}')
    print( 'structural features size', structural_features.size() )
    print( 'structural connectivity size', structural_connectivity.size() )
    print(f'time {time.time() - code_start_time:.3f}')
    return structural_features, structural_connectivity

def prepend_coords(node_coords:torch.Tensor, node_features:torch.Tensor):
    num_instances = node_features.size(dim=0)
    return torch.cat(   (  node_coords.unsqueeze(dim=0).repeat( (num_instances,1,1) ), node_features  ), dim=-1   )

def init_scaler(node_coords:torch.Tensor, node_features:torch.Tensor, edge_features:torch.Tensor):
    # Do any transformations we need to do prior to using the training data to init the scaler.
    node_features = node_features.transpose(dim0=-2, dim1=-1)
    node_features = prepend_coords(node_coords=node_coords, node_features=node_features)
    edge_features = edge_features.unsqueeze(dim=-1)
    scaler = StructuralDataScaler(training_node_features=node_features, training_edge_features=edge_features)
    return scaler

def transform_data(scaler:StructuralDataScaler, node_coords:torch.Tensor, node_features:torch.Tensor, edge_features:torch.Tensor, make_spherical:bool=True):
    node_features = node_features.transpose(dim0=-2, dim1=-1)
    node_features = prepend_coords(node_coords=node_coords, node_features=node_features)
    node_features = scaler.node_features_to_z_scores(node_features)
    if make_spherical:
        node_features = scaler.cartesian_to_spherical(node_features)
    edge_features = edge_features.unsqueeze(dim=-1)
    edge_features = scaler.rectangular_to_triu(edge_features)
    edge_features = scaler.edge_features_to_z_scores(edge_features)
    edge_features = scaler.triu_to_rectangular(edge_features)
    return node_features, edge_features

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
    training_structural_features, training_structural_connectivity = load_structural_features_data(data_subset='training')
    scaler = init_scaler(node_coords=coords, node_features=training_structural_features, edge_features=training_structural_connectivity)
    training_node_features, training_edge_features = transform_data(scaler=scaler, node_coords=coords, node_features=training_structural_features, edge_features=training_structural_connectivity, make_spherical=spherical)
    # Compute the mean features for each region or region pair.
    mean_training_node_features = training_node_features.mean(dim=0, keepdim=True)
    mean_training_edge_features = training_edge_features.mean(dim=0, keepdim=True)

    validation_structural_features, validation_structural_connectivity = load_structural_features_data(data_subset='validation')
    validation_node_features, validation_edge_features = transform_data(scaler=scaler, node_coords=coords, node_features=validation_structural_features, edge_features=validation_structural_connectivity, make_spherical=spherical)

    testing_structural_features, testing_structural_connectivity = load_structural_features_data(data_subset='testing')
    testing_node_features, testing_edge_features = transform_data(scaler=scaler, node_coords=coords, node_features=testing_structural_features, edge_features=testing_structural_connectivity, make_spherical=spherical)

    all_node_features = torch.cat( (mean_training_node_features, training_node_features, validation_node_features, testing_node_features), dim=0 )
    all_edge_features = torch.cat( (mean_training_edge_features, training_edge_features, validation_edge_features, testing_edge_features), dim=0 )

    if spherical:
        node_features_file = os.path.join(output_directory, 'node_features_group_training_and_individual_all_spherical.pt')
    else:
        node_features_file = os.path.join(output_directory, 'node_features_group_training_and_individual_all_rectangular.pt')
    torch.save(obj=all_node_features, f=node_features_file)
    print(f'time {time.time() - code_start_time:.3f}, saved {node_features_file}')

    edge_features_file = os.path.join(output_directory, 'edge_features_group_training_and_individual_all.pt')
    torch.save(obj=all_edge_features, f=edge_features_file)
    print(f'time {time.time() - code_start_time:.3f}, saved {edge_features_file}')
print(f'done, time {time.time() - code_start_time:.3f}')