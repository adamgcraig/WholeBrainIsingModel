import os
import torch
import time
import argparse
import hcpdatautils as hcp
from hcpdatautils import StructuralDataScaler
import isingmodellight

parser = argparse.ArgumentParser(description="Place the structural MRI and diffusion tensor MRI data into the form we want to use for our machine learning methods.")
parser.add_argument("-i", "--data_directory", type=str, default='E:\\HCP_data', help="directory where we can find the time series files")
parser.add_argument("-o", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the time series files")
args = parser.parse_args()
print('getting arguments...')
data_directory = args.data_directory
print(f'data_directory={data_directory}')
output_directory = args.output_directory
print(f'output_directory={output_directory}')

def load_structural_features_data(data_directory:str):
    subject_list = hcp.load_subject_subset(directory_path=data_directory, subject_subset='all', require_sc=True)
    num_subjects = len(subject_list)
    num_nodes = hcp.num_brain_areas
    structural_features = torch.zeros( (num_subjects, hcp.features_per_area, num_nodes), dtype=float_type, device=device )
    structural_connectivity = torch.zeros( (num_subjects, num_nodes, num_nodes), dtype=float_type, device=device )
    print(f'loading structural data...')
    for subject_index in range(num_subjects):
        subject_id = subject_list[subject_index]
        structural_features_file = hcp.get_area_features_file_path(directory_path=data_directory, subject_id=subject_id)
        structural_features[subject_index,:,:] = hcp.load_matrix_from_binary(file_path=structural_features_file, dtype=float_type, device=device)
        structural_connectivity_file = hcp.get_structural_connectivity_file_path(directory_path=data_directory, subject_id=subject_id)
        structural_connectivity[subject_index,:,:] = hcp.load_matrix_from_binary(file_path=structural_connectivity_file, dtype=float_type, device=device)
        print(f'subject {subject_index+1} of {num_subjects}, ID {subject_id}')
    print( 'structural features size', structural_features.size() )
    print( 'structural connectivity size', structural_connectivity.size() )
    print(f'time {time.time() - code_start_time:.3f}')
    return structural_features, structural_connectivity

# Assume xyz is a Nx3 2D vector like the first return value of hcp.load_roi_info().
# That is, each row is a set of 3D cartesian coordinates.
# radius_inclination_azimuth is also a Nx3 2D vector
# but with each set of 3D cartesian coordinates converted to a spherical coordinates.
def cartesian_to_spherical(xyz:torch.Tensor):
    # Center the point cloud so that the mean is 0 before we convert to angular coordinates.
    # xyz = xyz - torch.mean(xyz,dim=0)
    radius_inclination_azimuth = torch.zeros_like(xyz)
    square_sum_sqrt = torch.square(xyz).cumsum(dim=-1).sqrt()
    radius = square_sum_sqrt[:,2]# radius = sqrt(x^2 + y^2 + z^2)
    radius_inclination_azimuth[:,0] = radius
    radius_inclination_azimuth[:,1] = torch.arccos(xyz[:,2]/radius)# inclination = arccos(z/radius)
    radius_inclination_azimuth[:,2] = torch.sign(xyz[:,1]) * torch.arccos(xyz[:,0]/square_sum_sqrt[:,1])# azimuth = sgn(y)*arccos( x/sqrt(x^2+y^2) )
    return radius_inclination_azimuth

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')
    # Region names and centroid coordinates are features of the region description from the Atlas and do not vary by individual.
    node_features, structural_connectivity = load_structural_features_data(data_directory=data_directory)
    node_features = node_features.transpose(dim0=-2, dim1=-1)
    names, xyz = hcp.load_roi_info(directory_path=data_directory, dtype=float_type, device=device)
    ria = cartesian_to_spherical(xyz=xyz)
    coords = torch.cat( (xyz, ria), dim=-1 )
    num_subjects, num_nodes, _ = node_features.size()
    mean_sc = torch.mean(structural_connectivity, dim=-1, keepdim=True)
    node_features = torch.cat(   (  node_features, mean_sc, coords.unsqueeze(dim=0).repeat( (num_subjects, 1, 1) )  ), dim=-1   )
    print( 'extended node_features size', node_features.size() )
    node_features_file = os.path.join(output_directory, 'node_features_all_as_is.pt')
    node_feature_diffs = torch.abs(node_features[:,:,None,:] - node_features[:,None,:,:])
    torch.save(obj=node_features, f=node_features_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {node_features_file}')
    distance = (xyz[:,None,:] - xyz[None,:,:]).square().sum(dim=-1, keepdim=True).sqrt()
    triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products(num_nodes=num_nodes, device=device)
    edge_features = torch.cat(   (  structural_connectivity.unsqueeze(dim=-1), node_feature_diffs, distance.unsqueeze(dim=0).repeat( (num_subjects, 1, 1, 1) )  ), dim=-1   )[:,triu_rows,triu_cols,:]
    print( 'exended edge_features size', edge_features.size() )
    edge_features_file = os.path.join(output_directory, 'edge_features_all_as_is.pt')
    torch.save(obj=edge_features, f=edge_features_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {edge_features_file}')
    print(f'time {time.time()-code_start_time:.3f}, done')