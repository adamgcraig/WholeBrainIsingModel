# have not gotten this working right and have thought of a simpler way to do what we want to do.
import os
import torch
import isingmodel
import time
import argparse
from sklearn.cluster import KMeans

code_start_time = time.time()

int_type = torch.int
float_type = torch.float
device = torch.device('cuda')

parser = argparse.ArgumentParser(description="Sort feature-param point pairs by feature value. Bin them into groups of an equal number of points. Then calculate the mean and std. dev. of the param")
parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which we read the training examples")
parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
parser.add_argument("-c", "--file_name_fragment", type=str, default='group_training_and_individual_all_signed_params_rectangular_coords_times_beta', help="part of the input example files between example_[index]_ and _[validation|training]_example.pt, will also be part of the output file names")
parser.add_argument("-d", "--num_bins", type=int, default=36, help="number of bins into which to group the (feature value, param value) pairs")
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

# We use the Hartigan-Wong method.
# We can perform the search for the optimal move in parallel using GPU Tensors.
# https://en.wikipedia.org/wiki/K-means_clustering#Hartigan–Wong_method
# points contains the individual data points, has size num_points x num_features.
# assigned_center_index is a 1D int-valued Tensor of length num_points.
# assigned_center_index[i] is the initial cluster assignment of point points[i,:].
# We modify assigned_center_index in-place and also return it.
# The algorithm is a greedy minimization algorithm where we move points between clusters to minimize the sum of the squared distances of points in a cluster from the center of the cluster. 
def k_means(points:torch.Tensor, assigned_center_index:torch.Tensor, patience:int=1000):
    # Initial setup:
    # Find the centers of the initial clusters.
    assignment_mat = ( assigned_center_index.unique(sorted=True)[:,None] == assigned_center_index[None,:] ).float()
    cluster_sizes = torch.count_nonzero(assignment_mat, dim=1)
    # assigment_mat (num_clusters x num_points) @ points (num_points x num_features) = centers (num_clusters x num_features)
    centers = torch.matmul(assignment_mat, points)/cluster_sizes[:,None]
    # Find the squared distance of each point from each center.
    sq_dists = torch.sum( torch.square(centers[:,None,:] - points[None,:,:]), dim=-1 )
    for iteration in range(patience):
        # Estimate how much the sum of squared distances of points from their centers will decrease for each possible move of a point from one center to another.
        # size_over_size_minus_1 = cluster_sizes/(cluster_sizes-1)
        # size_over_size_plus_1 = cluster_sizes/(cluster_sizes+1)
        # size_over_size_minus_1_of_assigned = torch.matmul(size_over_size_minus_1, assignment_mat)
        sq_dist_to_assigned_center = torch.sum(sq_dists * assignment_mat, dim=0, keepdim=True)
        # value_of_move = (size_over_size_minus_1_of_assigned * sq_dist_to_assigned_center) - size_over_size_plus_1[:,None] * sq_dists
        value_of_move = sq_dist_to_assigned_center - sq_dists
        # Find the move that decreases the sum by the greatest amount.
        value_of_best_for_point, best_destination_for_point = torch.max(value_of_move, dim=0)
        best_value_total, best_point_to_move = torch.max(value_of_best_for_point, dim=0)
        print(f'time {time.time()-code_start_time:.3f}, iteration {iteration}, current cost {sq_dist_to_assigned_center.sum()}, best move has value {best_value_total:.3g}')
        if best_value_total <= 0:
            break
        # Update the community assignment of the chosen point.
        origin_of_best_point = assigned_center_index[best_point_to_move]
        destination_of_best_point = best_destination_for_point[best_point_to_move]
        assigned_center_index[best_point_to_move] = destination_of_best_point
        assignment_mat[origin_of_best_point,best_point_to_move] = 0.0
        assignment_mat[destination_of_best_point,best_point_to_move] = 1.0
        # Update the centers of the clusters the point is leaving and entering.
        best_point_to_move_coords = points[best_point_to_move,:]
        origin_size_old = cluster_sizes[origin_of_best_point]
        destination_size_old = cluster_sizes[destination_of_best_point]
        centers[origin_of_best_point,:] = (origin_size_old*centers[origin_of_best_point,:] - best_point_to_move_coords)/(origin_size_old - 1)
        centers[destination_of_best_point,:] = (destination_size_old*centers[destination_of_best_point,:] - best_point_to_move_coords)/(destination_size_old + 1)
        # Update the distances from all points to the newly moved centers.
        sq_dists[origin_of_best_point,:] = torch.sum( torch.square(points - centers[origin_of_best_point,:]), dim=-1 ).sqrt()
        sq_dists[destination_of_best_point,:] = torch.sum( torch.square(points - centers[destination_of_best_point,:]), dim=-1 ).sqrt()
        # Update the sizes of the clusters.
        cluster_sizes[origin_of_best_point] -= 1
        cluster_sizes[destination_of_best_point] += 1
    return assigned_center_index

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
    print('features size')
    print( features.size() )
    num_features = features.size(dim=-1)
    features_flat = torch.flatten(features, start_dim=0, end_dim=1)
    k_means_result = KMeans(n_clusters=num_bins).fit(X=features_flat)
    # assigned_center_index = torch.randint(  low=0, high=num_bins, size=( features_flat.size(dim=0), ), dtype=int_type, device=device  )
    # assigned_center_index = k_means(points=features_flat, assigned_center_index=assigned_center_index)

    # param_file = os.path.join(input_directory, f'{param_name}_{file_name_fragment}.pt')
    # param = torch.load(param_file)
    # print('param size')
    # print( param.size() )

    # num_pairs = param.numel()
    # bin_size = num_pairs//num_bins
    # print(f'using {num_bins} bins of size {bin_size}')
    # for feature_index in range(num_features):
    #     feature_stds, feature_means, param_stds, param_means = bin_feature_param_std_mean(feature=features[:,:,feature_index], param=param, bin_size=bin_size)
    #     mean_std_correlation = isingmodel.get_pairwise_correlation(mat1=feature_means, mat2=param_stds)
    #     mean_mean_correlation = isingmodel.get_pairwise_correlation(mat1=feature_means, mat2=param_means)
    #     std_std_correlation = isingmodel.get_pairwise_correlation(mat1=feature_stds, mat2=param_stds)
    #     std_mean_correlation = isingmodel.get_pairwise_correlation(mat1=feature_stds, mat2=param_means)
    #     feature_name = feature_names[feature_index]
    #     print(f'time\t{time.time() - code_start_time:.3f}\tcorrelation for binned\t{param_name}\tvs\t{feature_name}\tmean-mean\t{mean_mean_correlation:.3g}\tmean-std\t{mean_std_correlation:.3g}\tstd-mean\t{std_mean_correlation:.3g}\tmean-std\t{std_std_correlation:.3g}')
    #     feature_name_for_file = feature_names_for_files[feature_index]
    #     std_mean_pairs_file = os.path.join(output_directory, f'binned_{param_name}_{log_str}{feature_name_for_file}_std_mean_{file_name_fragment}_bin_size_{bin_size}.pt')
    #     torch.save( (feature_stds, feature_means, param_stds, param_means), std_mean_pairs_file )
    #     # print(f'{time.time() - code_start_time:.3f}, saved {std_mean_pairs_file}')

print(f'{time.time() - code_start_time:.3f}, done')