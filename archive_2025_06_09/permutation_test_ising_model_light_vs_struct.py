import os
import torch
import time
import argparse
from isingmodellight import IsingModelLight


with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Measure correlations between structural features and parameters of fitted Ising models.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="directory from which we read the training examples")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the trained models and pandas DataFrame of loss results")
    parser.add_argument("-c", "--feature_file_name_fragment", type=str, default='all_as_is', help="part of the structural feature file names between node_features_ or edge_features_ and .pt")
    parser.add_argument("-d", "--model_file_name_fragment", type=str, default='all_quantile_0.5_medium_init_uncentered_reps_10_steps_1200_beta_updates_31_lr_0.01_param_updates_3000', help="part of the model file name between ising_model_light_ and .pt")
    parser.add_argument("-e", "--training_index_start", type=int, default=0, help="first index of training subjects")
    parser.add_argument("-f", "--training_index_end", type=int, default=669, help="last index of training subjects + 1")
    parser.add_argument("-g", "--num_permutations", type=int, default=1000, help="number of permutations of permuted pairings to try")
    parser.add_argument("-i", "--abs_params", action='store_true', default=False, help="Set this flag in order to take the absolute values of parameters.")
    parser.add_argument("-k", "--multiply_beta", action='store_true', default=False, help="Set this flag in order to multiply beta into the h and J parameters before taking the correlations.")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    feature_file_name_fragment = args.feature_file_name_fragment
    print(f'feature_file_name_fragment={feature_file_name_fragment}')
    model_file_name_fragment = args.model_file_name_fragment
    print(f'model_file_name_fragment={model_file_name_fragment}')
    training_index_start = args.training_index_start
    print(f'training_index_start={training_index_start}')
    training_index_end = args.training_index_end
    print(f'training_index_end={training_index_end}')
    num_permutations = args.num_permutations
    print(f'num_permutations={num_permutations}')
    abs_params = args.abs_params
    print(f'abs_params={abs_params}')
    multiply_beta = args.multiply_beta
    print(f'multiply_beta={multiply_beta}')

    if abs_params:
        abs_str = 'abs_params'
    else:
        abs_str = 'signed_params'
    if multiply_beta:
        multiply_beta_str = 'times_beta'
    else:
        multiply_beta_str = 'no_beta'

    def get_correlation_names(abs_params:bool):
        if abs_params:
            hname ='|h|'
            jname = '|J|'
        else:
            hname ='h'
            jname = 'J'
        node_feature_names = ['thickness', 'myelination', 'curvature', 'sulcus depth'] + ['mean SC'] + ['x', 'y', 'z'] + ['radius', 'inclination', 'azimuth']
        print( 'node_feature_names size', len(node_feature_names) )
        edge_feature_names = ['SC'] + [f'|{pname} difference|' for pname in node_feature_names] + ['distance']
        print( 'edge_feature_names size', len(edge_feature_names) )
        correlation_names = [f'{fname} vs {hname}' for fname in node_feature_names] + [f'{fname} vs {jname}' for fname in edge_feature_names]
        print( 'num correlations', len(correlation_names) )
        return correlation_names
    
    def load_structural_features(input_directory:str, file_name_fragment:str, training_index_start:int, training_index_end:int):    
        node_features_file = os.path.join(input_directory, f'node_features_{file_name_fragment}.pt')
        node_features = torch.load(node_features_file)
        print(f'time {time.time()-code_start_time:.3f}, loaded {node_features_file}')
        print( 'node_features size', node_features.size() )
        edge_features_file = os.path.join(input_directory, f'edge_features_{file_name_fragment}.pt')
        edge_features = torch.load(edge_features_file)
        print(f'time {time.time()-code_start_time:.3f}, loaded {edge_features_file}')
        print( 'edge_features size', edge_features.size() )
        node_features = node_features[training_index_start:training_index_end,:,:].clone()
        edge_features = edge_features[training_index_start:training_index_end,:,:].clone()
        return node_features, edge_features

    def load_ising_model(input_directory:str, file_name_fragment:str, training_index_start:int, training_index_end:int, abs_params:bool, multiply_beta:bool):

        ising_model_file = os.path.join(input_directory, f'ising_model_light_{file_name_fragment}.pt')
        ising_model = torch.load(f=ising_model_file)
        print(f'{time.time()-code_start_time:.3f}, loaded ising model')
        beta = ising_model.beta
        print( 'loaded beta size', beta.size() )
        h = ising_model.h
        print( 'loaded h size', h.size() )
        J = ising_model.J
        print( 'loaded J size', J.size() )
        
        triu_rows, triu_cols = ising_model.get_triu_indices_for_products()
        J = J[:,:,triu_rows,triu_cols]

        beta = beta[:,training_index_start:training_index_end].unsqueeze(dim=-1)
        h = h[:,training_index_start:training_index_end,:]
        J = J[:,training_index_start:training_index_end,:]
        
        print('After taking upper triangular parts of square matrices, and selecting training subjects')
        print( 'h size', h.size() )
        print( 'J size', J.size() )

        if multiply_beta:
            h = beta*h
            J = beta*J
        
        if abs_params:
            h = torch.abs(h)
            J = torch.abs(J)

        # clone() so that we do not keep the larger underlying Tensors of which these are views.
        return h.clone(), J.clone()

    # model_param has size models_per_subject x num_subjects x num_nodes (or num_pairs)
    # features has size num_subjects x num_nodes (or num_pairs) x num_features
    # We want to replicate features across models_per_subject and replicate model_param across num_features
    # and take the correlation over models per subject, subjects, and nodes (or node pairs)
    # so that we end up with a correlation matrix that is 1D with num_features elements.
    def model_feature_correlation(model_param:torch.Tensor, feature:torch.Tensor, epsilon:float=0.0):
        std_1, mean_1 = torch.std_mean( model_param, dim=(0,1,2) )# Take std and mean over model instance, subject, and node/pair.
        std_2, mean_2 = torch.std_mean( feature, dim=(0,1) )# Take std and mean over subject and node/pair
        return ( torch.mean( model_param.unsqueeze(dim=-1) * feature.unsqueeze(dim=0), dim=(0,1,2) ) - mean_1 * mean_2 + epsilon )/(std_1 * std_2 + epsilon)

    def get_correlations(node_features:torch.Tensor, edge_features:torch.Tensor, h:torch.Tensor, J:torch.Tensor, epsilon:float=0.0):
        h_correlations = model_feature_correlation(model_param=h, feature=node_features, epsilon=epsilon)
        J_correlations = model_feature_correlation(model_param=J, feature=edge_features, epsilon=epsilon)
        return torch.cat( (h_correlations, J_correlations), dim=0 )

    def compare_correlation_to_permuted(node_features:torch.Tensor, edge_features:torch.Tensor, h:torch.Tensor, J:torch.Tensor, num_permutations:int, shuffle_subjects:bool, shuffle_nodes:bool, correlation_names:list, true_correlations:torch.Tensor, file_name:str):
        num_subjects = node_features.size(dim=0)
        num_nodes = node_features.size(dim=1)
        num_pairs = edge_features.size(dim=1)
        num_correlations = len(correlation_names)
        # if os.path.exists(file_name):
        #     perm_correlations = torch.load(f=file_name)
        #     print(f'loaded correlations from {file_name}')
        # else:
        perm_correlations = torch.zeros( (num_permutations, num_correlations), dtype=float_type, device=device )
        # Because of how indexing works in PyTorch, we have to permute one index at a time.
        for perm in range(num_permutations):
            # print(f'time {time.time() - code_start_time:.3f},\tpermutation {perm} of {num_permutations}...')
            node_features_perm = node_features
            edge_features_perm = edge_features
            if shuffle_subjects:
                subject_order = torch.randperm(n=num_subjects, dtype=int_type, device=device)
                node_features_perm = node_features_perm[subject_order,:,:]
                edge_features_perm = edge_features_perm[subject_order,:,:]
            if shuffle_nodes:
                node_order = torch.randperm(n=num_nodes, dtype=int_type, device=device)
                node_features_perm = node_features_perm[:,node_order,:]
                pair_order = torch.randperm(n=num_pairs, dtype=int_type, device=device)
                edge_features_perm = edge_features_perm[:,pair_order,:]
            perm_correlations[perm,:] = get_correlations(node_features=node_features_perm, edge_features=edge_features_perm, h=h, J=J)
        torch.save(obj=perm_correlations, f=file_name)
        print(f'saved correlations to {file_name}')
        p_bigger = torch.count_nonzero( perm_correlations.abs() >= true_correlations.abs().unsqueeze(dim=0), dim=0 )/num_permutations
        return p_bigger, perm_correlations

    def summarize_distribution(values:torch.Tensor):
        values = values.flatten()
        quantile_cutoffs = torch.tensor([0.005, 0.5, 0.995], dtype=float_type, device=device)
        quantiles = torch.quantile(values, quantile_cutoffs)
        min_val = torch.min(values)
        max_val = torch.max(values)
        return f'median\t{quantiles[1].item():.3g}\t0.5%ile\t{quantiles[0].item():.3g}\t99.5%ile\t{quantiles[2].item():.3g}\tmin\t{min_val.item():.3g}\tmax\t{max_val.item():.3g}'

    def try_all_permutation_cases(node_features:torch.Tensor, edge_features:torch.Tensor, h:torch.Tensor, J:torch.Tensor, num_permutations:int, correlation_names:list, output_directory:str, file_name_fragment:str):
        true_correlations_file = os.path.join(output_directory, f'true_correlations_{file_name_fragment}.pt')
        # if os.path.exists(true_correlations_file):
        #     true_correlations = torch.load(true_correlations_file)
        #     print(f'loaded true correlations from {true_correlations_file}')
        # else:
        print(f'time {time.time() - code_start_time:.3f},\tstarting true correlations...')
        true_correlations = get_correlations(node_features=node_features, edge_features=edge_features, h=h, J=J)
        torch.save(obj=true_correlations, f=true_correlations_file)
        print(f'saved true correlations to {true_correlations_file}')
        print(f'time {time.time() - code_start_time:.3f},\tstarting subject permutations...')
        perm_correlations_subjects_file = os.path.join(output_directory, f'perm_correlations_subjects_{file_name_fragment}.pt')
        p_bigger_subjects, perm_correlations_subjects = compare_correlation_to_permuted(node_features=node_features, edge_features=edge_features, h=h, J=J, num_permutations=num_permutations, shuffle_subjects=True, shuffle_nodes=False, correlation_names=correlation_names, true_correlations=true_correlations, file_name=perm_correlations_subjects_file)
        print(f'time {time.time() - code_start_time:.3f},\tstarting node and node-pair permutations...')
        perm_correlations_nodes_file = os.path.join(output_directory, f'perm_correlations_nodes_{file_name_fragment}.pt')
        p_bigger_nodes, perm_correlations_nodes = compare_correlation_to_permuted(node_features=node_features, edge_features=edge_features, h=h, J=J, num_permutations=num_permutations, shuffle_subjects=False, shuffle_nodes=True, correlation_names=correlation_names, true_correlations=true_correlations, file_name=perm_correlations_nodes_file)
        print(f'time {time.time() - code_start_time:.3f},\tstarting (subject, node) and (subject, node-pair) permutations...')
        perm_correlations_both_file = os.path.join(output_directory, f'perm_correlations_both_{file_name_fragment}.pt')
        p_bigger_both, perm_correlations_both = compare_correlation_to_permuted(node_features=node_features, edge_features=edge_features, h=h, J=J, num_permutations=num_permutations, shuffle_subjects=True, shuffle_nodes=True, correlation_names=correlation_names, true_correlations=true_correlations, file_name=perm_correlations_both_file)
        num_correlations = len(correlation_names)
        for correlation_index in range(num_correlations):
            subjects_distribution_str = summarize_distribution(perm_correlations_subjects[:,correlation_index])
            nodes_distribution_str = summarize_distribution(perm_correlations_nodes[:,correlation_index])
            both_distribution_str = summarize_distribution(perm_correlations_both[:,correlation_index])
            print(f'time\t{time.time() - code_start_time:.3f}\t{correlation_names[correlation_index]}\tcorrelation\t{true_correlations[correlation_index]:.3g}\tprobability abs(correlation) with randomized subjects >= actual\t{p_bigger_subjects[correlation_index]:.3g}\t{subjects_distribution_str}\tprobability abs(correlation) with randomized nodes >= actual\t{p_bigger_nodes[correlation_index]:.3g}\t{nodes_distribution_str}\tprobability abs(correlation) with randomized subjects and nodes >= actual\t{p_bigger_both[correlation_index]:.3g}\t{both_distribution_str}')
        return true_correlations, perm_correlations_subjects, perm_correlations_nodes, perm_correlations_both
    
    node_features, edge_features = load_structural_features(input_directory=input_directory, file_name_fragment=feature_file_name_fragment, training_index_start=training_index_start, training_index_end=training_index_end)
    print(  'NaNs in node_features', torch.count_nonzero( torch.isnan(node_features) )  )
    print(  'NaNs in edge_features', torch.count_nonzero( torch.isnan(edge_features) )  )
    h, J = load_ising_model(input_directory=input_directory, file_name_fragment=model_file_name_fragment, training_index_start=training_index_start, training_index_end=training_index_end, abs_params=abs_params, multiply_beta=multiply_beta)
    print(  'NaNs in h', torch.count_nonzero( torch.isnan(h) )  )
    print(  'NaNs in J', torch.count_nonzero( torch.isnan(J) )  )
    correlation_names = get_correlation_names(abs_params=abs_params)
    output_file_name_fragment = f'{feature_file_name_fragment}_{model_file_name_fragment}_{abs_str}_{multiply_beta_str}_permutations_{num_permutations}'
    try_all_permutation_cases(node_features=node_features, edge_features=edge_features, h=h, J=J, num_permutations=num_permutations, correlation_names=correlation_names, output_directory=output_directory, file_name_fragment=output_file_name_fragment)
    print(f'time {time.time()-code_start_time:.3f}, done')