import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Find linear regressions to predict individual differences in Ising model parameters from individual differences in structural features.")
    parser.add_argument("-a", "--data_directory", type=str, default='D:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--region_feature_file_name_part", type=str, default='node_features_all_as_is', help="part of the output file name before .pt")
    parser.add_argument("-d", "--region_pair_feature_file_name_part", type=str, default='edge_features_all_as_is', help="part of the output file name before .pt")
    parser.add_argument("-e", "--model_file_name_part", type=str, default='ising_model_light_all_mean_std_1_medium_init_uncentered_reps_5_lr_0.01_steps_1200_pupd_per_bopt_1000_num_opt_1_bopt_steps_68_popt_steps_40000', help="the part of the Ising model file name before .pt.")
    parser.add_argument("-f", "--mean_state_file_name_part", type=str, default='mean_state_all_mean_std_1', help="the data mean state file before .pt.")
    parser.add_argument("-g", "--mean_state_product_file_name_part", type=str, default='mean_state_product_all_mean_std_1', help="the data mean state product (uncentered covariance) file before .pt.")
    parser.add_argument("-i", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-j", "--training_subject_end", type=int, default=837, help="1 past last training subject index")
    parser.add_argument("-k", "--scale_by_beta", action='store_true', default=False, help="Set this flag to scale by beta.")
    parser.add_argument("-l", "--iteration_increment", type=int, default=100, help="amount by which to increment the number of iterations when trying low-rank PCA")
    parser.add_argument("-m", "--default_iterations", type=int, default=100, help="initial number of iterations to try")
    parser.add_argument("-n", "--min_improvement", type=float, default=0.001, help="If reconstruction RMSE drops by less than this between attempts, we stop increasing the number of iterations.")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    region_feature_file_name_part = args.region_feature_file_name_part
    print(f'region_feature_file_name_part={region_feature_file_name_part}')
    region_pair_feature_file_name_part = args.region_pair_feature_file_name_part
    print(f'region_pair_feature_file_name_part={region_pair_feature_file_name_part}')
    model_file_name_part = args.model_file_name_part
    print(f'model_file_name_part={model_file_name_part}')
    mean_state_file_name_part = args.mean_state_file_name_part
    print(f'mean_state_file_name_part={mean_state_file_name_part}')
    mean_state_product_file_name_part = args.mean_state_product_file_name_part
    print(f'mean_state_product_file_name_part={mean_state_product_file_name_part}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')
    scale_by_beta = args.scale_by_beta
    print(f'scale_by_beta={scale_by_beta}')
    if scale_by_beta:
        beta_str = '_beta'
    else:
        beta_str = ''
    iteration_increment = args.iteration_increment
    print(f'iteration_increment={iteration_increment}')
    default_iterations = args.default_iterations
    print(f'default_iterations={default_iterations}')
    min_improvement = args.min_improvement
    print(f'min_improvement={min_improvement}')

    def get_proportion_of_variance(variances:torch.Tensor, desired_proportion:float=0.95):
        variances_sorted, _ = torch.sort(input=variances, descending=True)
        proportion_of_variance = torch.cumsum(variances_sorted, dim=0)/torch.sum(variances_sorted)
        components_needed = torch.count_nonzero(proportion_of_variance < desired_proportion) + 1
        print(f'time {time.time()-code_start_time:.3f}, need {components_needed} components to explain {desired_proportion:.3g} of the variance')
        return proportion_of_variance, components_needed
    
    def do_pca(params:torch.Tensor, param_name:str):
        original_vars = params.var(dim=0)
        print(f'found variances of original {param_name}')
        proportion_of_variance_original, components_needed_original = get_proportion_of_variance(variances=original_vars)
        num_subjects, params_per_subject = params.size()
        num_pcs = min(num_subjects, params_per_subject)
        max_recon_error = isingmodellight.get_pairwise_rmse( mat1=params, mat2=torch.zeros_like(params), dim=-1 ).max()
        print(f'time {time.time()-code_start_time:.3f}, original max RMSE {max_recon_error:.3g}')
        old_recon_error = max_recon_error+1.0
        num_iterations = default_iterations - iteration_increment
        while (old_recon_error - max_recon_error) >= min_improvement:
            old_recon_error = max_recon_error
            num_iterations += iteration_increment
            print(f'running PCA for {num_iterations} iterations...')
            (U, S, V) = torch.pca_lowrank(A=params, q=num_pcs, center=True, niter=num_iterations)
            reconstruct_params = torch.matmul(  input=U, other=torch.matmul( input=torch.diag_embed(input=S), other=V.transpose(dim0=0, dim1=1) )  )
            max_recon_error = isingmodellight.get_pairwise_rmse(mat1=params, mat2=reconstruct_params, dim=-1).max()
            print(f'time {time.time()-code_start_time:.3f}, max reconstruction RMSE {max_recon_error:.3g}')
        print('found eigenvalues of principal components U size', U.size(), 'S size', S.size(), 'V size', V.size() )
        proportion_of_variance_pca, components_needed_pca = get_proportion_of_variance(variances=S)
        save_file_path = os.path.join( output_directory, f'pca_{param_name}_subj_{training_subject_start}_to_{training_subject_end}_iter_{num_iterations}_{model_file_name_part}.pt' )
        torch.save( obj=(U, S, V, proportion_of_variance_original, proportion_of_variance_pca, components_needed_original, components_needed_pca), f=save_file_path )
        print(f'time {time.time()-code_start_time:.3f}, saved {save_file_path}')
        return V
    
    cosine_similarity_func = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

    def get_cosine_similarity_mat(V1:torch.Tensor, V2:torch.Tensor, v1_name:str, v2_name:str):
        print(f'finding cosine similarities of principal directions for {v1_name} v. {v2_name}...')
        similarities = cosine_similarity_func( V1.unsqueeze(dim=1), V2.unsqueeze(dim=2) )
        print( f'time {time.time()-code_start_time:.3f}, max similarity {similarities.max():.3g}, size', similarities.size() )
        save_file_path = os.path.join( output_directory, f'pd_cosine_similarity_{v1_name}_{v2_name}_subj_{training_subject_start}_to_{training_subject_end}_iter_inc_{iteration_increment}_{model_file_name_part}.pt' )
        torch.save(obj=similarities, f=save_file_path)
        print(f'time {time.time()-code_start_time:.3f}, saved {save_file_path}')
        return similarities
    
    def get_cosine_similarity_mat_slow(V1:torch.Tensor, V2:torch.Tensor, v1_name:str, v2_name:str):
        print(f'finding cosine similarities of principal directions for {v1_name} v. {v2_name}...')
        num_components_v1 = V1.size(dim=1)
        num_components_v2 = V2.size(dim=1)
        similarities = torch.zeros( size=(num_components_v1, num_components_v2), dtype=V1.dtype, device=V1.device )
        for v1_index in range(num_components_v1):
            for v2_index in range(num_components_v2):
                similarities[v1_index, v2_index] = cosine_similarity_func(V1[:,v1_index], V2[:,v2_index])
        print( f'time {time.time()-code_start_time:.3f}, max similarity {similarities.max():.3g}, size', similarities.size() )
        save_file_path = os.path.join( output_directory, f'pd_cosine_similarity_{v1_name}_{v2_name}_{model_file_name_part}.pt' )
        torch.save(obj=similarities, f=save_file_path)
        print(f'time {time.time()-code_start_time:.3f}, saved {save_file_path}')
        return similarities
    
    def get_sc():
        region_pair_feature_file_name = os.path.join(data_directory, f'{region_pair_feature_file_name_part}.pt')
        sc = torch.load(f=region_pair_feature_file_name, weights_only=False)[training_subject_start:training_subject_end,:,0]
        print( f'time {time.time()-code_start_time:.3f}, loaded {region_pair_feature_file_name}, size', sc.size() )
        return sc.clone()# clone() so that we can de-allocate the larger Tensor of which this is a view.
    
    def get_J():
        model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name}' )
        num_regions = model.J.size(dim=-1)
        if scale_by_beta:
            J = model.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * model.J
        else:
            J = model.J
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products(num_nodes=num_regions, device=J.device)
        return torch.mean(input=J[:,training_subject_start:training_subject_end,triu_rows,triu_cols], dim=0)
    
    def get_fc():
        mean_state_file_name = os.path.join(data_directory, f'{mean_state_file_name_part}.pt')
        mean_state = torch.load(f=mean_state_file_name, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_file_name}, size', mean_state.size() )
        mean_state_product_file_name = os.path.join(data_directory, f'{mean_state_product_file_name_part}.pt')
        mean_state_product = torch.load(f=mean_state_product_file_name, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {mean_state_product_file_name}, size', mean_state_product.size() )
        num_regions = mean_state_product.size(dim=-1)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products(num_nodes=num_regions, device=mean_state_product.device)
        fc = isingmodellight.get_fc( state_mean=torch.mean(mean_state, dim=0), state_product_mean=torch.mean(mean_state_product, dim=0), epsilon=0.0 )[training_subject_start:training_subject_end,triu_rows,triu_cols]
        return fc.clone()# clone() so that we can deallocate the larger Tensor of which it is a view.
    
    def get_region_features(num_region_features:int=4):
        region_feature_file_name = os.path.join(data_directory, f'{region_feature_file_name_part}.pt')
        region_features = torch.load(f=region_feature_file_name, weights_only=False)[training_subject_start:training_subject_end,:,:num_region_features]
        print( f'time {time.time()-code_start_time:.3f}, loaded {region_feature_file_name}, size', region_features.size() )
        return region_features.clone()# clone() so that we can deallocate the larger Tensor of which it is a view.
    
    def get_h():
        model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name}' )
        # print( 'transposed and unsqueezed h to size', h_for_lstsq.size() )
        if scale_by_beta:
            h = model.beta.unsqueeze(dim=-1) * model.h
        else:
            h = model.h
        return torch.mean(input=h[:,training_subject_start:training_subject_end,:], dim=0)

    def do_fc_pca():
        return do_pca( params=get_fc(), param_name='fc' )
        
    def do_J_pca():
        return do_pca( params=get_J(), param_name=f'J{beta_str}' )
    
    def do_h_pca():
        return do_pca( params=get_h(), param_name=f'h{beta_str}' )
    
    def do_sc_pca():
        return do_pca( params=get_sc(), param_name='sc' )
    
    feature_names = ['thickness', 'myelination', 'curvature', 'sulcus_depth']
    
    def do_region_feature_pca():
        region_features = get_region_features()
        num_features = len(feature_names)
        return [do_pca(params=region_features[:,:,feature_index], param_name=feature_names[feature_index]) for feature_index in range(num_features) ]
    
    def compare_h_region_feature_pds():
        V_h = do_h_pca()
        V_regions_features = do_region_feature_pca()
        similarities = [get_cosine_similarity_mat(V1=V_h, V2=V_f, v1_name='h', v2_name=f_name) for V_f, f_name in zip(V_regions_features, feature_names)]
        return similarities
    
    def compare_sc_pds():
        V_sc = do_sc_pca()
        get_cosine_similarity_mat_slow( V1=do_J_pca(), V2=V_sc, v1_name='J', v2_name='sc' )
        get_cosine_similarity_mat_slow( V1=do_fc_pca(), V2=V_sc, v1_name='fc', v2_name='sc' )
        return 0
    
    compare_h_region_feature_pds()
    compare_sc_pds()

print('done')