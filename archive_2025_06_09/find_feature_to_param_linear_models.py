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
    parser.add_argument("-j", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    parser.add_argument("-k", "--permutations", type=int, default=100, help="number of permutations to use in each permutation test")
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
    permutations = args.permutations
    print(f'permutations={permutations}')

    output_file_name_part = f'perms_{permutations}_subj_{training_subject_start}_{training_subject_end}_{model_file_name_part}'

    def do_direct_correlation_analysis_and_perm_test(independent:torch.Tensor, dependent:torch.Tensor, num_perms:int, corr_name:str):
        obsrv_dim = 1
        num_observations = dependent.size(dim=obsrv_dim)
        corr = isingmodellight.get_pairwise_correlation(mat1=dependent, mat2=independent, epsilon=0.0, dim=obsrv_dim)
        corr_abs = corr.abs()
        corr_gt_count = torch.zeros_like(corr)
        for _ in range(num_perms):
            perm = torch.randperm(n=num_observations, dtype=int_type, device=dependent.device)
            corr_abs_perm = isingmodellight.get_pairwise_correlation(mat1=dependent[:,perm], mat2=independent, epsilon=0.0, dim=obsrv_dim).abs()
            corr_gt_count += (corr_abs_perm >= corr_abs).float()
            # corr_gt_count[corr_abs_perm >= corr_abs] += 1.0
        p_corr = corr_gt_count/num_perms
        print(f'time {time.time()-code_start_time:.3f}, {corr_name}, corr min {corr.min():.3g} mean {corr.mean():.3g} max {corr.max():.3g} p(corr) min {p_corr.min():.3g} mean {p_corr.mean():.3g} max {p_corr.max():.3g}')
        return corr, p_corr

    def do_lstsq_correlation_analysis(independent:torch.Tensor, dependent:torch.Tensor):
        coeffs = torch.linalg.lstsq(independent, dependent).solution
        dependent_prediction = torch.matmul(input=independent, other=coeffs)
        obsrv_dim = 1
        r_squared = (   1 - torch.mean( torch.square(dependent_prediction - dependent), dim=obsrv_dim )/torch.mean(  torch.square( torch.mean(dependent, dim=obsrv_dim, keepdim=True) - dependent ), dim=obsrv_dim  )   ).squeeze(dim=-1)
        pred_corr = isingmodellight.get_pairwise_correlation(mat1=dependent, mat2=dependent_prediction, epsilon=0.0, dim=obsrv_dim).squeeze(dim=-1)
        return coeffs, dependent_prediction, r_squared, pred_corr
    
    def do_lstsq_correlation_analysis_and_perm_test(independent:torch.Tensor, dependent:torch.Tensor, num_perms:int, corr_name:str):
        independent = torch.cat(  tensors=( independent, torch.ones_like(dependent) ), dim=-1  )
        coeffs, dependent_prediction, r_squared, pred_corr = do_lstsq_correlation_analysis(independent=independent, dependent=dependent)
        print(corr_name)
        # print( 'coeffs size', coeffs.size() )
        # print( 'dependent_prediction size', dependent_prediction.size() )
        # print( 'r_squared size', r_squared.size(), f'r^2 min {r_squared.min():.3g} mean {r_squared.mean():.3g} max {r_squared.max():.3g}' )
        # print( 'pred_corr size', pred_corr.size(), f'pred corr min {pred_corr.min():.3g} mean {pred_corr.mean():.3g} max {pred_corr.max():.3g}' )
        num_observations = dependent.size(dim=1)
        r_squared_gt_count = torch.zeros_like(r_squared)
        pred_corr_gt_count = torch.zeros_like(pred_corr)
        for _ in range(num_perms):
            perm = torch.randperm(n=num_observations, dtype=int_type, device=dependent.device)
            _, _, r_squared_perm, pred_corr_perm = do_lstsq_correlation_analysis(independent=independent, dependent=dependent[:,perm,:])
            r_squared_gt_count += (r_squared_perm >= r_squared).float()
            pred_corr_gt_count += (pred_corr_perm >= pred_corr).float()
            # r_squared_gt_count[r_squared_perm >= r_squared] += 1.0
            # pred_corr_gt_count[pred_corr_perm >= pred_corr] += 1.0
        p_r_squared = r_squared_gt_count/num_perms
        p_pred_corr = pred_corr_gt_count/num_perms
        print(f'time {time.time()-code_start_time:.3f}, {corr_name}, r^2 min {r_squared.min():.3g} mean {r_squared.mean():.3g} max {r_squared.max():.3g} p(r^2) min {p_r_squared.min():.3g} mean {p_r_squared.mean():.3g} max {p_r_squared.max():.3g} pred corr min {pred_corr.min():.3g} mean {pred_corr.mean():.3g} max {pred_corr.max():.3g} p(pred corr) min {p_pred_corr.min():.3g} mean {p_pred_corr.mean():.3g} max {p_pred_corr.max():.3g}')
        return coeffs, dependent_prediction, r_squared, pred_corr, p_r_squared, p_pred_corr
    
    def get_sc():
        region_pair_feature_file_name = os.path.join(data_directory, f'{region_pair_feature_file_name_part}.pt')
        sc = torch.load(f=region_pair_feature_file_name, weights_only=False)[training_subject_start:training_subject_end,:,:1]
        print( f'time {time.time()-code_start_time:.3f}, loaded {region_pair_feature_file_name}, size', sc.size() )
        sc_for_lstsq = torch.transpose(input=sc, dim0=0, dim1=1)
        # print( 'transposed SC to size', sc_for_lstsq.size() )
        return sc_for_lstsq.clone()# clone() so that we can de-allocate the larger Tensor of which this is a view.
    
    def get_J():
        model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name}' )
        num_regions = model.J.size(dim=-1)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products(num_nodes=num_regions, device=model.J.device)
        J = torch.mean(input=model.J[:,training_subject_start:training_subject_end,triu_rows,triu_cols], dim=0)
        # num_subjects_2, num_pairs = J.size()
        # print(f'after averaging over replicas and taking the part above the diagonal, J has size {num_subjects_2} x {num_pairs}')
        J_for_lstsq = torch.transpose(input=J, dim0=0, dim1=1).unsqueeze(dim=-1)
        # print( 'transposed and unsqueezed J to size', J_for_lstsq.size() )
        return J_for_lstsq
    
    def do_sc_J_analyses(sc_for_lstsq:torch.Tensor):
        corr_name = 'J-SC'
        J_for_lstsq = get_J()
        print(f'computing J v. SC regressions...')
        sc_J_coeffs, sc_J_dependent_prediction, sc_J_r_squared, sc_J_pred_corr, sc_J_p_r_squared, sc_J_p_pred_corr = do_lstsq_correlation_analysis_and_perm_test(independent=sc_for_lstsq, dependent=J_for_lstsq, num_perms=permutations, corr_name=corr_name)
        sc_J_corr, sc_J_p_corr = do_direct_correlation_analysis_and_perm_test(independent=sc_for_lstsq[:,:,0], dependent=J_for_lstsq[:,:,0], num_perms=permutations, corr_name=corr_name)
        sc_J_file_name = os.path.join(output_directory, f'sc_J_linear_model_{model_file_name_part}.pt')
        torch.save( obj=(sc_J_coeffs, sc_J_dependent_prediction, sc_J_r_squared, sc_J_pred_corr, sc_J_p_r_squared, sc_J_p_pred_corr, sc_J_corr, sc_J_p_corr), f=sc_J_file_name )
        print(f'time {time.time()-code_start_time:.3f}, saved {sc_J_file_name}')
        return 0
    
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
        # num_subjects_3, num_pairs_2 = fc.size()
        # print(f'after taking the mean over scans, computing FC, and extracting the part above the diagonal, FC has size {num_subjects_3} x {num_pairs_2}')
        fc_for_lstsq = torch.transpose(input=fc, dim0=0, dim1=1).unsqueeze(dim=-1)
        # print( 'transposed and unsqueezed FC to size', fc_for_lstsq.size() )
        return fc_for_lstsq.clone()# clone() so that we can deallocate the larger Tensor of which it is a view.
    
    def do_sc_fc_analyses(sc_for_lstsq:torch.Tensor):
        corr_name = 'FC-SC'
        fc_for_lstsq = get_fc()
        print(f'computing FC v. SC regressions...')
        sc_fc_coeffs, sc_fc_dependent_prediction, sc_fc_r_squared, sc_fc_pred_corr, sc_fc_p_r_squared, sc_fc_p_pred_corr = do_lstsq_correlation_analysis_and_perm_test(independent=sc_for_lstsq, dependent=fc_for_lstsq, num_perms=permutations, corr_name=corr_name)
        sc_fc_corr, sc_fc_p_corr = do_direct_correlation_analysis_and_perm_test(independent=sc_for_lstsq[:,:,0], dependent=fc_for_lstsq[:,:,0], num_perms=permutations, corr_name=corr_name)
        sc_fc_file_name = os.path.join(output_directory, f'sc_fc_linear_model_{model_file_name_part}.pt')
        torch.save( obj=(sc_fc_coeffs, sc_fc_dependent_prediction, sc_fc_r_squared, sc_fc_pred_corr, sc_fc_p_r_squared, sc_fc_p_pred_corr, sc_fc_corr, sc_fc_p_corr), f=sc_fc_file_name )
        print(f'time {time.time()-code_start_time:.3f}, saved {sc_fc_file_name}')
        return 0
    
    def do_sc_analyses():
        sc_for_lstsq = get_sc()
        do_sc_J_analyses(sc_for_lstsq=sc_for_lstsq)
        do_sc_fc_analyses(sc_for_lstsq=sc_for_lstsq)
        return 0
    
    def get_region_features(num_region_features:int):
        region_feature_file_name = os.path.join(data_directory, f'{region_feature_file_name_part}.pt')
        region_features = torch.load(f=region_feature_file_name, weights_only=False)[training_subject_start:training_subject_end,:,:num_region_features]
        print( f'time {time.time()-code_start_time:.3f}, loaded {region_feature_file_name}, size', region_features.size() )
        # Do these transposes so that
        # region or region pair is the batch (first) dimension,
        # subject is the observation (second) dimension,
        # and feature/singleton is the feature (third) dimension.
        region_features_for_lstsq = torch.transpose(input=region_features, dim0=0, dim1=1)
        # print( 'transposed region features to size', region_features_for_lstsq.size() )
        return region_features_for_lstsq.clone()# clone() so that we can deallocate the larger Tensor of which it is a view.
    
    def get_h():
        model_file_name = os.path.join(data_directory, f'{model_file_name_part}.pt')
        model = torch.load(f=model_file_name, weights_only=False)
        print( f'time {time.time()-code_start_time:.3f}, loaded {model_file_name}' )
        h = torch.mean(input=model.h[:,training_subject_start:training_subject_end,:], dim=0)
        # print(f'after averaging over replicas, h has size {num_subjects} x {num_regions}')
        h_for_lstsq = torch.transpose(input=h, dim0=0, dim1=1).unsqueeze(dim=-1)
        # print( 'transposed and unsqueezed h to size', h_for_lstsq.size() )
        return h_for_lstsq
    
    def do_all_region_feature_h_analyses(region_features_for_lstsq:torch.Tensor, h_for_lstsq:torch.Tensor):
        print(f'computing h v. all features regressions...')
        all_features_h_coeffs, all_features_h_dependent_prediction, all_features_h_r_squared, all_features_h_pred_corr, all_features_h_p_r_squared, all_features_h_p_pred_corr = do_lstsq_correlation_analysis_and_perm_test(independent=region_features_for_lstsq, dependent=h_for_lstsq, num_perms=permutations, corr_name='h-all features')
        all_features_h_file_name = os.path.join(output_directory, f'all_features_h_linear_model_{model_file_name_part}.pt')
        torch.save( obj=(all_features_h_coeffs, all_features_h_dependent_prediction, all_features_h_r_squared, all_features_h_pred_corr, all_features_h_p_r_squared, all_features_h_p_pred_corr), f=all_features_h_file_name )
        print(f'time {time.time()-code_start_time:.3f}, saved {all_features_h_file_name}')
        return 0
    
    def do_one_region_feature_h_analyses(region_features_for_lstsq:torch.Tensor, h_for_lstsq:torch.Tensor, region_feature_names:list):
        num_regions, num_subjects, num_region_features = region_features_for_lstsq.size()
        print(f'computing h v. single feature regressions...')
        single_feature_h_coeffs = torch.zeros( size=(num_regions, 2, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_predictions = torch.zeros( size=(num_regions, num_subjects, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_r_squared = torch.zeros( size=(num_regions, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_pred_corr = torch.zeros( size=(num_regions, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_p_r_squared = torch.zeros( size=(num_regions, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_p_pred_corr = torch.zeros( size=(num_regions, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_corr = torch.zeros( size=(num_regions, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_p_corr = torch.zeros( size=(num_regions, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        for feature_index in range(num_region_features):
            feature_name = region_feature_names[feature_index]
            corr_name = f'h-{feature_name}'
            selected_feature = region_features_for_lstsq[:,:,feature_index:(feature_index+1)]
            coeffs, dependent_prediction, r_squared, pred_corr, p_r_squared, p_pred_corr = do_lstsq_correlation_analysis_and_perm_test(independent=selected_feature, dependent=h_for_lstsq, num_perms=permutations, corr_name=corr_name)
            corr, p_corr = do_direct_correlation_analysis_and_perm_test(independent=selected_feature[:,:,0], dependent=h_for_lstsq[:,:,0], num_perms=permutations, corr_name=corr_name)
            single_feature_h_coeffs[:,:,feature_index] = torch.squeeze(coeffs, dim=-1)
            single_feature_h_predictions[:,:,feature_index] = torch.squeeze(dependent_prediction, dim=-1)
            single_feature_h_r_squared[:,feature_index] = r_squared
            single_feature_h_pred_corr[:,feature_index] = pred_corr
            single_feature_h_p_r_squared[:,feature_index] = p_r_squared
            single_feature_h_p_pred_corr[:,feature_index] = p_pred_corr
            single_feature_h_corr[:,feature_index] = corr
            single_feature_h_p_corr[:,feature_index] = p_corr
        single_feature_h_file_name = os.path.join(output_directory, f'one_feature_h_linear_model_{model_file_name_part}.pt')
        torch.save( obj=(single_feature_h_coeffs, single_feature_h_predictions, single_feature_h_r_squared, single_feature_h_pred_corr, single_feature_h_p_r_squared, single_feature_h_p_pred_corr, single_feature_h_corr, single_feature_h_p_corr), f=single_feature_h_file_name )
        print(f'time {time.time()-code_start_time:.3f}, saved {single_feature_h_file_name}')
        return 0
    
    def do_all_but_one_region_feature_h_analyses(region_features_for_lstsq:torch.Tensor, h_for_lstsq:torch.Tensor, region_feature_names:list):
        num_regions, num_subjects, num_region_features = region_features_for_lstsq.size()
        print(f'leave-one-out h v. single feature regressions...')
        single_feature_h_coeffs = torch.zeros( size=(num_regions, 4, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_predictions = torch.zeros( size=(num_regions, num_subjects, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_r_squared = torch.zeros( size=(num_regions, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_pred_corr = torch.zeros( size=(num_regions, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_p_r_squared = torch.zeros( size=(num_regions, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        single_feature_h_p_pred_corr = torch.zeros( size=(num_regions, num_region_features), dtype=h_for_lstsq.dtype, device=h_for_lstsq.device )
        all_feature_indices = set( range(num_region_features) )
        for feature_index in range(num_region_features):
            feature_name = region_feature_names[feature_index]
            selected_indices = list( all_feature_indices - set([feature_index]) )
            selected_feature = region_features_for_lstsq[:,:,selected_indices]
            coeffs, dependent_prediction, r_squared, pred_corr, p_r_squared, p_pred_corr = do_lstsq_correlation_analysis_and_perm_test(independent=selected_feature, dependent=h_for_lstsq, num_perms=permutations, corr_name=f'h-all-other-than-{feature_name}')
            single_feature_h_coeffs[:,:,feature_index] = torch.squeeze(coeffs, dim=-1)
            single_feature_h_predictions[:,:,feature_index] = torch.squeeze(dependent_prediction, dim=-1)
            single_feature_h_r_squared[:,feature_index] = r_squared
            single_feature_h_pred_corr[:,feature_index] = pred_corr
            single_feature_h_p_r_squared[:,feature_index] = p_r_squared
            single_feature_h_p_pred_corr[:,feature_index] = p_pred_corr
        single_feature_h_file_name = os.path.join(output_directory, f'leave_one_feature_out_h_linear_model_{model_file_name_part}.pt')
        torch.save( obj=(single_feature_h_coeffs, single_feature_h_predictions, single_feature_h_r_squared, single_feature_h_pred_corr, single_feature_h_p_r_squared, single_feature_h_p_pred_corr), f=single_feature_h_file_name )
        print(f'time {time.time()-code_start_time:.3f}, saved {single_feature_h_file_name}')
        return 0
    
    def do_region_feature_h_analyses():
        region_feature_names = ['thickness', 'myelination', 'curvature', 'sulcus depth']
        num_region_features = len(region_feature_names)
        region_features_for_lstsq = get_region_features(num_region_features=num_region_features)
        h_for_lstsq = get_h()
        do_all_region_feature_h_analyses(region_features_for_lstsq=region_features_for_lstsq, h_for_lstsq=h_for_lstsq)
        do_one_region_feature_h_analyses(region_features_for_lstsq=region_features_for_lstsq, h_for_lstsq=h_for_lstsq, region_feature_names=region_feature_names)
        do_all_but_one_region_feature_h_analyses(region_features_for_lstsq=region_features_for_lstsq, h_for_lstsq=h_for_lstsq, region_feature_names=region_feature_names)
        return 0
    
    do_region_feature_h_analyses()
    do_sc_analyses()
print('done')