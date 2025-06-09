import os
import torch
import time
import argparse
import isingmodellight

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Load and threshold unbinarized time series data. Make and save a Tensor counts such that counts[x] is the number of times x nodes flip in a single step.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-c", "--region_feature_file_part", type=str, default='node_features_all_as_is', help='region feature file name except for the .pt file extension')
    parser.add_argument("-d", "--sc_file_part", type=str, default='edge_features_all_as_is', help='SC file name except for the .pt file extension')
    parser.add_argument("-e", "--ts_file_part", type=str, default='data_ts_all_as_is', help='data time series file name except for the .pt file extension')
    parser.add_argument("-f", "--num_thresholds", type=int, default=31, help="number of thresholds to try")
    parser.add_argument("-g", "--min_threshold", type=float, default=0.0, help="minimum threshold to try")
    parser.add_argument("-i", "--max_threshold", type=float, default=3.0, help="maximum threshold to try")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    region_feature_file_part = args.region_feature_file_part
    print(f'region_feature_file_part={region_feature_file_part}')
    sc_file_part = args.sc_file_part
    print(f'sc_file_part={sc_file_part}')
    ts_file_part = args.ts_file_part
    print(f'ts_file_part={ts_file_part}')
    num_thresholds = args.num_thresholds
    print(f'num_thresholds={num_thresholds}')
    min_threshold = args.min_threshold
    print(f'min_threshold={min_threshold}')
    max_threshold = args.max_threshold
    print(f'max_threshold={max_threshold}')

    def count_non_nan(m:torch.Tensor):
        return (   torch.count_nonzero(  torch.logical_not( torch.isnan(m) )  )/m.numel()   ).item()

    def get_mean_and_fc(data_ts:torch.Tensor):
        mean_state, mean_state_product = isingmodellight.get_time_series_mean(time_series=data_ts)
        triu_rows, triu_cols = isingmodellight.get_triu_indices_for_products( num_nodes=mean_state_product.size(dim=-1), device=mean_state_product.device )
        mean_state = mean_state.mean(dim=0)
        mean_state_product = mean_state_product.mean(dim=0)
        fc = isingmodellight.get_fc(state_mean=mean_state, state_product_mean=mean_state_product, epsilon=0.0)
        return mean_state, fc[:,triu_rows,triu_cols].clone()
    
    def compute_and_save_correlations(means:torch.Tensor, features:torch.Tensor, file_suffix:str):
        correlations = isingmodellight.get_pairwise_correlation( mat1=means, mat2=features, epsilon=0.0, dim=0 )
        print( 'correlations size', correlations.size(), f'fraction non-NaN {count_non_nan(correlations):.3g}' )
        print('min, median, max')
        print( correlations.min(dim=0).values.tolist(), correlations.median(dim=0).values.tolist(), correlations.max(dim=0).values.tolist() )
        correlations_file = os.path.join(output_directory, f'corr_{file_suffix}.pt')
        torch.save(obj=correlations, f=correlations_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {correlations_file}')
        return 0
    
    def get_least_squares_prediction(means:torch.Tensor, features:torch.Tensor):
        means = means.transpose(dim0=0, dim1=1)
        print( 'means size', means.size(), f'fraction non-NaN {count_non_nan(means):.3g}' )
        features = torch.cat(  ( features.transpose(dim0=0, dim1=1), torch.ones_like(means) ), dim=-1  )
        print( 'features size', features.size(), f'fraction non-NaN {count_non_nan(features):.3g}' )
        coeffs = torch.linalg.lstsq(features, means).solution
        print( 'coeffs size', coeffs.size(), f'fraction non-NaN {count_non_nan(coeffs):.3g}' )
        predictions = torch.matmul(features, coeffs)
        print( 'predictions size', predictions.size(), f'fraction non-NaN {count_non_nan(predictions):.3g}' )
        return predictions.transpose(dim0=0, dim1=1)
    
    def save_time_series_correlations(data_ts:torch.Tensor, region_features:torch.Tensor, sc:torch.Tensor, file_suffix:str):
        mean_state, fc = get_mean_and_fc(data_ts=data_ts)
        mean_state = mean_state.unsqueeze(dim=-1)
        print('getting mean state v. region feature correlations...')
        compute_and_save_correlations( means=mean_state, features=region_features, file_suffix=f'mean_state_region_feature_{file_suffix}' )
        print('getting mean state v. linear model prediction correlations...')
        compute_and_save_correlations( means=mean_state, features=get_least_squares_prediction(means=mean_state, features=region_features), file_suffix=f'lstsq_mean_state_region_feature_{file_suffix}' )
        print('getting FC v. SC correlations...')
        compute_and_save_correlations( means=fc, features=sc, file_suffix=f'fc_sc_{file_suffix}' )
        return 0

    region_features_file = os.path.join(input_directory, f'{region_feature_file_part}.pt')
    region_features = torch.load(region_features_file, weights_only=False)[:,:,:4].clone()
    print( f'time {time.time()-code_start_time:.3f}, loaded {region_features_file}, size', region_features.size() )

    sc_file = os.path.join(input_directory, f'{sc_file_part}.pt')
    sc = torch.load(sc_file, weights_only=False)[:,:,0].clone()
    print( f'time {time.time()-code_start_time:.3f}, loaded {sc_file}, size', sc.size() )

    data_ts_file = os.path.join(input_directory, f'{ts_file_part}.pt')
    data_ts = torch.load(data_ts_file, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded {data_ts_file}, size', data_ts.size() )

    thresholds = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device)

    print('getting unbinarized time series correlations...')
    save_time_series_correlations(data_ts=data_ts, region_features=region_features, sc=sc, file_suffix='as_is')
    for threshold_index in range(num_thresholds):
        threshold = thresholds[threshold_index]
        print(f'getting data binarized at threshold {threshold:.3g}...')
        save_time_series_correlations( data_ts=isingmodellight.binarize_data_ts_z(data_ts=data_ts, threshold=threshold), region_features=region_features, sc=sc, file_suffix=f'thresh_{threshold:.3g}' )

    print(f'time {time.time()-code_start_time:.3f}, done')