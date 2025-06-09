import os
import torch
import time
import argparse
import isingmodelshort
from isingmodelshort import IsingModel

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Compare a batch of Ising models (h: replica x subject x ROI, J: replica x subject x ROI x ROI) to a batch of target means (subject x ROI) and uncentered covariances (subject x ROI x ROI).")
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\aal90_short_pytorch', help="directory where we can find the target mean state and mean state product files")
    parser.add_argument("-b", "--model_directory", type=str, default='E:\\aal90_short_models', help="directory where we can find the fitted model files")
    parser.add_argument("-c", "--result_directory", type=str, default='E:\\aal90_short_results', help="directory to which we write the results of the testing")
    parser.add_argument("-d", "--mean_state_file_name", type=str, default='data_mean_all_mean_std_1.pt', help="name of the target mean state file")
    parser.add_argument("-e", "--mean_state_product_file_name", type=str, default='data_mean_product_all_mean_std_1.pt', help="name of the target mean state product file")
    parser.add_argument("-f", "--model_file_name", type=str, default='ising_model_aal_short_threshold_1_beta_updates_9.pt', help="name of the Ising model file")
    parser.add_argument("-g", "--sim_length", type=int, default=120000, help="number of simulation steps in the test")
    parser.add_argument("-i", "--save_full_ts", action='store_true', default=False, help="Set this flag to save the entire simulation time series.")
    parser.add_argument("-j", "--beta_test", action='store_true', default=False, help="Set this flag to try a range of beta values across replicas.")
    parser.add_argument("-k", "--min_beta", type=float, default=10e-10, help="low end of beta interval")
    parser.add_argument("-l", "--max_beta", type=float, default=1.0, help="high end of beta interval")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    model_directory = args.model_directory
    print(f'model_directory={model_directory}')
    result_directory = args.result_directory
    print(f'result_directory={result_directory}')
    mean_state_file_name = args.mean_state_file_name
    print(f'mean_state_file_name={mean_state_file_name}')
    mean_state_product_file_name = args.mean_state_product_file_name
    print(f'mean_state_product_file_name={mean_state_product_file_name}')
    model_file_name = args.model_file_name
    print(f'model_file_name={model_file_name}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
    save_full_ts = args.save_full_ts
    print(f'save_full_ts={save_full_ts}')
    beta_test = args.beta_test
    print(f'beta_test={beta_test}')
    min_beta = args.min_beta
    print(f'min_beta={min_beta}')
    max_beta = args.max_beta
    print(f'max_beta={max_beta}')
    
    print('loading target data time series state and state product means...')

    mean_state_file_path = os.path.join(data_directory, mean_state_file_name)
    target_mean_state = torch.load(f=mean_state_file_path, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded target mean state from {mean_state_file_path}, size', target_mean_state.size() )
    
    mean_state_product_file_path = os.path.join(data_directory, mean_state_product_file_name)
    target_mean_state_product = torch.load(f=mean_state_product_file_path, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded target mean state product from {mean_state_product_file_path}, size', target_mean_state_product.size() )

    model_file_path = os.path.join(model_directory, model_file_name)
    model = torch.load(f=model_file_path, weights_only=False)
    print( f'time {time.time()-code_start_time:.3f}, loaded Ising model from {model_file_path}' )

    results_file_suffix = f'test_length_{sim_length}_{model_file_name}'

    if beta_test:
        model.beta = isingmodelshort.get_linspace_beta_like(input=model.beta, min_beta=min_beta, max_beta=max_beta)
        results_file_suffix = f'{results_file_suffix}_beta_min_{min_beta:.3g}_max_{max_beta:.3g}'

    print(f'time {time.time()-code_start_time:.3f}, running simulation...')
    if save_full_ts:
        sim_ts = model.simulate_and_record_time_series(num_steps=sim_length)
        sim_mean_state, sim_mean_state_product = isingmodelshort.get_time_series_mean_step_by_step(time_series=sim_ts)
        print(f'time {time.time()-code_start_time:.3f}, simulation complete')
        print( 'sim_ts size', sim_ts.size() )
        sim_ts_file = os.path.join(result_directory, f'sim_ts_{results_file_suffix}')
        torch.save(obj=sim_ts, f=sim_ts_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {sim_ts_file}')
    else:
        sim_mean_state, sim_mean_state_product = model.simulate_and_record_means(num_steps=sim_length)
        print(f'time {time.time()-code_start_time:.3f}, simulation complete')

    print( 'sim_mean_state size', sim_mean_state.size() )
    print(f'sim mean state min {sim_mean_state.min():.3g}, mean {sim_mean_state.mean():.3g}, max {sim_mean_state.max():.3g}')
    sim_mean_state_file = os.path.join(result_directory, f'sim_mean_state_{results_file_suffix}')
    torch.save(obj=sim_mean_state, f=sim_mean_state_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_mean_state_file}')

    print( 'sim_mean_state_product size', sim_mean_state_product.size() )    
    print(f'sim mean state product min {sim_mean_state_product.min():.3g}, mean {sim_mean_state_product.mean():.3g}, max {sim_mean_state_product.max():.3g}')
    sim_mean_state_product_file = os.path.join(result_directory, f'sim_mean_state_product_{results_file_suffix}')
    torch.save(obj=sim_mean_state_product, f=sim_mean_state_product_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_mean_state_product_file}')

    mean_state_rmse = isingmodelshort.get_pairwise_rmse(mat1=sim_mean_state, mat2=target_mean_state)
    print(f'mean state RMSE min {mean_state_rmse.min():.3g}, mean {mean_state_rmse.mean():.3g}, max {mean_state_rmse.max():.3g}')
    state_mean_rmse_file = os.path.join(result_directory, f'mean_state_rmse_{results_file_suffix}')
    torch.save(obj=mean_state_rmse, f=state_mean_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {state_mean_rmse_file}')

    mean_state_product_rmse = isingmodelshort.get_pairwise_rmse_ut(mat1=sim_mean_state_product, mat2=target_mean_state_product)
    print(f'mean state product RMSE min {mean_state_product_rmse.min():.3g}, mean {mean_state_product_rmse.mean():.3g}, max {mean_state_product_rmse.max():.3g}')
    mean_state_product_rmse_file = os.path.join(result_directory, f'mean_state_product_rmse_{results_file_suffix}')
    torch.save(obj=mean_state_product_rmse, f=mean_state_product_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {mean_state_product_rmse_file}')

    target_cov = isingmodelshort.get_cov(state_mean=target_mean_state, state_product_mean=target_mean_state_product)
    sim_cov = isingmodelshort.get_cov(state_mean=sim_mean_state, state_product_mean=sim_mean_state_product)
    print(f'sim covariance min {sim_cov.min():.3g}, mean {sim_cov.mean():.3g}, max {sim_cov.max():.3g}')
    sim_cov_file = os.path.join(result_directory, f'sim_cov_{results_file_suffix}')
    torch.save(obj=sim_cov, f=sim_cov_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_cov_file}')
    
    cov_rmse = isingmodelshort.get_pairwise_rmse_ut(mat1=sim_cov, mat2=target_cov)
    print(f'covariance RMSE min {cov_rmse.min():.3g}, mean {cov_rmse.mean():.3g}, max {cov_rmse.max():.3g}')
    cov_rmse_file = os.path.join(result_directory, f'cov_rmse_{results_file_suffix}')
    torch.save(obj=cov_rmse, f=cov_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {cov_rmse_file}')

    target_fc = isingmodelshort.get_fc(state_mean=target_mean_state, state_product_mean=target_mean_state_product)
    sim_fc = isingmodelshort.get_fc(state_mean=sim_mean_state, state_product_mean=sim_mean_state_product)
    print(f'sim FC min {sim_fc.min():.3g}, mean {sim_fc.mean():.3g}, max {sim_fc.max():.3g}')
    sim_fc_file = os.path.join(result_directory, f'sim_fc_{results_file_suffix}')
    torch.save(obj=sim_fc, f=sim_fc_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {sim_fc_file}')
    
    fc_rmse = isingmodelshort.get_pairwise_rmse_ut(mat1=sim_fc, mat2=target_fc)
    print(f'FC RMSE min {fc_rmse.min():.3g}, mean {fc_rmse.mean():.3g}, max {fc_rmse.max():.3g}')
    fc_rmse_file = os.path.join(result_directory, f'fc_rmse_{results_file_suffix}')
    torch.save(obj=fc_rmse, f=fc_rmse_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file}')

    fc_corr = isingmodelshort.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc)
    print(f'FC correlation min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}')
    fc_corr_file = os.path.join(result_directory, f'fc_corr_{results_file_suffix}')
    torch.save(obj=fc_corr, f=fc_corr_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {fc_corr_file}')
    
    print(f'time {time.time()-code_start_time:.3f}, done')