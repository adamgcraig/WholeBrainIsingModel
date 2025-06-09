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

    parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
    parser.add_argument("-a", "--data_directory", type=str, default='/data/agcraig', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='/data/agcraig', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-d", "--training_subject_end", type=int, default=670, help="1 past last training subject index")
    parser.add_argument("-e", "--num_thresholds", type=int, default=31, help="number of thresholds at which to binarize the data")
    parser.add_argument("-f", "--min_threshold", type=float, default=0.0, help="minimum threshold at which to binarize the data")
    parser.add_argument("-g", "--max_threshold", type=float, default=3.0, help="maximum threshold at which to binarize the data")
    parser.add_argument("-i", "--learning_rate", type=float, default=0.01, help="learning rate of pseudolikelihood maximization")
    parser.add_argument("-j", "--num_saves", type=int, default=1000, help="Save the fitted parameters this many times.")
    parser.add_argument("-k", "--steps_per_save", type=int, default=1000, help="Save the fitted parameters every this many steps.")
    parser.add_argument("-l", "--num_betas", type=int, default=100, help="number of betas to test in parallel")
    parser.add_argument("-m", "--min_beta", type=float, default=1.0e-10, help="minimum beta to try")
    parser.add_argument("-n", "--max_beta", type=float, default=1.0, help="maximum beta to try")
    parser.add_argument("-o", "--beta_optimization_steps", type=int, default=1000, help="Rounds of beta optimization. Set it to a large number. We terminate early if it converges.")
    parser.add_argument("-p", "--sim_length", type=int, default=120000, help="length of simulations to use in beta test")

    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')
    num_thresholds = args.num_thresholds
    print(f'num_thresholds={num_thresholds}')
    min_threshold = args.min_threshold
    print(f'min_threshold={min_threshold}')
    max_threshold = args.max_threshold
    print(f'max_threshold={max_threshold}')
    learning_rate = args.learning_rate
    print(f'learning_rate={learning_rate}')
    num_saves = args.num_saves
    print(f'num_saves={num_saves}')
    steps_per_save = args.steps_per_save
    print(f'steps_per_save={steps_per_save}')
    num_betas = args.num_betas
    print(f'num_betas={num_betas}')
    min_beta = args.min_beta
    print(f'min_beta={min_beta}')
    max_beta = args.max_beta
    print(f'max_beta={max_beta}')
    beta_optimization_steps = args.beta_optimization_steps
    print(f'beta_optimization_steps={beta_optimization_steps}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')

    def load_data_ts(input_directory:str, training_subject_start:int, training_subject_end:int, thresholds:torch.Tensor):
        # data_ts is stored as ts_per_subject x subjects x nodes x times.
        data_ts_file = os.path.join(input_directory, f'data_ts_all_as_is.pt')
        # Select training subjects only.
        data_ts = torch.load(f=data_ts_file)[:,training_subject_start:training_subject_end,:,:]
        # z-score.
        data_ts_std, data_ts_mean = torch.std_mean(input=data_ts, dim=-1, keepdim=True)
        data_ts -= data_ts_mean
        data_ts /= data_ts_std
        # Make a separate copy binarized at each possible threshold along the first dim, so now thresholds x ts_per_subject x subjects x nodes x times.
        data_ts = ( data_ts.unsqueeze(dim = 0) > thresholds.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) ).float()
        data_ts *= 2.0
        data_ts -= 1.0
        # Permute to thresholds x nodes x ts_per_subject x subjects x times.
        data_ts = torch.permute( input=data_ts, dims=(0,3,1,2,4) )
        # Flatten to thresholds x nodes x (ts_per_subject * subjects * times).
        data_ts = data_ts.flatten(start_dim=2, end_dim=-1)
        return data_ts
    
    def get_mean_state_product(data_ts:torch.Tensor):
        # Assume data_ts is thresholds x nodes x steps.
        # thresholds x nodes x steps @ thresholds x steps x nodes -> thresholds x nodes x nodes
        return torch.matmul( data_ts, data_ts.transpose(dim0=-2, dim1=-1) )/data_ts.size(dim=-1)
    
    def get_cov(mean_state:torch.Tensor, mean_state_product:torch.Tensor):
        # (1, num_nodes, num_nodes) - (1, num_nodes, 1) * (1, 1, num_nodes) = (1, num_nodes, num_nodes)
        return mean_state_product - mean_state.unsqueeze(dim=-1) * mean_state.unsqueeze(dim=-2)

    def maximize_pseudolikelihood(data_ts:torch.Tensor, learning_rate:float, num_saves:int, steps_per_save:int, output_directory:str, pl_params_string:str):
        print( 'data_ts size', data_ts.size() )
        # data_ts: num_thresholds x num_nodes x num_states
        num_states = data_ts.size(dim=-1)
        # data_ts_nodes_first: num_thresholds x num_nodes x num_states
        data_ts_nodes_first = data_ts
        # data_ts_states_first: num_thresholds x num_states x num_nodes
        data_ts_states_first = data_ts.transpose(dim0=-2, dim1=-1)
        # mean_state: num_thresholds x num_nodes x 1
        mean_state = data_ts_nodes_first.mean(dim=-1, keepdim=True)
        print( 'mean_state size', mean_state.size() )
        # mean_state_product: num_thresholds x num_nodes x num_nodes
        mean_state_product = get_mean_state_product(data_ts=data_ts)
        print( 'mean_state_product size', mean_state_product.size() )
        # h: num_thresholds x num_nodes x 1
        h = mean_state.clone()
        print( 'h size', h.size() )
        # J: num_thresholds x num_nodes x num_nodes
        J = mean_state_product.clone()
        print( 'J size', J.size() )
        # Zero the diagonal.
        J -= torch.diag_embed( input=torch.diagonal(input=J, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
        # mean_field: num_thresholds x num_nodes x num_states
        mean_field = torch.zeros_like(input=data_ts_nodes_first)
        print( 'mean_field size', mean_field.size() )
        # mean_mean_field_product: num_thresholds x num_nodes x num_nodes
        mean_mean_field_product = torch.zeros_like(J)
        print( 'mean_mean_field_product size', mean_mean_field_product.size() )
        for save_index in range(num_saves):
            for _ in range(steps_per_save):
                # num_thresholds x num_nodes x num_nodes @ num_thresholds x num_nodes x num_states -> num_thresholds x num_nodes x num_states
                torch.matmul(J, data_ts_nodes_first, out=mean_field)
                # num_thresholds x num_nodes x num_states + num_thresholds x num_nodes x 1 -> num_thresholds x num_nodes x num_states
                mean_field += h
                # element-wise
                mean_field.tanh_()
                # sum(num_thresholds x num_nodes x num_states, dim=-1, keepdim=True) -> num_thresholds x num_nodes x 1
                # num_thresholds x num_nodes x 1 - num_thresholds x num_nodes x 1 -> num_thresholds x num_nodes x 1
                h_diff = mean_state - torch.mean(mean_field, dim=-1, keepdim=True)
                # num_thresholds x num_nodes x 1 * scalar + num_thresholds x num_nodes x 1 -> num_thresholds x num_nodes x 1
                h += learning_rate*h_diff
                # num_thresholds x num_nodes x num_states @ num_thresholds x num_states x num_nodes -> num_thresholds x num_nodes x num_nodes
                torch.matmul( mean_field, data_ts_states_first, out=mean_mean_field_product )
                mean_mean_field_product /= num_states
                # num_thresholds x num_nodes x num_nodes - num_thresholds x num_nodes x num_nodes -> num_thresholds x num_nodes x num_nodes
                J_diff = mean_state_product - mean_mean_field_product
                # Zero the diagonal.
                J_diff -= torch.diag_embed( input=torch.diagonal(input=J_diff, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
                J += learning_rate*J_diff
            num_updates = (save_index+1)*steps_per_save
            print(f'time {time.time()-code_start_time:.3f}, update {num_updates}, h_diff min {h_diff.min():.3g}, mean {h_diff.mean():.3g}, max {h_diff.max():.3g}, J_diff min {J_diff.min():.3g}, J_diff mean {J_diff.mean():.3g}, J_diff max {J_diff.max():.3g}')
            h_file = os.path.join( output_directory, f'h_{pl_params_string}_updates_{num_updates}.pt')
            torch.save(obj=h, f=h_file)
            print(f'time {time.time()-code_start_time:.3f}, saved {h_file}')
            J_file = os.path.join( output_directory, f'J_{pl_params_string}_updates_{num_updates}.pt')
            torch.save(obj=J, f=J_file)
            print(f'time {time.time()-code_start_time:.3f}, saved {J_file}')
        return h.squeeze(dim=-1), J
    
    def run_and_save_tests(model:IsingModelLight, output_file_fragment:str, sim_length:int, target_cov:torch.Tensor, target_fc:torch.Tensor):
        print(f'time {time.time()-code_start_time:.3f}, starting simulation...')
        sim_mean_state, sim_mean_state_product, flip_rate = model.simulate_and_record_means_and_flip_rate_pmb(num_steps=sim_length)
        print(f'time {time.time()-code_start_time:.3f}, simulation complete')
        sim_state_mean_file = os.path.join(output_directory, f'sim_state_mean_{output_file_fragment}.pt')
        torch.save(obj=sim_mean_state, f=sim_state_mean_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {sim_state_mean_file}')
        sim_state_product_mean_file = os.path.join(output_directory, f'sim_state_product_mean_{output_file_fragment}.pt')
        torch.save(obj=sim_mean_state_product, f=sim_state_product_mean_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {sim_state_product_mean_file}')
        flip_rate_file = os.path.join(output_directory, f'flip_rate_{output_file_fragment}.pt')
        torch.save(obj=flip_rate, f=flip_rate_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {flip_rate_file}')
        sim_cov = isingmodellight.get_cov(state_mean=sim_mean_state, state_product_mean=sim_mean_state_product)
        cov_rmse = isingmodellight.get_pairwise_rmse_ut(mat1=sim_cov, mat2=target_cov)
        cov_rmse_file = os.path.join(output_directory, f'cov_rmse_{output_file_fragment}.pt')
        torch.save(obj=cov_rmse, f=cov_rmse_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {cov_rmse_file}')
        cov_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_cov, mat2=target_cov, epsilon=epsilon)
        cov_corr_file = os.path.join(output_directory, f'cov_corr_{output_file_fragment}.pt')
        torch.save(obj=cov_corr, f=cov_corr_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {cov_corr_file}')
        sim_fc = isingmodellight.get_fc_binary(state_mean=sim_mean_state, state_product_mean=sim_mean_state_product, epsilon=epsilon)
        fc_rmse = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc)
        fc_rmse_file = os.path.join(output_directory, f'fc_rmse_{output_file_fragment}.pt')
        torch.save(obj=fc_rmse, f=fc_rmse_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {fc_rmse_file}')
        fc_corr = isingmodellight.get_pairwise_correlation_ut(mat1=sim_fc, mat2=target_fc, epsilon=epsilon)
        fc_corr_file = os.path.join(output_directory, f'fc_corr_{output_file_fragment}.pt')
        torch.save(obj=fc_corr, f=fc_corr_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {fc_corr_file}')

    # thresholds = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device)
    epsilon = 0.0
    thresholds = torch.tensor(data=[0.0, 1.0, 2.4], dtype=float_type, device=device)
    num_thresholds = thresholds.numel()
    data_ts = load_data_ts(input_directory=data_directory, training_subject_start=training_subject_start, training_subject_end=training_subject_end, thresholds=thresholds)
    pl_params_string = f'pseudolikelihood_thresholds_{num_thresholds}_min_{min_threshold:.3g}_max_{max_threshold:.3g}_lr_{learning_rate:.3g}'
    h, J = maximize_pseudolikelihood(data_ts=data_ts, learning_rate=learning_rate, num_saves=num_saves, steps_per_save=steps_per_save, output_directory=output_directory, pl_params_string=pl_params_string)
    # Create a model with multiple instances of each fitted (h,J) pair for testing with different beta.
    h = h.unsqueeze(dim=0).repeat( (num_betas,1,1) )
    J = J.unsqueeze(dim=0).repeat( (num_betas,1,1,1) )
    beta = isingmodellight.get_linspace_beta(models_per_subject=num_betas, num_subjects=num_thresholds, dtype=h.dtype, device=h.device, min_beta=min_beta, max_beta=max_beta)
    s = isingmodellight.get_neg_state_like(input=h)
    model = IsingModelLight(beta=beta, J=J, h=h, s=s)
    # Compute our optimization target mean state, mean state product, and covariance.
    mean_state = data_ts.mean(dim=-1, keepdim=False)
    mean_state_product = get_mean_state_product(data_ts=data_ts)
    target_cov = get_cov(mean_state=mean_state, mean_state_product=mean_state_product)
    target_fc = isingmodellight.get_fc_binary(state_mean=mean_state, state_product_mean=mean_state_product, epsilon=epsilon)
    # Test the models with different beta values.
    output_file_fragment = f'{pl_params_string}_updates_{num_saves*steps_per_save}_betas_{num_betas}_min_{min_beta:.3g}_max_{max_beta:.3g}_sim_length_{sim_length}'
    run_and_save_tests(model=model, output_file_fragment=output_file_fragment, sim_length=sim_length, target_cov=target_cov, target_fc=target_fc)
    print(f'time {time.time()-code_start_time:.3f}, starting beta optimization...')
    model.optimize_beta_pmb(target_cov=target_cov, num_updates=beta_optimization_steps, num_steps=sim_length, min_beta=min_beta, max_beta=max_beta, verbose=True)
    output_file_fragment = f'{pl_params_string}_updates_{num_saves*steps_per_save}_betas_{num_betas}_min_{min_beta:.3g}_max_{max_beta:.3g}_sim_length_{sim_length}_beta_updates_{beta_optimization_steps}'
    run_and_save_tests(model=model, output_file_fragment=output_file_fragment, sim_length=sim_length, target_cov=target_cov, target_fc=target_fc)
    print(f'time {time.time()-code_start_time:.3f}, done')

print(f'time {time.time()-code_start_time:.3f}, done')