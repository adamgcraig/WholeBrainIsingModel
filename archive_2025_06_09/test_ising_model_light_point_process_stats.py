import os
import torch
import time
import argparse
import isingmodellight
from isingmodellight import IsingModelLight
import numpy

code_start_time = time.time()
float_type = torch.float
int_type = torch.int
device = torch.device('cuda')
# device = torch.device('cpu')

def get_min_possible_exponent(num_passes:int, num_tries_per_pass:int, x_min:float=1.0):
    # The exponent needs to be larger than 1 for the zeta function to return a finite value.
    # First, we figure out how much above 1 the exponent needs to be.
    highest_inf = 1.0
    lowest_non_inf = 2.0
    # print(f'searching for lowest exponent that gives a non-inf zeta function value for x_min {x_min:.3g}...')
    for _ in range(num_passes):
        # print(f'pass {pass_index+1} [1+{highest_inf-1:.3g}, 1+{lowest_non_inf-1:.3g}]')
        try_exponents = torch.linspace(start=highest_inf, end=lowest_non_inf, steps=num_tries_per_pass, dtype=float_type, device=device)
        zeta_is_inf_indices = torch.nonzero(  torch.isinf( torch.special.zeta(try_exponents, x_min) )  ).flatten()
        # If no values are inf, quit.
        if zeta_is_inf_indices.numel() == 0:
            break
        min_index = zeta_is_inf_indices[-1]
        highest_inf = try_exponents[min_index]
        # If all values are inf, quit.
        if min_index == (num_tries_per_pass-1):
            break
        lowest_non_inf = try_exponents[min_index+1]
        if highest_inf == lowest_non_inf:
            print(f'time {time.time()-code_start_time:.3f}, converged to lowest exponent for which zeta is finite 1+{lowest_non_inf-1.0:.3g}')
            break
    # print(f'final [1+{highest_inf-1:.3g}, 1+{lowest_non_inf-1:.3g}]')
    return lowest_non_inf

def get_max_possible_exponent(num_passes:int, num_tries_per_pass:int, exponent_min:float, x_min:float=1.0):
    # If the exponent is too large, zeta(exponent, x_min) is effectively equal to its limit at infinity.
    # If x_min is 1, the limit is 1.
    # Otherwise, the limit is 0.
    # Find this maximum value.
    # print(f'searching for highest exponent that gives a non-limit zeta function value for x_min {x_min:.3g}...')
    highest_not_at_limit = exponent_min
    # First, just increase the exponent exponentially until we get the limit.
    num_tries_per_pass_float = float(num_tries_per_pass)
    lowest_at_limit = exponent_min * num_tries_per_pass_float
    if x_min == 1.0:
        zeta_limit = 1.0
    else:
        zeta_limit = 0.0
    for _ in range(num_passes):
        lowest_at_limit *= num_tries_per_pass_float
        if torch.special.zeta(lowest_at_limit, x_min) == zeta_limit:
            break
    # Then walk it back.
    for _ in range(num_passes):
        # print(f'pass {pass_index+1} [{highest_non_0:.3g}, {lowest_0:.3g}]')
        try_exponents = torch.linspace(start=highest_not_at_limit, end=lowest_at_limit, steps=num_tries_per_pass, dtype=float_type, device=device)
        zeta_at_limit_indices = torch.nonzero( torch.special.zeta(try_exponents, x_min) == zeta_limit ).flatten()
        # If no zeta in this range reach the limit, stop.
        if zeta_at_limit_indices.numel() == 0:
            break
        lowest_zeta_at_limit_index = zeta_at_limit_indices[0]
        lowest_at_limit = try_exponents[lowest_zeta_at_limit_index]
        # If all zet in this range reach the limit, stop.
        if lowest_zeta_at_limit_index == 0:
            break
        highest_not_at_limit = try_exponents[lowest_zeta_at_limit_index-1]
        if highest_not_at_limit == lowest_at_limit:
            print(f'time {time.time()-code_start_time:.3f}, search for highest possible exponent converged to {highest_not_at_limit:.3g}')
            break
    return highest_not_at_limit

def get_empirical_cdf(counts:torch.Tensor, count_dim:int=-1):
    # Flip so that it adds from the count for greatest value to the count for least value.
    # That way, cdf[x] = P( rand_value >= values[x] ). 
    return torch.cumsum( counts/counts.sum(dim=count_dim), dim=count_dim )

def get_power_law_cdf(values:torch.Tensor, exponent:torch.Tensor):
    # See equation 2.7 of Clauset et al. 2009.
    return 1.0 - torch.special.zeta(exponent, values)/torch.special.zeta(exponent, 1.0)

def get_ks_distance(empirical_cdf:torch.Tensor, distribution_cdf:torch.Tensor, prob_dim:int=-1):
    return torch.max( torch.abs(empirical_cdf - distribution_cdf), dim=prob_dim ).values

def sample_ks_distances(distribution_cdf:torch.Tensor, num_distances:int, samples_per_distance:int):
    # Create a sampling of KS distances of data sampled from the distribution described by the CDF.
    # distribution_cdf size (num_values,)
    # uniform_sample size (num_distances, samples_per_distance)
    uniform_sample = torch.rand( size=(num_distances, samples_per_distance), dtype=distribution_cdf.dtype, device=distribution_cdf.device )
    # uniform_sample -> (num_distances, samples_per_distance, 1) < distribution_cdf -> (1, 1, num_values)
    # result (num_distances, samples_per_distance, num_values) -count_nonzero-> (num_distances, samples_per_distance)
    sample_bin_index = torch.count_nonzero( uniform_sample.unsqueeze(dim=-1) < distribution_cdf.unsqueeze(dim=0).unsqueeze(dim=0), dim=-1 )
    # all_indices (num_values,)
    all_indices = torch.arange( start=1, end=distribution_cdf.size(dim=0)+1, dtype=sample_bin_index.dtype, device=sample_bin_index.device )
    # all_indices -> (1, num_values, 1) == sample_bin_index -> (num_distances, 1, samples_per_distance)
    # result (num_distances, num_values, samples_per_distance) -count_nonzero-> (num_distances, num_values)
    index_counts = torch.count_nonzero( all_indices.unsqueeze(dim=0).unsqueeze(dim=-1) == sample_bin_index.unsqueeze(dim=-2), dim=-1 )
    # still (num_distances, num_values)
    empirical_cdf = get_empirical_cdf(counts=index_counts, count_dim=-1)
    # (num_distances, num_values) - distribution_cdf -> (1, num_values)
    # result (num_distances, num_values) -max-> num_distances
    return get_ks_distance(empirical_cdf=empirical_cdf, distribution_cdf=distribution_cdf, prob_dim=-1)

def sample_ks_distances_low_mem(distribution_cdf:torch.Tensor, num_distances:int, samples_per_distance:int, max_samples_at_once:int):
    # Create a sampling of KS distances of data sampled from the distribution described by the CDF.
    # distribution_cdf size (num_values,)
    distances = torch.zeros( size=(num_distances,), dtype=distribution_cdf.dtype, device=distribution_cdf.device )
    # all_indices (num_values,1)
    # Have an extra bin for values that fall beyond the finite number of bins we have.
    # We will discard these later.
    all_indices = torch.arange( start=1, end=distribution_cdf.size(dim=-1)+2, dtype=distribution_cdf.dtype, device=distribution_cdf.device )
    all_indices_col = all_indices.unsqueeze(dim=-1)
    distribution_cdf_row = distribution_cdf.unsqueeze(dim=0)
    for dist_index in range(num_distances):
        counts = torch.zeros_like(input=all_indices)
        samples_remaining = samples_per_distance
        while samples_remaining > 0:
            batch_size = min(max_samples_at_once, samples_remaining)
            samples = torch.rand( size=(batch_size,1), dtype=distribution_cdf.dtype, device=distribution_cdf.device )
            bin_indices = torch.count_nonzero(samples < distribution_cdf_row, dim=-1)
            counts += torch.count_nonzero( all_indices_col ==  bin_indices.unsqueeze(dim=0), dim=-1 )
            samples_remaining = samples_remaining - batch_size
        # Drop the extra count with samples that go past our maximum bin.
        counts = counts[:-1]
        empirical_cdf = get_empirical_cdf(counts=counts, count_dim=-1)
        distances[dist_index] = get_ks_distance(empirical_cdf=empirical_cdf, distribution_cdf=distribution_cdf, prob_dim=-1)
    print(f'sampled KS distances min {distances.min():.3g} mean {distances.nanmean():.3g} max {distances.max():.3g}')
    return distances

def sample_ks_distances_fine(exponent:torch.Tensor, num_distances:int, num_samples:int):
    ks_distance = torch.zeros( size=(num_distances,), dtype=exponent.dtype, device=exponent.device )
    for dist_index in range(num_distances):
        # print(f'num_samples {num_samples}')
        samples = torch.tensor(  data=numpy.random.pareto( exponent.item()-1, num_samples ), dtype=exponent.dtype, device=exponent.device  ).sort().values.ceil()
        # print( 'samples', samples[:9], samples[-9:] )
        sample_cdf = torch.linspace(start=0, end=1, steps=num_samples, dtype=exponent.dtype, device=exponent.device)
        # print( 'sample_cdf', sample_cdf[:9], sample_cdf[-9:] )
        power_law_cdf = 1 - torch.special.zeta(exponent, samples)/torch.special.zeta(exponent, 1)
        # print( 'power_law_cdf', power_law_cdf[:9], power_law_cdf[-9:] )
        ks_distance[dist_index] = torch.max( torch.abs(sample_cdf-power_law_cdf) )
        # print(f'KS distance {ks_distance[dist_index]:.3g}')
    return ks_distance

# , max_samples_at_once:int=1000
def fit_power_law(counts:torch.Tensor, count_name:str, num_passes:int, num_tries_per_pass:int, exponent_min:float=1.00001, exponent_max:float=2.0, num_p_value_distances:int=1000):
    print(count_name)
    # ideal_exponent = torch.tensor(data=[1.5], dtype=float_type, device=device)
    max_value = counts.numel()
    values = torch.arange(start=1, end=max_value+1, dtype=float_type, device=counts.device)
    is_gt_0 = counts > 0
    # print( f'{torch.count_nonzero(is_gt_0)} of {max_value} counts are nonzero.' )
    counts = counts[is_gt_0]
    values = values[is_gt_0]
    if counts.numel() < 3:
        print(f'time {time.time()-code_start_time:.3f}, {count_name}, {counts.numel()} of {max_value} counts are nonzero. skipping')
        return 0.0, 1.0
    count_total = counts.sum()
    log_values = values.log()
    count_log_sum = torch.sum(counts * log_values)
    for pass_index in range(num_passes):
        exponent_choices = torch.linspace(start=exponent_min, end=exponent_max, steps=num_tries_per_pass, dtype=float_type, device=counts.device)
        log_likelihoods = -count_total * torch.log( torch.special.zeta(exponent_choices, 1.0) ) - exponent_choices * count_log_sum
        is_okay = torch.logical_not(  torch.logical_and( torch.isinf(log_likelihoods), torch.isnan(log_likelihoods) )  )
        exponent_choices = exponent_choices[is_okay]
        log_likelihoods = log_likelihoods[is_okay]
        max_ll_index = torch.argmax(log_likelihoods)
        exponent_min_index = max(max_ll_index-1, 0)
        exponent_min = exponent_choices[exponent_min_index]
        exponent_max_index = min(max_ll_index+1, num_tries_per_pass-1)
        exponent_max = exponent_choices[exponent_max_index]
        best_exponent = exponent_choices[max_ll_index]
        best_log_likelihood = log_likelihoods[max_ll_index]
        if exponent_min == exponent_max:
            print(f'time {time.time()-code_start_time:.3f}, power law fitting pass {pass_index+1} power law converged')
            break
    empirical_cdf = get_empirical_cdf(counts=counts)
    power_law_cdf = get_power_law_cdf(values=values, exponent=best_exponent)
    # ideal_power_law_cdf = get_power_law_cdf(values=values, exponent=ideal_exponent)
    ks_distance = get_ks_distance( empirical_cdf=empirical_cdf, distribution_cdf=power_law_cdf )
    # ks_distance_sample = sample_ks_distances_low_mem(distribution_cdf=power_law_cdf, num_distances=num_p_value_distances, samples_per_distance=count_total.int().item(), max_samples_at_once=max_samples_at_once)
    ks_distance_sample = sample_ks_distances_fine( exponent=best_exponent, num_distances=num_p_value_distances, num_samples=count_total.int() )
    p_value = torch.count_nonzero(ks_distance_sample >= ks_distance)/num_p_value_distances
    # kappa = torch.mean( torch.abs(empirical_cdf - ideal_power_law_cdf) )
    print(f'time {time.time()-code_start_time:.3f}, {count_name}, {counts.numel()} of {max_value} counts are nonzero. exponent {best_exponent:.3g} with log likelihood {best_log_likelihood:.3g}, KS distance {ks_distance:.3g}, P(KS distance of sample from power law > KS distance) {p_value:.3g}')
    return best_exponent, ks_distance

def fit_power_law_lstsq(mean_sizes:torch.Tensor):
    max_value = mean_sizes.numel()
    values = torch.arange(start=1, end=max_value+1, dtype=float_type, device=mean_sizes.device)
    is_gt_0 = mean_sizes > 0
    mean_sizes = mean_sizes[is_gt_0]
    values = values[is_gt_0]
    log_values_with_1s = torch.stack(  ( values.log(), torch.ones_like(values) ), dim=-1  )
    log_mean_sizes = torch.log( mean_sizes ).unsqueeze(dim=-1)
    coeffs = torch.linalg.lstsq(log_values_with_1s, log_mean_sizes).solution
    r_squared = 1 - torch.sum(  torch.square( log_mean_sizes - torch.matmul(log_values_with_1s, coeffs) )  )/torch.sum(  torch.square( log_mean_sizes - torch.mean(log_mean_sizes) )  )
    coeffs_flat = torch.flatten(coeffs)
    exponent = coeffs_flat[0]
    scale_factor = coeffs_flat[1]
    print(f'time {time.time()-code_start_time:.3f}, fitted power law with least squares: log(<S>) = {exponent:.3g}log(T) + {scale_factor:.3g}, R^2 = {r_squared:.3g}')
    return exponent, scale_factor, r_squared

with torch.no_grad():

    parser = argparse.ArgumentParser(description="Simulate several Ising models while tracking the expected values of their observables. Compare to expected values from data. Do an Euler step update. Repeat.")
    parser.add_argument("-a", "--data_directory", type=str, default='E:\\Ising_model_results_daai', help="directory where we can find the target mean state product files")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the fitted model files")
    parser.add_argument("-c", "--data_file_name_part", type=str, default='all_quantile_0.5', help="part of the output file name between mean_state_ or mean_state_product_ and .pt")
    parser.add_argument("-d", "--model_file_fragment", type=str, default='all_quantile_0.5_medium_init_uncentered_reps_10_steps_1200_beta_updates_31_lr_0.01_param_updates_3000', help="additional identifier to append to output file names to distinguish different runs with the same arguments")
    parser.add_argument("-f", "--sim_length", type=int, default=120000, help="number of simulation steps between updates")
    parser.add_argument("-g", "--combine_scans", action='store_true', default=False, help="Set this flag in order to take the mean over scans of the target mean state and state product. Otherwise, we just flatten the scan (0) and subject (1) dimensions together.")
    parser.add_argument("-i", "--reset_params", action='store_true', default=False, help="Set this flag in order to do a simulation with h and J reset to the mean states and mean state products, respectively.")
    parser.add_argument("-j", "--zero_h", action='store_true', default=False, help="Set this flag to zero out the h values.")
    parser.add_argument("-k", "--num_passes", type=int, default=1000, help="number of passes to use when searching for the optimal exponent")
    parser.add_argument("-l", "--num_tries_per_pass", type=int, default=1000, help="number of values to use in an individual search pass")
    parser.add_argument("-m", "--num_p_value_distances", type=int, default=1000, help="number of sample KS-distances to use when computing a p-value for the KS distance")
    parser.add_argument("-n", "--num_beta_multiplier_passes", type=int, default=10, help="number of passes to use when searching for the best beta multiplier")
    parser.add_argument("-o", "--num_beta_multipliers_per_pass", type=int, default=10, help="number of beta multipliers to try in one pass, done in sequence")
    parser.add_argument("-p", "--min_beta_multiplier", type=float, default=1.0, help="min number by which to multiply beta, may be necessary to slow down flips enough to see individual avalanches")
    parser.add_argument("-q", "--max_beta_multiplier", type=float, default=10.0, help="max number by which to multiply beta, may be necessary to slow down flips enough to see individual avalanches")
    # parser.add_argument("-r", "--max_samples_at_once", type=int, default=1000, help="max number of samples to take in parallel when computing the p-value, set to a smaller number to avoid running out of memory")
    args = parser.parse_args()
    print('getting arguments...')
    data_directory = args.data_directory
    print(f'data_directory={data_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_file_name_part = args.data_file_name_part
    print(f'data_file_name_part={data_file_name_part}')
    model_file_fragment = args.model_file_fragment
    print(f'model_file_fragment={model_file_fragment}')
    sim_length = args.sim_length
    print(f'sim_length={sim_length}')
    combine_scans = args.combine_scans
    print(f'combine_scans={combine_scans}')
    reset_params = args.reset_params
    print(f'reset_params={reset_params}')
    zero_h = args.zero_h
    print(f'zero_h={zero_h}')
    num_passes = args.num_passes
    print(f'num_passes={num_passes}')
    num_tries_per_pass = args.num_tries_per_pass
    print(f'num_tries_per_pass={num_tries_per_pass}')
    num_p_value_distances = args.num_p_value_distances
    print(f'num_p_value_distances={num_p_value_distances}')
    num_beta_multiplier_passes = args.num_beta_multiplier_passes
    print(f'num_beta_multiplier_passes={num_beta_multiplier_passes}')
    num_beta_multipliers_per_pass = args.num_beta_multipliers_per_pass
    print(f'num_beta_multipliers_per_pass={num_beta_multipliers_per_pass}')
    min_beta_multiplier = args.min_beta_multiplier
    print(f'min_beta_multiplier={min_beta_multiplier}')
    max_beta_multiplier = args.max_beta_multiplier
    print(f'max_beta_multiplier={max_beta_multiplier}')
    # max_samples_at_once = args.max_samples_at_once
    # print(f'max_samples_at_once={max_samples_at_once}')

    model_file = os.path.join(data_directory, f'ising_model_light_{model_file_fragment}.pt')
    model = torch.load(f=model_file)
    print( f'time {time.time()-code_start_time:.3f}, loaded {model_file}' )
    if reset_params:
        print('loading data time series state and state product means')
        target_state_mean_file = os.path.join(data_directory, f'mean_state_{data_file_name_part}.pt')
        target_state_mean = torch.load(target_state_mean_file)
        # On load, the dimensions of target_state_mean should be scan x subject x node or scan x subject x node-pair.
        print( f'time {time.time()-code_start_time:.3f}, loaded target_state_mean with size', target_state_mean.size() )
        if combine_scans:
            target_state_mean = torch.mean(target_state_mean, dim=0, keepdim=False)
        else:
            target_state_mean = torch.flatten(target_state_mean, start_dim=0, end_dim=1)
        target_state_mean = target_state_mean.unsqueeze(dim=0)
        print( f'time {time.time()-code_start_time:.3f}, flattened scan and subject dimensions', target_state_mean.size() )
        target_state_product_mean_file = os.path.join(data_directory, f'mean_state_product_{data_file_name_part}.pt')
        target_state_product_mean = torch.load(target_state_product_mean_file)
        # On load, the dimensions of target_state_product_mean should be scan x subject x node x node or scan x subject x node-pair.
        print( f'time {time.time()-code_start_time:.3f}, loaded target_state_product_mean with size', target_state_product_mean.size() )
        if len( target_state_product_mean.size() ) < 4:
            target_state_product_mean = isingmodellight.triu_to_square_pairs( triu_pairs=target_state_product_mean, diag_fill=0 )
        if combine_scans:
            target_state_product_mean = torch.mean(target_state_product_mean, dim=0, keepdim=False)
        else:
            target_state_product_mean = torch.flatten(target_state_product_mean, start_dim=0, end_dim=1)
        target_state_product_mean = target_state_product_mean.unsqueeze(dim=0)
        print( f'time {time.time()-code_start_time:.3f}, converted to square format and flattened scan and subject dimensions', target_state_product_mean.size() )
        model.h[:,:,:] = target_state_mean
        model.J[:,:,:,:] = target_state_product_mean
        reset_str = '_reset'
    else:
        reset_str = ''
    if zero_h:
        model.h.zero_()
        zero_h_str = '_no_h'
    else:
        zero_h_str = ''
    original_beta = model.beta
    # We only count a gap or an avalanche if it has both a start step and an end step, so we need to set aside one time point to be before the start and another to be after the end.
    # max_duration = sim_length - 1
    # Since the size is the number of activations, meaning transitions from -1 to +1, any given node can contribute to an avalanche in at most half of the time steps in a time series. 
    num_nodes = model.s.size(dim=-1)
    # max_size = max_duration*num_nodes//2
    exponent_min = get_min_possible_exponent(num_passes=num_passes, num_tries_per_pass=num_tries_per_pass)
    exponent_max = get_max_possible_exponent(num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, exponent_min=exponent_min)
    for beta_pass_index in range(num_beta_multiplier_passes):
        print(f'time {time.time()-code_start_time:.3f}, searching for best beta multiplier in range [{min_beta_multiplier:.3g}, {max_beta_multiplier:.3g}]')
        beta_multipliers = torch.linspace(start=min_beta_multiplier, end=max_beta_multiplier, steps=num_beta_multipliers_per_pass, dtype=float_type, device=original_beta.device)
        avalanche_duration_ks_distances = torch.ones_like(beta_multipliers)
        avalanche_size_ks_distances = torch.ones_like(beta_multipliers)
        diff_from_critical = torch.ones_like(beta_multipliers)
        for beta_index in range(num_beta_multipliers_per_pass):
            beta_multiplier = beta_multipliers[beta_index]
            print(f'time {time.time()-code_start_time:.3f}, pass {beta_pass_index+1} of {num_beta_multiplier_passes}, beta multiplier {beta_index+1} of {num_beta_multipliers_per_pass}, value {beta_multiplier:.3g}')
            model.beta = beta_multiplier * original_beta
            branching_parameter, gap_duration_count, avalanche_duration_count, avalanche_size_count, avalanche_mean_size_for_duration = model.simulate_and_record_point_process_stats_pmb(num_steps=sim_length)
            # print( 'branching_parameter size', branching_parameter.size() )
            print(f'time {time.time()-code_start_time:.3f}, branching parameter min {branching_parameter.min():.3g}, mean {branching_parameter.mean():.3g}, max {branching_parameter.max():.3g}')
            # print( 'gap_duration_count size', gap_duration_count.size() )
            # print( 'gap_duration_count sum', gap_duration_count.sum() )
            # print( 'avalanche_duration_count size', avalanche_duration_count.size() )
            # print( 'avalanche_duration_count sum', avalanche_duration_count.sum() )
            # print( 'avalanche_size_count size', avalanche_size_count.size() )
            # print( 'avalanche_size_count sum', avalanche_size_count.sum() )
            # print( 'avalanche_mean_size_for_duration size', avalanche_mean_size_for_duration.size() )
            # print( 'avalanche_mean_size_for_duration max', avalanche_mean_size_for_duration.max() )
            # power_law_exponents = [ fit_power_law(counts=counts, count_name=count_name, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, exponent_min=exponent_min, exponent_max=exponent_max, num_p_value_distances=num_p_value_distances) for counts, count_name in zip([gap_duration_count, avalanche_duration_count, avalanche_size_count], ['gap_duration_count', 'avalanche_duration_count', 'avalanche_size_count']) ]
            # , max_samples_at_once=max_samples_at_once
            gap_duration_exponent, _ = fit_power_law(counts=gap_duration_count, count_name='gap_duration_count', num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, exponent_min=exponent_min, exponent_max=exponent_max, num_p_value_distances=100)
            avalanche_duration_exponent, avalanche_duration_ks_distances[beta_index] = fit_power_law(counts=avalanche_duration_count, count_name='avalanche_duration_count', num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, exponent_min=exponent_min, exponent_max=exponent_max, num_p_value_distances=100)
            avalanche_size_exponent, avalanche_size_ks_distances[beta_index] = fit_power_law(counts=avalanche_size_count, count_name='avalanche_size_count', num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, exponent_min=exponent_min, exponent_max=exponent_max, num_p_value_distances=100)
            mean_size_for_duration_exponent, scale_factor, r_squared = fit_power_law_lstsq(mean_sizes=avalanche_mean_size_for_duration)
            diff_from_critical[beta_index] = (avalanche_duration_exponent-1)/(avalanche_size_exponent-1) - mean_size_for_duration_exponent
            print(f'(duration_exponent-1)/(size_exponent-1) - mean_size_for_duration_exponent = ({avalanche_duration_exponent:.3g}-1)/({avalanche_size_exponent:.3g}-1) - {mean_size_for_duration_exponent:.3g} = {diff_from_critical[beta_index]:.3g}')
        if beta_pass_index == 0:
            beta_ks_dist_file = os.path.join(output_directory, f'first_pass_beta_and_ks_dist_{model_file_fragment}_betax_num_{num_beta_multipliers_per_pass}_min_{min_beta_multiplier:.3g}_max_{max_beta_multiplier:.3g}_steps_{sim_length}.pt')
            torch.save( obj=(beta_multipliers, avalanche_duration_ks_distances, avalanche_size_ks_distances), f=beta_ks_dist_file )
            print(f'time {time.time()-code_start_time:.3f}, saved {beta_ks_dist_file}')
        # log_likelihood_sum = avalanche_duration_log_likelihoods + avalanche_size_log_likelihoods# log( P(a)*P(b) ) = log( P(a) ) + log( P(b) )
        best_beta_index = torch.argmin( avalanche_duration_ks_distances + avalanche_size_ks_distances )
        best_beta_multiplier = beta_multipliers[best_beta_index]
        print(f'time {time.time()-code_start_time:.3f}, best beta at index {best_beta_index}, beta multiplier {best_beta_multiplier:.3g}, duration KS distance {avalanche_duration_ks_distances[best_beta_index]:.3g}, size KS distance {avalanche_size_ks_distances[best_beta_index]:.3g}, diff from critical point {diff_from_critical[best_beta_index]:.3g}')
        min_beta_index = max(best_beta_index-1, 0)
        max_beta_index = min(best_beta_index+1, num_beta_multipliers_per_pass-1)
        min_beta_multiplier = beta_multipliers[min_beta_index]
        max_beta_multiplier = beta_multipliers[max_beta_index]
        if min_beta_multiplier == max_beta_multiplier:
            print(f'time {time.time()-code_start_time:.3f}, range of beta multipliers has converged to 0, quitting')
            break
    print(f'time {time.time()-code_start_time:.3f}, redoing counts for best beta multiplier {best_beta_multiplier:.3g} so that we can save them')
    model.beta = best_beta_multiplier * original_beta
    branching_parameter, gap_duration_count, avalanche_duration_count, avalanche_size_count, avalanche_mean_size_for_duration = model.simulate_and_record_point_process_stats_pmb(num_steps=sim_length)
    print(f'branching parameter min {branching_parameter.min():.3g}, mean {branching_parameter.mean():.3g}, max {branching_parameter.max():.3g}')
    for counts, count_name in zip([branching_parameter, gap_duration_count, avalanche_duration_count, avalanche_size_count, avalanche_mean_size_for_duration], ['branching_parameter', 'gap_duration_count', 'avalanche_duration_count', 'avalanche_size_count', 'avalanche_mean_size_for_duration']):
        count_file_name = os.path.join(output_directory, f'{count_name}_{model_file_fragment}_betatimes_{best_beta_multiplier:.3g}_steps_{sim_length}.pt')
        torch.save(obj=counts, f=count_file_name)
        print(f'time {time.time()-code_start_time:.3f}, saved {count_file_name}')
    gap_duration_exponent, _ = fit_power_law(counts=gap_duration_count, count_name='gap_duration_count', num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, exponent_min=exponent_min, exponent_max=exponent_max, num_p_value_distances=num_p_value_distances)
    avalanche_duration_exponent, avalanche_duration_ks_distances[best_beta_index] = fit_power_law(counts=avalanche_duration_count, count_name='avalanche_duration_count', num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, exponent_min=exponent_min, exponent_max=exponent_max, num_p_value_distances=num_p_value_distances)
    avalanche_size_exponent, avalanche_size_ks_distances[best_beta_index] = fit_power_law(counts=avalanche_size_count, count_name='avalanche_size_count', num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, exponent_min=exponent_min, exponent_max=exponent_max, num_p_value_distances=num_p_value_distances)
    mean_size_for_duration_exponent, scale_factor, r_squared = fit_power_law_lstsq(mean_sizes=avalanche_mean_size_for_duration)
    diff_from_critical = (avalanche_duration_exponent-1)/(avalanche_size_exponent-1) - mean_size_for_duration_exponent
    print(f'(duration_exponent-1)/(size_exponent-1) - mean_size_for_duration_exponent = ({avalanche_duration_exponent:.3g}-1)/({avalanche_size_exponent:.3g}-1) - {mean_size_for_duration_exponent:.3g} = {diff_from_critical:.3g}')
    print(f'time {time.time()-code_start_time:.3f}, done')