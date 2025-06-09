import os
import torch
import numpy
import time
import argparse
import math

def load_and_standardize_ts(output_directory:str, data_subset:str, file_name_fragment:str, training_subject_indices:torch.Tensor):
    # Compute these stats using only the training data, since they inform the design of our ML models.
    # Flatten together scan and subject dims so we only need to keep track of one batch dim.
    data_ts_file = os.path.join(output_directory, f'data_ts_{data_subset}_{file_name_fragment}.pt')
    data_ts = torch.flatten( torch.load(data_ts_file)[:,training_subject_indices,:,:], start_dim=0, end_dim=1 )
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    data_ts -= data_ts_mean
    data_ts /= data_ts_std
    return data_ts

def non_shuffle_ts_test(data_ts:torch.Tensor, dim:int=-1, norm:str='backward'):
    num_ts = data_ts.size(dim=0)
    num_steps = data_ts.size(dim=-1)
    min_err = torch.zeros( size=(num_ts,), dtype=data_ts.dtype, device=data_ts.device )
    mean_err = torch.zeros_like(min_err)
    max_err = torch.zeros_like(min_err)
    for ts_index in range(num_ts):
        data_ts_i = data_ts[ts_index,:,:]
        data_ft = torch.fft.rfft(data_ts_i, n=num_steps, dim=dim, norm=norm)
        data_abs = data_ft.abs()
        data_angle = data_ft.angle()
        # data_angle +=  2 * math.pi * torch.rand_like(input=data_angle)
        data_ft = data_abs * ( torch.cos(data_angle) + 1j * torch.sin(data_angle) )
        new_data_ts = torch.fft.irfft(data_ft, n=num_steps, dim=dim, norm=norm)
        ts_abs_err = torch.abs(data_ts_i - new_data_ts)
        min_err[ts_index] = ts_abs_err.min()
        mean_err[ts_index] = ts_abs_err.mean()
        max_err[ts_index] = ts_abs_err.max()
    print(f'error from round-trip of time series data into angular coordinates of Fourier coefficients and back min {min_err.min():.3g}, mean {mean_err.mean():.3g}, max {max_err.max():.3g}')
    return (min_err, mean_err, max_err)

def phase_shuffle_ts(data_ts:torch.Tensor, dim:int=-1, norm:str='backward'):
    num_ts = data_ts.size(dim=0)
    num_steps = data_ts.size(dim=-1)
    for ts_index in range(num_ts):
        data_ft = torch.fft.rfft(data_ts[ts_index,:,:], n=num_steps, dim=dim, norm=norm)
        data_abs = data_ft.abs()
        data_angle = data_ft.angle()
        data_angle +=  2 * math.pi * torch.rand_like(input=data_angle)
        data_ft = data_abs * ( torch.cos(data_angle) + 1j * torch.sin(data_angle) )
        data_ts[ts_index,:,:] = torch.fft.irfft(data_ft, n=num_steps, dim=dim, norm=norm)
    return data_ts

def get_num_activations(data_ts:torch.Tensor, threshold:float):
    # Define an activation as a crossing of the threshold from below.
    return torch.count_nonzero( torch.diff( (data_ts >= threshold).int(), dim=-1 ) == 1, dim=-2 )

def get_branching_parameter(num_activations:torch.Tensor):
    ancestors = num_activations[:,:-1]
    descendants = num_activations[:,1:]
    descendants_per_ancestor = descendants/ancestors
    descendants_per_ancestor[ancestors == 0] = 0
    return descendants_per_ancestor.sum(dim=-1)/torch.count_nonzero(ancestors > 0, dim=-1)

def march_avalanche_counts(num_activations:torch.Tensor):
    num_ts, num_steps = num_activations.size()
    max_duration = num_steps-1
    max_size = num_activations.sum(dim=-1).max()
    all_durations = torch.unsqueeze( torch.arange(start=1, end=max_duration+1, dtype=int_type, device=device), dim=-1 )
    all_sizes = torch.unsqueeze( torch.arange(start=1, end=max_size+1, dtype=int_type, device=device), dim=-1 )
    has_any_diff = torch.diff( (num_activations > 0).int(), dim=-1 )
    has_avalanche_start = has_any_diff == 1
    has_avalanche_end = has_any_diff == -1
    num_activations = num_activations[:,1:]
    start_step = torch.zeros( size=(num_ts,), dtype=int_type, device=device )
    end_step = torch.zeros_like(start_step)
    size_so_far = torch.zeros_like(start_step)
    gap_duration_counts = torch.zeros( size=(max_duration,), dtype=int_type, device=device )
    duration_counts = torch.zeros( size=(max_duration,), dtype=int_type, device=device )
    size_counts = torch.zeros( size=(max_size,), dtype=int_type, device=device )
    mean_size_for_duration = torch.zeros( size=(max_duration,), dtype=float_type, device=device )
    for step in range(num_steps-1):
        is_at_start = has_avalanche_start[:,step]
        is_at_end = has_avalanche_end[:,step]
        start_step[is_at_start] = step
        end_step[is_at_end] = step
        size_so_far[is_at_start] = 0
        size_so_far += num_activations[:,step]
        is_duration = all_durations == (step - start_step[is_at_end]).unsqueeze(dim=0)
        final_size = size_so_far[is_at_end]
        gap_duration_counts += torch.count_nonzero( all_durations == (step - end_step[is_at_start]).unsqueeze(dim=0), dim=-1 )
        duration_counts += torch.count_nonzero( is_duration, dim=-1 )
        size_counts += torch.count_nonzero( all_sizes == final_size.unsqueeze(dim=0), dim=-1 )
        mean_size_for_duration += torch.matmul( is_duration.float(), final_size.unsqueeze(dim=-1).float() ).squeeze(dim=-1)
    duration_count_gt_0 = duration_counts > 0
    mean_size_for_duration[duration_count_gt_0] /= duration_counts[duration_count_gt_0]
    return gap_duration_counts, duration_counts, size_counts, mean_size_for_duration

def shorten_and_save_counts(counts_tensor:torch.Tensor, output_directory:str, count_name:str, out_file_suffix:str):
    print( f'{count_name} originally', counts_tensor.size() )
    max_count = torch.flatten(  torch.nonzero( torch.sum(counts_tensor, dim=0) )  )[-1]+1
    counts_tensor = counts_tensor[:,:max_count].clone()
    print( 'shortened to', counts_tensor.size() )
    counts_file = os.path.join(output_directory, f'{count_name}_{out_file_suffix}.pt')
    torch.save(obj=counts_tensor, f=counts_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {counts_file}')
    return counts_tensor

def get_min_possible_exponent(x_min:float, num_passes:int, num_tries_per_pass:int):
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
            print(f'converged to lowest exponent for which zeta is finite 1+{lowest_non_inf-1.0:.3g}')
            break
    # print(f'final [1+{highest_inf-1:.3g}, 1+{lowest_non_inf-1:.3g}]')
    return lowest_non_inf

def get_max_possible_exponent(x_min:float, min_exponent:float, num_passes:int, num_tries_per_pass:int):
    # If the exponent is too large, zeta(exponent, x_min) is effectively equal to its limit at infinity.
    # If x_min is 1, the limit is 1.
    # Otherwise, the limit is 0.
    # Find this maximum value.
    # print(f'searching for highest exponent that gives a non-limit zeta function value for x_min {x_min:.3g}...')
    highest_not_at_limit = min_exponent
    # First, just increase the exponent exponentially until we get the limit.
    num_tries_per_pass_float = float(num_tries_per_pass)
    lowest_at_limit = min_exponent * num_tries_per_pass_float
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
            print(f'search for highest possible exponent converged to {highest_not_at_limit:.3g}')
            break
    return highest_not_at_limit

def get_log_likelihood(exponent:torch.Tensor, x_min:float, total_count:torch.Tensor, total_sum_of_logs:torch.Tensor):
    return -total_count * torch.log( torch.special.zeta(exponent, x_min) ) - exponent * total_sum_of_logs

def get_best_exponent(x_min:float, total_count:torch.Tensor, total_sum_of_logs:torch.Tensor, min_possible_exponent:float, max_possible_exponent:float, num_passes:int, num_tries_per_pass:int):
    # print(f'total_count_from_x_min {total_count_from_x_min}')
    # print(f'total_sum_of_logs_from_x_min {total_sum_of_logs_from_x_min:.3g}')
    # print(f'searching for exponent that gives highest finite log likelihood for x_min {x_min:.3g}...')
    # We assume that the function is smooth and concave downward so that we do not need to worry about being caught in local maxima.
    best_exponent = torch.full_like( input=total_count, fill_value=min_possible_exponent )
    best_log_likelihood = get_log_likelihood(exponent=best_exponent, x_min=x_min, total_count=total_count, total_sum_of_logs=total_sum_of_logs)
    min_guess = min_possible_exponent
    max_guess = max_possible_exponent
    for pass_index in range(num_passes):
        # print(f'pass {pass_index+1} [{guess_min:.3g}, {guess_max:.3g}], best {best_exponent:.3g}, ll {best_log_likelihood:.3g}')
        try_exponents = torch.linspace(start=min_guess, end=max_guess, steps=num_tries_per_pass, dtype=float_type, device=device)
        try_log_likelihoods = get_log_likelihood(exponent=try_exponents, x_min=x_min, total_count=total_count, total_sum_of_logs=total_sum_of_logs)
        is_okay = torch.logical_not(  torch.logical_or( torch.isinf(try_log_likelihoods), torch.isnan(try_log_likelihoods) )  )
        try_exponents = try_exponents[is_okay]
        try_log_likelihoods = try_log_likelihoods[is_okay]
        best_index = torch.argmax(try_log_likelihoods)
        best_exponent = try_exponents[best_index]
        best_log_likelihood = try_log_likelihoods[best_index]
        min_index = max(best_index-1, 0)
        max_index = min(best_index+1, num_tries_per_pass-1)
        min_guess = try_exponents[min_index]
        max_guess = try_exponents[max_index]
        if min_guess == max_guess:
            print(f'search for best exponent converged to {best_exponent:.3g} on pass {pass_index+1}')
            break
    # print(f'final [{guess_min:.3g}, {guess_max:.3g}], best on final pass {best_exponent_this_pass:.3g}, ll {best_log_likelihood_this_pass:.3g}, best overall {best_exponent:.3g}, ll {best_log_likelihood:.3g}')
    return best_exponent, best_log_likelihood

def get_empirical_cdf(counts:torch.Tensor, count_dim:int=-1):
    return torch.cumsum( counts/counts.sum(dim=count_dim, keepdim=True), dim=count_dim )

def get_power_law_cdf(values:torch.Tensor, exponent:torch.Tensor):
    # See equation 2.7 of Clauset et al. 2009.
    return 1.0 - torch.special.zeta(exponent, values)/torch.special.zeta(exponent, values[0])

def get_ks_distance(empirical_cdf:torch.Tensor, distribution_cdf:torch.Tensor, prob_dim:int=-1):
    return torch.max( torch.abs(empirical_cdf - distribution_cdf), dim=prob_dim ).values

def sample_ks_distances_low_mem(distribution_cdf:torch.Tensor, num_distances:int, samples_per_distance:int, max_samples_at_once:int):
    # Create a sampling of KS distances of data sampled from the distribution described by the CDF.
    # distribution_cdf size (num_values,)
    distances = torch.zeros( size=(num_distances,), dtype=distribution_cdf.dtype, device=distribution_cdf.device )
    count_totals = torch.zeros_like(distances)
    # all_indices (num_values,1)
    # Have an extra bin for values that fall beyond the finite number of bins we have.
    # We will discard these later.
    all_indices = torch.arange( start=1, end=distribution_cdf.size(dim=0)+2, dtype=distribution_cdf.dtype, device=distribution_cdf.device )
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
        count_totals[dist_index] = counts.sum()
        empirical_cdf = get_empirical_cdf(counts=counts, count_dim=-1)
        distances[dist_index] = get_ks_distance(empirical_cdf=empirical_cdf, distribution_cdf=distribution_cdf, prob_dim=-1)
    # print(f'num samples used min {count_totals.min():.3g} mean {count_totals.nanmean():.3g} max {count_totals.max():.3g} 0s {torch.count_nonzero(count_totals == 0)} nans {torch.count_nonzero( torch.isnan(count_totals) )}')
    print(f'sampled KS distances min {distances.min():.3g} mean {distances.nanmean():.3g} max {distances.max():.3g} nans {torch.count_nonzero( torch.isnan(distances) )}')
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

# , max_samples_per_batch:int
def try_power_laws_one_by_one(counts:torch.Tensor, num_passes:int, num_tries_per_pass:int, p_value_sample_size:int):
    # All output Tensors are of suze num_thresholds=counts.size(dim=0) by max_x_min.
    # This lets us guarantee that the stats Tensors for the real and phase-shuffled versions of the data are identical.
    # We only fill in values up to the smaller of num_values and max_x_min.
    # This prevents us from running out of counts.
    num_thresholds, num_values = counts.size()
    best_exponent = torch.zeros( size=(num_thresholds,), dtype=float_type, device=device )
    best_log_likelihood = torch.zeros_like(best_exponent)
    ks_distance = torch.zeros_like(best_exponent)
    p_value = torch.zeros_like(best_exponent)
    x_min = 1.0
    values = torch.arange(start=x_min, end=x_min+num_values, step=1, dtype=float_type, device=device)
    values_2d = values.unsqueeze(dim=0)
    # In each row, add from the end of the row up to the current column to get the sum from x_min to the maximum value encountered.
    count_gt_0 = counts > 0
    num_nonzero_counts = torch.count_nonzero(count_gt_0, dim=-1)
    total_count = counts.sum(dim=-1)
    total_sum_of_logs = torch.sum( counts * torch.log(values_2d), dim=-1 )
    empirical_cdf = get_empirical_cdf(counts=counts, count_dim=-1)
    approx_exponent = 1 + total_count/( total_sum_of_logs - total_count * math.log(x_min - 0.5) )
    approx_log_likelihood = get_log_likelihood(exponent=approx_exponent, x_min=x_min, total_count=total_count, total_sum_of_logs=total_count)
    min_possible_exponent = get_min_possible_exponent(x_min=x_min, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass)
    max_possible_exponent = get_max_possible_exponent(x_min=x_min, min_exponent=min_possible_exponent, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass)
    for threshold_index in range(num_thresholds):
        total_count_for_threshold = total_count[threshold_index]
        total_sum_of_logs_for_threshold = total_sum_of_logs[threshold_index]
        num_nonzero_counts_for_threshold = num_nonzero_counts[threshold_index]
        empirical_cdf_for_threshold = empirical_cdf[threshold_index,:]
        if num_nonzero_counts_for_threshold > 3:
            best_exponent_for_threshold, best_log_likelihood_for_threshold = get_best_exponent(x_min=x_min, total_count=total_count_for_threshold, total_sum_of_logs=total_sum_of_logs_for_threshold, min_possible_exponent=min_possible_exponent, max_possible_exponent=max_possible_exponent, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass)
            power_law_cdf = get_power_law_cdf(values=values, exponent=best_exponent_for_threshold)
            ks_distance_for_threshold = get_ks_distance(empirical_cdf=empirical_cdf_for_threshold, distribution_cdf=power_law_cdf, prob_dim=-1)
            # p_value_for_threshold = torch.count_nonzero( sample_ks_distances_low_mem(distribution_cdf=power_law_cdf, num_distances=p_value_sample_size, samples_per_distance=total_count_for_threshold.int().int(), max_samples_at_once=max_samples_per_batch) >= ks_distance_for_threshold )/p_value_sample_size
            p_value_for_threshold = torch.count_nonzero(  sample_ks_distances_fine( exponent=best_exponent_for_threshold, num_distances=p_value_sample_size, num_samples=total_count_for_threshold.int() ) >= ks_distance_for_threshold  )/p_value_sample_size
            best_exponent[threshold_index] = best_exponent_for_threshold
            best_log_likelihood[threshold_index] = best_log_likelihood_for_threshold
            ks_distance[threshold_index] = ks_distance_for_threshold
            p_value[threshold_index] = p_value_for_threshold
            print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1} of {num_thresholds}, best exponent {best_exponent_for_threshold:.3g} best log likelihood {best_log_likelihood_for_threshold:.3g} approx exponent {approx_exponent[threshold_index]:.3g} approx log likelihood {approx_log_likelihood[threshold_index]:.3g} KS distance {ks_distance_for_threshold:.3g} p-value {p_value_for_threshold:.3g}')
        else:
            print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1} of {num_thresholds}, only {num_nonzero_counts_for_threshold} non-0 counts')
    return best_exponent, best_log_likelihood, approx_exponent, approx_log_likelihood, ks_distance, p_value
# , max_samples_per_batch:int , max_samples_per_batch=max_samples_per_batch
def compute_and_save_power_law(counts:torch.Tensor, num_passes:int, num_tries_per_pass:int, output_directory:str, count_name:str, out_file_suffix:str, p_value_sample_size:int):
    best_exponent, best_log_likelihood, approx_exponent, approx_log_likelihood, ks_distance, p_value = try_power_laws_one_by_one(counts=counts, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, p_value_sample_size=p_value_sample_size)
    for stat_to_store, stat_name in zip([best_exponent, best_log_likelihood, approx_exponent, approx_log_likelihood, ks_distance, p_value],['best_exponent', 'best_log_likelihood', 'approx_exponent', 'approx_log_likelihood', 'ks_distance', 'p_value']):
        stat_file = os.path.join(output_directory, f'{stat_name}_{count_name}_{out_file_suffix}.pt')
        torch.save(obj=stat_to_store, f=stat_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {stat_file}')
    return best_exponent

def try_one_positive_power_law(log_values_and_1s:torch.Tensor, log_counts:torch.Tensor):
    # This power law relationship is not for a probability distribution, and the exponent is not assumed to be negative.
    # Using the maximum likelihood estimator does not make sense.
    # As such, we revert to just using least squares regression in log-log space.
    # We still filter out 0 values, since these are cases where we did not encounter any avalanches of this size.
    # An avalanche, by definition, cannot be of size 0.
    num_non_0_counts = log_counts.numel()
    # print(f'num non-0 counts {num_non_0_counts}')
    if num_non_0_counts < 3:
        print('skipping...')
        # exponent = 0. We do not have a conventient impossible value for this, but getting exactly 0 is unlikely.
        # scale_factor = 0, because the counts are all 0.
        # r_squared = 0, because we cannot explain any of the variance. There is no variance to explain.
        return 0.0, 0.0, 0.0
    coeffs = torch.linalg.lstsq( log_values_and_1s, log_counts ).solution
    r_squared = 1 - torch.sum(  torch.square( log_counts - torch.matmul(log_values_and_1s, coeffs) )  )/torch.sum(  torch.square( log_counts - torch.mean(log_counts) )  )
    coeffs_flat = torch.flatten(coeffs)
    exponent = coeffs_flat[0]
    scale_factor = coeffs_flat[1]
    # print(f'time {time.time()-code_start_time:.3f}, found power law log(<S>) = {exponent:.3g}log(T) + {scale_factor:.3g} with R^2 {r_squared:.3g}')
    return exponent, scale_factor, r_squared

def try_positive_power_laws_one_by_one(counts:torch.Tensor):
    num_thresholds, num_values = counts.size()
    # Use max_x_min to enforce a consistent size of the output Tensors.
    exponent = torch.zeros( size=(num_thresholds,), dtype=float_type, device=device )
    scale_factor = torch.zeros_like(exponent)
    r_squared = torch.zeros_like(exponent)
    log_values = torch.log( torch.arange(start=1, end=num_values+1, step=1, dtype=float_type, device=device) )
    log_values_and_1s = torch.stack(  ( log_values, torch.ones_like(log_values) ), dim=-1  )
    log_counts = torch.log(counts).unsqueeze(dim=-1)
    count_gt_0 = counts > 0
    for threshold_index in range(num_thresholds):
        count_gt_0_for_threshold = count_gt_0[threshold_index,:]
        log_values_and_1s_for_non_0 = log_values_and_1s[count_gt_0_for_threshold,:]
        log_counts_for_non_0 = log_counts[threshold_index,count_gt_0_for_threshold,:]
        exponent[threshold_index], scale_factor[threshold_index] , r_squared[threshold_index] = try_one_positive_power_law(log_values_and_1s=log_values_and_1s_for_non_0, log_counts=log_counts_for_non_0)
        print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1} of {num_thresholds}, found power law log(<S>) = {exponent[threshold_index]:.3g}log(T) + {scale_factor[threshold_index]:.3g} with R^2 {r_squared[threshold_index]:.3g}')
    return exponent, scale_factor, r_squared

def compute_and_save_positive_power_law(counts:torch.Tensor, output_directory:str, count_name:str, out_file_suffix:str):
    exponent, scale_factor, r_squared = try_positive_power_laws_one_by_one(counts=counts)
    for stat_to_store, stat_name in zip([exponent, scale_factor, r_squared],['exponent', 'scale_factor', 'r_squared']):
        stat_file = os.path.join(output_directory, f'{stat_name}_{count_name}_{out_file_suffix}.pt')
        torch.save(obj=stat_to_store, f=stat_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {stat_file}')
    return exponent

def do_all_the_tests(data_ts:torch.Tensor, data_subset:str):
    num_ts, num_nodes, num_steps = data_ts.size()
    # A node cannot go from inactive to active in in one step and then go from inactive to active again in the next step.
    # The largest possible avalanche is one in which all nodes are alternating between active and inactive states with two groups of nodes antialigned with each other so that we do not encounter a gap.
    # Since we require that the avalanche have a gap before it and a gap after it, the longest it can last is num_steps-2.
    max_duration = num_steps-2# need a step before the end and a step after the end to know the avalanche or gap is complete
    # print(f'max duration {max_duration}')
    max_size = (num_nodes*max_duration)//2
    # print(f'max size {max_size}')
    thresholds = torch.linspace(start=min_threshold, end=max_threshold, steps=num_thresholds, dtype=float_type, device=device)
    branching_parameters = torch.zeros( size=(num_thresholds, num_ts), dtype=float_type, device=device )
    gap_duration_counts = torch.zeros( size=(num_thresholds, max_duration), dtype=float_type, device=device )
    duration_counts = torch.zeros( size=(num_thresholds, max_duration), dtype=float_type, device=device )
    size_counts = torch.zeros( size=(num_thresholds, max_size), dtype=float_type, device=device )
    mean_size_for_duration = torch.zeros( size=(num_thresholds, max_duration), dtype=float_type, device=device )
    for threshold_index in range(num_thresholds):
        threshold = thresholds[threshold_index]
        # print(f'threshold {threshold_index+1} of {num_thresholds}: {threshold:.3g}')
        num_activations = get_num_activations(data_ts=data_ts, threshold=threshold)
        branching_parameters[threshold_index,:] = get_branching_parameter(num_activations=num_activations)
        gap_duration_counts[threshold_index,:], duration_counts[threshold_index,:], current_size_counts, mean_size_for_duration[threshold_index,:] = march_avalanche_counts(num_activations=num_activations)
        size_counts[threshold_index,:current_size_counts.numel()] = current_size_counts
        print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1} of {num_thresholds} ({threshold:.3g}) activations {num_activations.sum()}, gaps {gap_duration_counts.sum()}, avalanches {duration_counts.sum()}')
    # print( f'NaNs in gap_duration_counts {torch.count_nonzero( torch.isnan(gap_duration_counts) )}' )
    # print( f'NaNs in duration_counts {torch.count_nonzero( torch.isnan(duration_counts) )}' )
    # print( f'NaNs in size_counts {torch.count_nonzero( torch.isnan(size_counts) )}' )
    # print( f'NaNs in mean_size_for_duration {torch.count_nonzero( torch.isnan(mean_size_for_duration) )}' )
    # To make the file size more reasonable, remove sizes beyond the largest one that actually occurs.
    # Use clone() so we do not save a view + all underlying data.
    out_file_suffix = f'{data_subset}_thresholds_{num_thresholds}_from_{min_threshold:.3g}_to_{max_threshold:.3g}_passes_{num_passes}_tries_{num_tries_per_pass}_pvalsamples_{p_value_sample_size}'
    
    branching_parameters_file = os.path.join(output_directory, f'branching_parameter_minus_1_{out_file_suffix}.pt')
    torch.save(obj=branching_parameters-1, f=branching_parameters_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {branching_parameters_file}')
    
    # Get a shorter version of the counts so that we only work with values for which at least one threshold has a non-0 count.
    counts_short_array = [ shorten_and_save_counts(counts_tensor=counts_tensor, output_directory=output_directory, count_name=count_name, out_file_suffix=out_file_suffix) for (counts_tensor, count_name) in zip([gap_duration_counts, duration_counts, size_counts, mean_size_for_duration],['gap_duration_counts', 'duration_counts', 'size_counts', 'mean_size_for_duration']) ]
    gap_duration_counts = counts_short_array[0]
    duration_counts = counts_short_array[1]
    size_counts = counts_short_array[2]
    mean_size_for_duration = counts_short_array[3]
    
    # Try fitting power laws to things.
    # , max_samples_per_batch=max_samples_per_batch
    exponents = [ compute_and_save_power_law( counts=counts_tensor, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, output_directory=output_directory, count_name=count_name, out_file_suffix=out_file_suffix, p_value_sample_size=p_value_sample_size ) for (counts_tensor, count_name) in zip([gap_duration_counts, duration_counts, size_counts, mean_size_for_duration],['gap_duration', 'avalanche_duration', 'avalanche_size']) ]
    duration_exponent = exponents[1]
    size_exponent = exponents[2]
    mean_size_for_duration_exponent = compute_and_save_positive_power_law(counts=mean_size_for_duration, output_directory=output_directory, count_name='mean_size_for_duration', out_file_suffix=out_file_suffix)

    # See Xu et al. 2021 Eq. 9.
    # diff_from_critical_point = ( duration_exponent.unsqueeze(dim=-2)-1 )/( size_exponent.unsqueeze(dim=-1)-1 ) - mean_size_for_duration_exponent.unsqueeze(dim=-2)
    diff_from_critical_point = ( duration_exponent-1 )/( size_exponent-1 ) - mean_size_for_duration_exponent
    diff_from_critical_point_file = os.path.join(output_directory, f'diff_from_critical_point_{out_file_suffix}.pt')
    torch.save(obj=diff_from_critical_point, f=diff_from_critical_point_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {diff_from_critical_point_file}')
    return out_file_suffix

# modes:
# 'above' to get the probability that the permuted data produces a value >= that of the real data.
# 'below' to get the probability that the permuted data produces a value <= that of the real data.
# 'outside' to get the probability that the permuted data produces an absolute value >= that of the real data.
# 'inside' to get the probability that the permuted data produces an absolute value <= that of the real data.
def compile_p_value(output_directory:str, file_prefix:str, original_file_suffix:str, permuted_file_suffixes:list, mode:str='above'):
    original_file = os.path.join(output_directory, f'{file_prefix}_{original_file_suffix}.pt')
    original_tensor = torch.unsqueeze( input=torch.load(original_file), dim=0 )# Add a batch dimension for comparison with the permuted versions.
    print(f'time {time.time()-code_start_time:.3f}, loaded {original_file}')
    permuted_tensor = torch.stack(  tensors=[torch.load( os.path.join(output_directory, f'{file_prefix}_{permuted_file_suffix}.pt') ) for permuted_file_suffix in permuted_file_suffixes], dim=0  )
    num_permutations = permuted_tensor.size(dim=0)
    print(f'time {time.time()-code_start_time:.3f}, loaded {num_permutations} versions computed with surrogate phase-shuffled data')
    if (mode == 'inside') or (mode == 'outside'):
        original_tensor.abs_()
        permuted_tensor.abs_()
    if (mode == 'below') or (mode == 'inside'):
        original_tensor *= -1
        permuted_tensor *= -1
    p_value = torch.count_nonzero( original_tensor <= permuted_tensor, dim=0 )
    p_value_file = os.path.join(output_directory, f'p_value_{file_prefix}_{original_file_suffix}_perms_{num_permutations}.pt')
    torch.save(obj=p_value, f=p_value_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {p_value_file}')
    return p_value, p_value_file

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Load and threshold unbinarized time series data. Make and save a Tensor counts such that counts[x] is the number of times x nodes flip in a single step.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-c", "--data_subset", type=str, default='all', help="'training', 'validation', 'testing', or 'all'")
    parser.add_argument("-d", "--file_name_fragment", type=str, default='as_is', help="part of the output file name between mean_state_[data_subset]_ or mean_state_product_[data_subset]_ and .pt")
    parser.add_argument("-f", "--min_threshold", type=float, default=0, help="minimum threshold in std. dev.s")
    parser.add_argument("-g", "--max_threshold", type=float, default=5, help="minimum threshold in std. dev.s")
    parser.add_argument("-i", "--num_thresholds", type=int, default=100, help="number of thresholds to try")
    parser.add_argument("-j", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-k", "--training_subject_end", type=int, default=670, help="one past last training subject index")
    parser.add_argument("-l", "--num_passes", type=int, default=10, help="number of iterations of maximum log likelihood search to do to find power law")
    parser.add_argument("-m", "--num_tries_per_pass", type=int, default=10, help="number of equally spaced exponents to try per iteration")
    parser.add_argument("-p", "--num_permutations", type=int, default=10, help="number of phase-shuffled surrogate time series for which to repeat the calculations")
    parser.add_argument("-q", "--p_value_sample_size", type=int, default=1000, help="number of sample KS distances to compute to estimate P(KS distance for data sampled from power law >= KS distance for actual data)")
    # parser.add_argument("-r", "--max_samples_per_batch", type=int, default=1000, help="set this to limit the number of samples we take in parallel when computing the KS distance p-value in order to avoid running out of memory")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
    data_subset = args.data_subset
    print(f'data_subset={data_subset}')
    file_name_fragment = args.file_name_fragment
    print(f'file_name_fragment={file_name_fragment}')
    min_threshold = args.min_threshold
    print(f'min_threshold={min_threshold}')
    max_threshold = args.max_threshold
    print(f'max_threshold={max_threshold}')
    num_thresholds = args.num_thresholds
    print(f'num_thresholds={num_thresholds}')
    training_subject_start = args.training_subject_start
    print(f'training_subject_start={training_subject_start}')
    training_subject_end = args.training_subject_end
    print(f'training_subject_end={training_subject_end}')
    num_passes = args.num_passes
    print(f'num_passes={num_passes}')
    num_tries_per_pass = args.num_tries_per_pass
    print(f'num_tries_per_pass={num_tries_per_pass}')
    num_permutations = args.num_permutations
    print(f'num_permutations={num_permutations}')
    p_value_sample_size = args.p_value_sample_size
    print(f'p_value_sample_size={p_value_sample_size}')
    # max_samples_per_batch = args.max_samples_per_batch
    # print(f'pmax_samples_per_batch={max_samples_per_batch}')

    training_subject_indices = torch.arange(start=training_subject_start, end=training_subject_end, step=1, dtype=int_type, device=device)
    data_ts = load_and_standardize_ts(output_directory=output_directory, data_subset=data_subset, file_name_fragment=file_name_fragment, training_subject_indices=training_subject_indices)
    # non_shuffle_ts_test(data_ts=data_ts)
    original_file_suffix = do_all_the_tests(data_ts=data_ts, data_subset=data_subset)
    if num_permutations > 0:
        permuted_file_suffixes = [do_all_the_tests( data_ts=phase_shuffle_ts(data_ts=data_ts), data_subset=f'{data_subset}_perm_{perm_index+1}' ) for perm_index in range(num_permutations)]
        # Skip 'mean_size_for_duration', because it is not a probability distribution, meaning we do not have log likelihoods, a KS distance, or a kappa for it.
        for count_name in ['gap_duration', 'avalanche_duration', 'avalanche_size']:
            # Skip the exponents, because we do not have any specific expectations about whether they should be high or low or close to or far from 0.
            for stat_name, comparison_mode in zip(['best_log_likelihood', 'approx_log_likelihood', 'ks_distance', 'kappa'],['above', 'above', 'below', 'below']):
                compile_p_value(output_directory=output_directory, file_prefix=f'{stat_name}_{count_name}', original_file_suffix=original_file_suffix, permuted_file_suffixes=permuted_file_suffixes, mode=comparison_mode)
        # What are branching parameters supposed to be?
        compile_p_value(output_directory=output_directory, file_prefix='r_squared_mean_size_for_duration', original_file_suffix=original_file_suffix, permuted_file_suffixes=permuted_file_suffixes, mode='above')
        compile_p_value(output_directory=output_directory, file_prefix='branching_parameter_minus_1', original_file_suffix=original_file_suffix, permuted_file_suffixes=permuted_file_suffixes, mode='inside')
        compile_p_value(output_directory=output_directory, file_prefix='diff_from_critical_point', original_file_suffix=original_file_suffix, permuted_file_suffixes=permuted_file_suffixes, mode='inside')

    print(f'time {time.time()-code_start_time:.3f}, done')