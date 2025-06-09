import os
import torch
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
        for try_exponent in torch.linspace(start=highest_inf, end=lowest_non_inf, steps=num_tries_per_pass, dtype=float_type, device=device):
            if torch.isinf( torch.special.zeta(try_exponent, x_min) ):
                highest_inf = try_exponent
            else:
                lowest_non_inf = try_exponent
                break
    # print(f'final [1+{highest_inf-1:.3g}, 1+{lowest_non_inf-1:.3g}]')
    return lowest_non_inf

def get_max_possible_exponent(x_min:float, min_exponent:float, num_passes:int, num_tries_per_pass:int):
    # If the exponent is too large, zeta(exponent, x_min) is effectively equal to its limit at infinity.
    # If x_min is 1, the limit is 1.
    # Otherwise, the limit is 0.
    # Find this maximum value.
    # print(f'searching for highest exponent that gives a non-limit zeta function value for x_min {x_min:.3g}...')
    highest_non_0 = min_exponent
    # First, just increase the exponent exponentially until we get the limit.
    num_tries_per_pass_float = float(num_tries_per_pass)
    lowest_0 = num_tries_per_pass_float
    if x_min == 1.0:
        zeta_limit = 1.0
    else:
        zeta_limit = 0.0
    for _ in range(num_passes):
        lowest_0 *= num_tries_per_pass_float
        if torch.special.zeta(lowest_0, x_min) == zeta_limit:
            break
    # Then walk it back.
    for _ in range(num_passes):
        # print(f'pass {pass_index+1} [{highest_non_0:.3g}, {lowest_0:.3g}]')
        for try_exponent in torch.linspace(start=highest_non_0, end=lowest_0, steps=num_tries_per_pass, dtype=float_type, device=device):
            if torch.special.zeta(try_exponent, x_min) == zeta_limit:
                lowest_0 = try_exponent
                break
            else:
                highest_non_0 = try_exponent
    # print(f'final [{highest_non_0:.3g}, {lowest_0:.3g}]')
    return highest_non_0

def get_log_likelihood(exponent:float, x_min:float, total_count_from_x_min:float, total_sum_of_logs_from_x_min:float):
    return -total_count_from_x_min * torch.log( torch.special.zeta(exponent, x_min) ) - exponent * total_sum_of_logs_from_x_min

def get_best_exponent(x_min:float, total_count_from_x_min:float, total_sum_of_logs_from_x_min:float, num_passes:int, num_tries_per_pass:int):
    # The exponent needs to be larger than 1 for the zeta function to return a finite value.
    # First, we figure out how much above 1 the exponent needs to be.
    guess_min = get_min_possible_exponent(x_min=x_min, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass)
    # If the exponent is too large, zeta(exponent, x_min) goes to 0, which makes log(zeta) go to -Inf.
    # Find this maximum value.
    guess_max = get_max_possible_exponent(x_min=x_min, min_exponent=guess_min, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass)
    # print(f'total_count_from_x_min {total_count_from_x_min}')
    # print(f'total_sum_of_logs_from_x_min {total_sum_of_logs_from_x_min:.3g}')
    # print(f'searching for exponent that gives highest finite log likelihood for x_min {x_min:.3g}...')
    # We assume that the function is smooth and concave downward so that we do not need to worry about being caught in local maxima.
    best_exponent = guess_min
    best_log_likelihood = get_log_likelihood(exponent=best_exponent, x_min=x_min, total_count_from_x_min=total_count_from_x_min, total_sum_of_logs_from_x_min=total_sum_of_logs_from_x_min)
    for _ in range(num_passes):
        # print(f'pass {pass_index+1} [{guess_min:.3g}, {guess_max:.3g}], best {best_exponent:.3g}, ll {best_log_likelihood:.3g}')
        best_exponent_this_pass = guess_min
        best_log_likelihood_this_pass = get_log_likelihood(exponent=best_exponent_this_pass, x_min=x_min, total_count_from_x_min=total_count_from_x_min, total_sum_of_logs_from_x_min=total_sum_of_logs_from_x_min)
        for try_exponent in torch.linspace(start=guess_min, end=guess_max, steps=num_tries_per_pass, dtype=float_type, device=device):
            try_log_likelihood = get_log_likelihood(exponent=try_exponent, x_min=x_min, total_count_from_x_min=total_count_from_x_min, total_sum_of_logs_from_x_min=total_sum_of_logs_from_x_min)
            # print(f'try exponent {try_exponent:.3g}, ll {try_log_likelihood:.3g}')
            # needs to be >=, since we initialize best_log_likelihood_this_pass to the ll for guess_min
            if (try_log_likelihood >= best_log_likelihood_this_pass) and torch.logical_not(  torch.logical_or( torch.isinf(try_log_likelihood), torch.isnan(try_log_likelihood) )  ):
                guess_min = best_exponent_this_pass
                best_exponent_this_pass = try_exponent
                best_log_likelihood_this_pass = try_log_likelihood
            else:
                guess_max = try_exponent
                break
            if try_log_likelihood > best_log_likelihood:
                best_exponent = try_exponent
                best_log_likelihood = try_log_likelihood
    # print(f'final [{guess_min:.3g}, {guess_max:.3g}], best on final pass {best_exponent_this_pass:.3g}, ll {best_log_likelihood_this_pass:.3g}, best overall {best_exponent:.3g}, ll {best_log_likelihood:.3g}')
    return best_exponent, best_log_likelihood

# power_law_cdf[t,x_min_index,x_index] = CDF of power law P(values[:,x_index]) ~ ( values[:,x_index]**(-exponent[t,x_min_index]) )/zeta( exponent[t,x_min_index], values[:,x_min_index] )
def get_one_power_law_cdf(values:torch.Tensor, exponent:torch.Tensor):
    return torch.cumsum(  torch.pow(input=values, exponent=-1*exponent)/torch.special.zeta(exponent, values[0]), dim=-1  )

def get_cumsum_from_above(counts:torch.Tensor, count_dim:int):
    return torch.cumsum(  counts.flip( dims=(count_dim,) ), dim=count_dim  ).flip( dims=(count_dim,) )

def get_empirical_cdf(counts:torch.Tensor, count_dim:int=-1):
    return torch.cumsum( counts/counts.sum(dim=count_dim, keepdim=True), dim=count_dim )

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

def sample_ks_distances_low_mem(distribution_cdf:torch.Tensor, num_distances:int, samples_per_distance:int):
    # Create a sampling of KS distances of data sampled from the distribution described by the CDF.
    # distribution_cdf size (num_values,)
    distances = torch.zeros( size=(num_distances,), dtype=distribution_cdf.dtype, device=distribution_cdf.device )
    # all_indices (num_values,1)
    all_indices = torch.arange( start=1, end=distribution_cdf.size(dim=0)+1, dtype=distribution_cdf.dtype, device=distribution_cdf.device ).unsqueeze(dim=-1)
    for dist_index in range(num_distances):
        # uniform_sample size (samples_per_distance,)
        uniform_sample = torch.rand( size=(samples_per_distance,), dtype=distribution_cdf.dtype, device=distribution_cdf.device )
        # uniform_sample -> (samples_per_distance, 1) < distribution_cdf -> (1, num_values)
        # result (samples_per_distance, num_values) -count_nonzero-> (samples_per_distance,)
        sample_bin_index = torch.count_nonzero( uniform_sample.unsqueeze(dim=-1) < distribution_cdf.unsqueeze(dim=-2), dim=-1 )
        # (num_values, 1) == sample_bin_index -> (1, samples_per_distance)
        # result (num_values, samples_per_distance) -count_nonzero-> (num_values,)
        index_counts = torch.count_nonzero( all_indices == sample_bin_index.unsqueeze(dim=-2), dim=-1 )
        # still (num_values,)
        empirical_cdf = get_empirical_cdf(counts=index_counts)
        # (num_values,) - (num_values,)
        # result (num_values,) -max-> 1
        distances[dist_index] = get_ks_distance(empirical_cdf=empirical_cdf, distribution_cdf=distribution_cdf)
    return distances

def try_one_power_law(values:torch.Tensor, counts:torch.Tensor, total_count_from_x_min:float, total_sum_of_logs_from_x_min:float, num_passes:int, num_tries_per_pass:int, x_min:int, p_value_sample_size:int):
    # numerical maximization of log likelihood for discrete integer-valued variable 
    # See Eqs. 3.5 and 3.7 of Clauset et al. 2009.
    # We want a num_thresholds x num_values matrix of possible powers for the power law, one value for each (threshold, x_min) pair.
    # Use an approximate value as our initial guess.
    # Eq. 3.7 alpha approx.= 1 + n/sum[i=1...n](  ln( x_i/(x_min-1/2) )  )
    # For our purposes, it is easier to expand this out to 1 + n/(  sum[i=1...n]( ln(x_i) ) - n*ln(x_min-1/2)  ).
    # For a given threshold index t and x_min index x,
    # n = total_count_from_x_min[t,x]
    # ln(x_min-1/2) = log_x_min_minus_half[x] 
    # sum[i=1...n]( ln(x_i) ) = total_sum_of_logs_from_x_min[t,x]
    # Do a grid-based search for a value of alpha that can achieve a higher log-likelihood.
    # Use Eq. 3.5: LL(alpha) = -n*ln( zeta(alpha,x_min) ) - alpha*sum[i=1...n]( ln(x_i) ).
    # Search strategy:
    # 0. Start with a search interval alpha_min=1, alpha_max=1+log2(n).
    # Repeat the following loop num_passes times:
    # 1. Compute the width of the search interval w = alpha_max - alpha_min.
    # 2. Let d = w/m where m=num_tries_per_pass.
    # 3. Choose m values of alpha on the search interval: alpha_min+d, alpha_min+2d, ..., alpha_min+md = alpha_min+w = alpha_max.
    # 4. Compute the log likelihood for each choice of alpha.
    # 5. Store the one with the maximum log likelihood, alpha_best.
    # 6. Define a new search interval alpha_min=alpha_best-d, alpha_max=alpha_best+d.
    # print(f'num non-0 counts {num_non_0_counts}')
    # total_count_from_x_min = counts.sum()
    # print(f'total_count_from_x_min {total_count_from_x_min}')
    # total_sum_of_logs_from_x_min = torch.sum(count_times_log_value)
    # print(f'total_sum_of_logs_from_x_min {total_sum_of_logs_from_x_min:.3g}')
    best_exponent, best_log_likelihood = get_best_exponent(x_min=x_min, total_count_from_x_min=total_count_from_x_min, total_sum_of_logs_from_x_min=total_sum_of_logs_from_x_min, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass)
    # print(f'continuous case approximation exponent {approx_exponent:.3g} log likelihood {approx_log_likelihood:.3g}')
    # As an additional check, compute the Kolmogorov-Smirnov distance between the power law and imperical distributions.
    empirical_cdf = get_empirical_cdf(counts=counts)
    power_law_cdf = get_power_law_cdf(values=values, exponent=best_exponent)
    ks_distance = get_ks_distance( empirical_cdf=empirical_cdf, distribution_cdf=power_law_cdf )
    sampled_distances = sample_ks_distances_low_mem( distribution_cdf=power_law_cdf, num_distances=p_value_sample_size, samples_per_distance=int(total_count_from_x_min) )
    p_value = torch.count_nonzero(sampled_distances >= ks_distance)/sampled_distances.numel()
    # See Xu et al. 2021 Eq. 11.
    # exponential_indices = torch.arange(    start=0, end=int(   math.floor(  math.log2( counts.numel() )  )   ), step=1, dtype=float_type, device=device    ).exp2().floor().int()
    # kappa = torch.mean(   torch.abs(  get_one_power_law_cdf( values=values, exponent=torch.full_like(input=best_exponent, fill_value=1.5) )[exponential_indices] - empirical_cdf[exponential_indices]  ), dim=-1   )
    # For simplicity, just use KS distance instead of the mean distance at exponentially spaced indices.
    kappa = get_ks_distance(  empirical_cdf=empirical_cdf, distribution_cdf=get_one_power_law_cdf( values=values, exponent=torch.full_like(input=best_exponent, fill_value=1.5) )  )
    return best_exponent, best_log_likelihood, ks_distance, p_value, kappa

def try_power_laws_one_by_one(counts:torch.Tensor, num_passes:int, num_tries_per_pass:int, max_x_min:int, p_value_sample_size:int):
    # All output Tensors are of suze num_thresholds=counts.size(dim=0) by max_x_min.
    # This lets us guarantee that the stats Tensors for the real and phase-shuffled versions of the data are identical.
    # We only fill in values up to the smaller of num_values and max_x_min.
    # This prevents us from running out of counts.
    num_thresholds, num_values = counts.size()
    # print( 'counts size', counts.size() )
    num_x_min = min( num_values, max_x_min )
    best_exponent = torch.zeros( size=(num_thresholds, max_x_min), dtype=float_type, device=device )
    best_log_likelihood = torch.zeros_like(best_exponent)
    approx_exponent = torch.zeros_like(best_exponent)
    approx_log_likelihood = torch.zeros_like(best_exponent)
    ks_distance = torch.zeros_like(best_exponent)
    p_value = torch.zeros_like(best_exponent)
    kappa = torch.zeros_like(best_exponent)
    values = torch.arange(start=1, end=num_values+1, step=1, dtype=float_type, device=device)
    values_2d = values.unsqueeze(dim=0)
    # count_times_log_value = counts * torch.unsqueeze( input=torch.log(values), dim=0 )
    # In each row, add from the end of the row up to the current column to get the sum from x_min to the maximum value encountered.
    total_count_from_x_min_all = get_cumsum_from_above(counts=counts, count_dim=-1)
    total_sum_of_logs_from_x_min_all = get_cumsum_from_above( counts=counts * torch.log(values_2d), count_dim=-1 )
    approx_exponent = 1 + total_count_from_x_min_all/( total_sum_of_logs_from_x_min_all - total_count_from_x_min_all * torch.log(values_2d - 0.5) )
    approx_log_likelihood = get_log_likelihood(exponent=approx_exponent, x_min=values_2d, total_count_from_x_min=total_count_from_x_min_all, total_sum_of_logs_from_x_min=total_count_from_x_min_all)
    count_gt_0 = counts > 0
    # exponential_indices = torch.arange( start=0, end=math.log2(num_values), step=1, dtype=float_type, device=device ).exp2().floor().int()
    for threshold_index in range(num_thresholds):
        counts_for_threshold = counts[threshold_index,:]
        # ctlv_for_threshold = count_times_log_value[threshold_index,:]
        count_gt_0_for_threshold = count_gt_0[threshold_index,:]
        total_count_from_x_min_for_threshold = total_count_from_x_min_all[threshold_index,:]
        total_sum_of_logs_from_x_min_for_threshold = total_sum_of_logs_from_x_min_all[threshold_index,:]
        for x_min_index in range(num_x_min):
            x_min = values[x_min_index]
            total_count_from_x_min = total_count_from_x_min_for_threshold[x_min_index]
            total_sum_of_logs_from_x_min = total_sum_of_logs_from_x_min_for_threshold[x_min_index]
            values_for_x_min = values[x_min_index:]
            counts_for_x_min = counts_for_threshold[x_min_index:]
            # ctlv_for_x_min = ctlv_for_threshold[x_min_index:]
            count_gt_0_for_x_min = count_gt_0_for_threshold[x_min_index:]
            num_nonzero_counts = torch.count_nonzero(count_gt_0_for_x_min)
            if num_nonzero_counts > 3:
                # values_for_non_0 = values_for_x_min[count_gt_0_for_x_min]
                # counts_for_non_0 = counts_for_x_min[count_gt_0_for_x_min]
                # ctlv_for_non_0 = ctlv_for_x_min[count_gt_0_for_x_min]
                best_exponent[threshold_index, x_min_index], best_log_likelihood[threshold_index, x_min_index], ks_distance[threshold_index, x_min_index], p_value[threshold_index, x_min_index], kappa[threshold_index, x_min_index] = try_one_power_law(values=values_for_x_min, counts=counts_for_x_min, total_count_from_x_min=total_count_from_x_min, total_sum_of_logs_from_x_min=total_sum_of_logs_from_x_min, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, x_min=x_min, p_value_sample_size=p_value_sample_size)
                print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1} of {num_thresholds}, x_min {x_min_index+1} of {num_x_min} best exponent {best_exponent[threshold_index, x_min_index]:.3g} best log likelihood {best_log_likelihood[threshold_index, x_min_index]:.3g} approx exponent {approx_exponent[threshold_index, x_min_index]:.3g} approx log likelihood {approx_log_likelihood[threshold_index, x_min_index]:.3g} KS distance {ks_distance[threshold_index, x_min_index]:.3g} p-value {p_value[threshold_index, x_min_index]:.3g} kappa {kappa[threshold_index, x_min_index]:.3g}')
            else:
                print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1} of {num_thresholds}, only {num_nonzero_counts} non-0 counts')
    return best_exponent, best_log_likelihood, approx_exponent, approx_log_likelihood, ks_distance, p_value, kappa

def compute_and_save_power_law(counts:torch.Tensor, num_passes:int, num_tries_per_pass:int, output_directory:str, count_name:str, out_file_suffix:str, max_x_min:float, p_value_sample_size:int):
    best_exponent, best_log_likelihood, approx_exponent, approx_log_likelihood, ks_distance, p_value, kappa = try_power_laws_one_by_one(counts=counts, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, max_x_min=max_x_min, p_value_sample_size=p_value_sample_size)
    for stat_to_store, stat_name in zip([best_exponent, best_log_likelihood, approx_exponent, approx_log_likelihood, ks_distance, p_value, kappa],['best_exponent', 'best_log_likelihood', 'approx_exponent', 'approx_log_likelihood', 'ks_distance', 'p_value', 'kappa']):
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

def try_positive_power_laws_one_by_one(counts:torch.Tensor, max_x_min:int):
    num_thresholds, num_values = counts.size()
    num_x_min = min(num_values, max_x_min)
    # Use max_x_min to enforce a consistent size of the output Tensors.
    exponent = torch.zeros( size=(num_thresholds, max_x_min), dtype=float_type, device=device )
    scale_factor = torch.zeros_like(exponent)
    r_squared = torch.zeros_like(exponent)
    log_values = torch.log( torch.arange(start=1, end=num_values+1, step=1, dtype=float_type, device=device) )
    log_values_and_1s = torch.stack(  ( log_values, torch.ones_like(log_values) ), dim=-1  )
    log_counts = torch.log(counts).unsqueeze(dim=-1)
    count_gt_0 = counts > 0
    for threshold_index in range(num_thresholds):
        log_counts_for_threshold = log_counts[threshold_index,:,:]
        count_gt_0_for_threshold = count_gt_0[threshold_index,:]
        for x_min_index in range(num_x_min):
            # print(f'threshold {threshold_index+1} of {num_thresholds}, x_min {x_min_index+1} of {num_x_min}')
            log_values_and_1s_for_x_min = log_values_and_1s[x_min_index:,:]
            log_counts_for_x_min = log_counts_for_threshold[x_min_index:,:]
            count_gt_0_for_x_min = count_gt_0_for_threshold[x_min_index:]
            log_values_and_1s_for_non_0 = log_values_and_1s_for_x_min[count_gt_0_for_x_min,:]
            log_counts_for_non_0 = log_counts_for_x_min[count_gt_0_for_x_min,:]
            exponent[threshold_index, x_min_index], scale_factor[threshold_index, x_min_index] , r_squared[threshold_index, x_min_index] = try_one_positive_power_law(log_values_and_1s=log_values_and_1s_for_non_0, log_counts=log_counts_for_non_0)
            print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1} of {num_thresholds}, x_min {x_min_index+1} of {num_x_min} found power law log(<S>) = {exponent[threshold_index, x_min_index]:.3g}log(T) + {scale_factor[threshold_index, x_min_index]:.3g} with R^2 {r_squared[threshold_index, x_min_index]:.3g}')
    return exponent, scale_factor, r_squared

def compute_and_save_positive_power_law(counts:torch.Tensor, output_directory:str, count_name:str, out_file_suffix:str, max_x_min:int):
    exponent, scale_factor, r_squared = try_positive_power_laws_one_by_one(counts=counts, max_x_min=max_x_min)
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
    out_file_suffix = f'{data_subset}_thresholds_{num_thresholds}_from_{min_threshold:.3g}_to_{max_threshold:.3g}_maxxmin_{max_x_min}_passes_{num_passes}_tries_{num_tries_per_pass}_pvalsamples_{p_value_sample_size}'
    
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
    exponents = [ compute_and_save_power_law( counts=counts_tensor, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, output_directory=output_directory, count_name=count_name, out_file_suffix=out_file_suffix, max_x_min=max_x_min, p_value_sample_size=p_value_sample_size ) for (counts_tensor, count_name) in zip([gap_duration_counts, duration_counts, size_counts, mean_size_for_duration],['gap_duration', 'avalanche_duration', 'avalanche_size']) ]
    size_exponent = exponents[1]
    duration_exponent = exponents[2]
    mean_size_for_duration_exponent = compute_and_save_positive_power_law(counts=mean_size_for_duration, output_directory=output_directory, count_name='mean_size_for_duration', out_file_suffix=out_file_suffix, max_x_min=max_x_min)

    # See Xu et al. 2021 Eq. 9.
    diff_from_critical_point = ( duration_exponent.unsqueeze(dim=-2)-1 )/( size_exponent.unsqueeze(dim=-1)-1 ) - mean_size_for_duration_exponent.unsqueeze(dim=-2)
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
    parser.add_argument("-o", "--max_x_min", type=int, default=1200, help="x-min to use for fitting power law distributions, applied to both sizes and durations")
    parser.add_argument("-p", "--num_permutations", type=int, default=10, help="number of phase-shuffled surrogate time series for which to repeat the calculations")
    parser.add_argument("-q", "--p_value_sample_size", type=int, default=1000, help="number of sample KS distances to compute to estimate P(KS distance for data sampled from power law >= KS distance for actual data)")
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
    max_x_min = args.max_x_min
    print(f'max_x_min={max_x_min}')
    num_permutations = args.num_permutations
    print(f'num_permutations={num_permutations}')
    p_value_sample_size = args.p_value_sample_size
    print(f'p_value_sample_size={p_value_sample_size}')

    training_subject_indices = torch.arange(start=training_subject_start, end=training_subject_end, step=1, dtype=int_type, device=device)
    data_ts = load_and_standardize_ts(output_directory=output_directory, data_subset=data_subset, file_name_fragment=file_name_fragment, training_subject_indices=training_subject_indices)
    # non_shuffle_ts_test(data_ts=data_ts)
    original_file_suffix = do_all_the_tests(data_ts=data_ts, data_subset=data_subset)
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