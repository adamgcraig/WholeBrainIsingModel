import os
import torch
import time
import argparse

def load_and_standardize_ts(output_directory:str, training_subject_start:int, training_subject_end:int):
    # Compute these stats using only the training data, since they inform the design of our ML models.
    # Flatten together scan and subject dims so we only need to keep track of one batch dim.
    data_ts_file = os.path.join(output_directory, f'data_ts_all_as_is.pt')
    data_ts = torch.flatten( torch.load(data_ts_file)[:,training_subject_start:training_subject_end,:,:], start_dim=0, end_dim=1 )
    data_ts_std, data_ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    data_ts -= data_ts_mean
    data_ts /= data_ts_std
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

def march_avalanche_counts(num_activations:torch.Tensor, max_size:int):
    num_ts, num_steps = num_activations.size()
    max_duration = num_steps-1
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

def get_counts(output_directory:str, out_file_suffix:str, training_subject_start:int, training_subject_end:int, num_thresholds:int, min_threshold:float, max_threshold:float):

    data_ts = load_and_standardize_ts(output_directory=output_directory, training_subject_start=training_subject_start, training_subject_end=training_subject_end)
    num_ts, num_nodes, num_steps = data_ts.size()
    # A node cannot go from inactive to active in in one step and then go from inactive to active again in the next step.
    # The largest possible avalanche is one in which all nodes are alternating between active and inactive states with two groups of nodes antialigned with each other so that we do not encounter a gap.
    # Since we require that the avalanche have a gap before it and a gap after it, the longest it can last is num_steps-2.
    max_duration = num_steps-2# need a step before the end and a step after the end to know the avalanche or gap is complete
    # print(f'max duration {max_duration}')
    max_size = (num_nodes*max_duration)//2
    # print(f'max size {max_size}')
    float_type = data_ts.dtype
    device = data_ts.device
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
        gap_duration_counts[threshold_index,:], duration_counts[threshold_index,:], size_counts[threshold_index,:], mean_size_for_duration[threshold_index,:] = march_avalanche_counts(num_activations=num_activations, max_size=max_size)
        print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1} of {num_thresholds} ({threshold:.3g}) activations {num_activations.sum()}, gaps {gap_duration_counts.sum()}, avalanches {duration_counts.sum()}')
    branching_parameters_file = os.path.join(output_directory, f'branching_parameter_minus_1_{out_file_suffix}.pt')
    torch.save(obj=branching_parameters-1, f=branching_parameters_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {branching_parameters_file}')
    # To make the file size more reasonable, remove sizes beyond the largest one that actually occurs.
    # Use clone() so we do not save a view + all underlying data.
    # Get a shorter version of the counts so that we only work with values for which at least one threshold has a non-0 count.
    counts_short_array = [ shorten_and_save_counts(counts_tensor=counts_tensor, output_directory=output_directory, count_name=count_name, out_file_suffix=out_file_suffix) for (counts_tensor, count_name) in zip([gap_duration_counts, duration_counts, size_counts, mean_size_for_duration],['gap_duration_counts', 'duration_counts', 'size_counts', 'mean_size_for_duration']) ]
    gap_duration_counts = counts_short_array[0]
    duration_counts = counts_short_array[1]
    size_counts = counts_short_array[2]
    mean_size_for_duration = counts_short_array[3]
    return gap_duration_counts, duration_counts, size_counts, mean_size_for_duration

def get_max_possible_exponent(values:torch.Tensor, min_exponent:torch.Tensor, num_passes:int, num_tries_per_pass:int):
    # We assume that
    # values contains positive integer values,
    # min_exponent is a float >= 1 (should be strictly > 1 for CDF to be finite, but we are working with a finite subset of values),
    # num_passes is an integer >= 1 (need to try some values at least once),
    # num_tries_per_pass is an integer > 1 (need to try more than one value to home in on which is better).
    # We are looking for the largest float A such that sum( pow(values, -A) ) > 0.
    # We take the log of this sum when calculating the log likelihood, so we need it to be positive.
    values = values[values > 1]
    highest_not_at_limit_exponent = min_exponent
    # First, just increase the exponent exponentially until we get the limit.
    lowest_at_limit_exponent = torch.tensor(data=[float(num_tries_per_pass)], dtype=values.dtype, device=values.device)
    for _ in range(num_passes):
        lowest_at_limit_exponent.square_()
        if torch.abs(  torch.sum( torch.pow(input=values, exponent=-lowest_at_limit_exponent) )  ) <= precision:
            # print(f'found exponent such that sum of powers of counts is 0: {lowest_at_limit_exponent.item():.3g}')
            break
    lowest_at_limit_exponent = lowest_at_limit_exponent
    # Then walk it back.
    # print('searching for maximum usable exponent')
    for pass_index in range(num_passes):
        # print(f'pass {pass_index+1} [{highest_not_at_limit_exponent.item():.3g}, {lowest_at_limit_exponent.item():.3g}], range {(lowest_at_limit_exponent - highest_not_at_limit_exponent).item():.3g}')
        try_exponents = torch.linspace(start=highest_not_at_limit_exponent.item(), end=lowest_at_limit_exponent.item(), steps=num_tries_per_pass, dtype=float_type, device=device)
        at_limit_indices = torch.nonzero(    torch.abs(   torch.sum(  torch.pow( input=torch.unsqueeze(input=values, dim=0), exponent=torch.unsqueeze(input=-try_exponents, dim=1) ), dim=1  )   ) <= precision    )
        # If no exponent zeroed everything, stop.
        if at_limit_indices.numel() == 0:
            # print(f'reached point where no exponents were too high after {highest_not_at_limit_exponent.item():.3g}')
            break
        lowest_at_limit_index = at_limit_indices[0]
        lowest_at_limit_exponent = try_exponents[lowest_at_limit_index]
        # If all exponents in this range zeroed everything, stop.
        if lowest_at_limit_index == 0:
            # print(f'reached point where all exponents were too high after {highest_not_at_limit_exponent.item():.3g}')
            break
        highest_not_at_limit_exponent = try_exponents[lowest_at_limit_index-1]
        if torch.abs(highest_not_at_limit_exponent - lowest_at_limit_exponent) <= precision:
            # print(f'search for highest possible exponent converged to {highest_not_at_limit_exponent.item():.3g}')
            break
    return highest_not_at_limit_exponent

def get_best_exponent(values:torch.Tensor, total_sum_of_logs_over_total_count:torch.Tensor, num_passes:int, num_tries_per_pass:int):
    # print(f'total_count_from_x_min {total_count_from_x_min}')
    # print(f'total_sum_of_logs_from_x_min {total_sum_of_logs_from_x_min:.3g}')
    # print(f'searching for exponent that gives highest finite log likelihood for x_min {x_min:.3g}...')
    # We assume that the function is smooth and concave downward so that we do not need to worry about being caught in local maxima.
    neg_inf = -float("Inf")
    best_exponent = torch.full( size=(1,), fill_value=neg_inf, dtype=values.dtype, device=values.device )
    best_log_likelihood = best_exponent.clone()
    min_guess = torch.ones_like(input=best_exponent)# For theoretical reasons, we consider 1 the minimum possible exponent.
    max_guess = get_max_possible_exponent(values=values, min_exponent=min_guess, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass)
    # print('searching for maximum likelihood power law exponent')
    for pass_index in range(num_passes):
        # print(f'pass {pass_index+1} [{min_guess.item():.3g}, {max_guess.item():.3g}], range {(max_guess-min_guess).item():.3g}, best so far {best_exponent.item():.3g}, ll {best_log_likelihood.item():.3g}')
        try_exponents = torch.linspace( start=min_guess.item(), end=max_guess.item(), steps=num_tries_per_pass, dtype=best_exponent.dtype, device=best_exponent.device )
        try_log_likelihoods = -torch.log(   torch.sum(  torch.pow( torch.unsqueeze(input=values, dim=0), torch.unsqueeze(input=-try_exponents, dim=1) ), dim=1  )   ) - try_exponents * total_sum_of_logs_over_total_count
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
        if torch.abs(min_guess - max_guess) <= precision:
            # print(f'search for best exponent converged to {best_exponent.item():.3g}, ll {best_log_likelihood.item():.3g} on pass {pass_index+1}')
            break
    # print(f'final exponent {best_exponent.item():.3g}, ll {best_log_likelihood.item():.3g} on pass {pass_index+1}')
    return best_exponent, best_log_likelihood

def get_empirical_cdf(counts:torch.Tensor):
    count_cs = torch.cumsum( counts.flatten(), dim=0 )
    return count_cs/count_cs[-1]

def get_power_law_cdf(values:torch.Tensor, exponent:torch.Tensor):
    return get_empirical_cdf( counts=torch.pow(input=values, exponent=-exponent) )

def get_ks_distance(empirical_cdf:torch.Tensor, distribution_cdf:torch.Tensor):
    return torch.max( torch.abs(empirical_cdf - distribution_cdf) )

def sample_ks_distances(cdf:torch.Tensor, num_distances:int, num_points_per_distance:int):
    ks_distance = torch.zeros( size=(num_distances,), dtype=cdf.dtype, device=cdf.device )
    cdf_with_zero = torch.cat(  tensors=( torch.tensor(data=[0], dtype=cdf.dtype, device=cdf.device), cdf ), dim=0  ).unsqueeze(dim=-1)
    for dist_index in range(num_distances):
        samples = torch.rand( size=(1,num_points_per_distance), dtype=cdf.dtype, device=cdf.device )
        bin_count = torch.sum( torch.diff(samples < cdf_with_zero, dim=0), dim=1 )
        empirical_cdf = get_empirical_cdf(counts=bin_count)
        ks_distance[dist_index] = get_ks_distance(empirical_cdf=empirical_cdf, distribution_cdf=cdf)
    return ks_distance

# , max_samples_per_batch:int
def try_power_laws_one_by_one(counts:torch.Tensor, num_passes:int, num_tries_per_pass:int, p_value_sample_size:int, min_fit_points:int=3):
    # All output Tensors are of suze num_thresholds=counts.size(dim=0) by max_x_min.
    # This lets us guarantee that the stats Tensors for the real and phase-shuffled versions of the data are identical.
    # We only fill in values up to the smaller of num_values and max_x_min.
    # This prevents us from running out of counts.
    num_thresholds, num_values = counts.size()
    exponents = torch.zeros( size=(num_thresholds, num_values, num_values), dtype=float_type, device=device )
    log_likelihoods = torch.zeros_like(exponents)
    ks_distances = torch.zeros_like(exponents)
    p_values = torch.zeros_like(exponents)
    values = torch.arange(start=1, end=num_values+1, step=1, dtype=float_type, device=device)
    counts_times_log_values = counts * values.log().unsqueeze(dim=0)
    empirical_cdf = get_empirical_cdf(counts=counts)
    for threshold_index in range(num_thresholds):
        for x_min_index in range(num_values-min_fit_points):
            for x_max_index in range(x_min_index+min_fit_points-1, num_values):
                # print(f'threshold {threshold_index+1} of {num_thresholds}, value interval [{values[x_min_index]:.3g}, {values[x_max_index]:.3g}]')
                x_end = x_max_index+1
                counts_in_range = counts[threshold_index,x_min_index:x_end]
                num_nonzero_counts_in_range = torch.count_nonzero(counts_in_range > 0)
                if num_nonzero_counts_in_range < min_fit_points:
                    print(f'time {time.time() - code_start_time:.3f}, threshold {threshold_index+1}, x_min {values[x_min_index]:.3g}, x_max {values[x_max_index]:.3g} has {num_nonzero_counts_in_range} nonzero counts. Skip.')
                    continue
                total_count_in_range = torch.sum(counts_in_range)
                empirical_cdf = get_empirical_cdf(counts=counts_in_range)
                values_in_range = values[x_min_index:x_end]
                sum_of_logs_over_total_count = torch.sum(counts_times_log_values[threshold_index,x_min_index:x_end])/total_count_in_range
                exponent, log_likelihood = get_best_exponent(values=values_in_range, total_sum_of_logs_over_total_count=sum_of_logs_over_total_count, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass)
                power_law_cdf = get_power_law_cdf(values=values_in_range, exponent=exponent)
                ks_distance = get_ks_distance(empirical_cdf=empirical_cdf, distribution_cdf=power_law_cdf)
                p_value = torch.count_nonzero(  sample_ks_distances( cdf=power_law_cdf, num_distances=p_value_sample_size, num_points_per_distance=total_count_in_range.round().int().item() ) >= ks_distance  )/p_value_sample_size
                exponents[threshold_index, x_min_index, x_max_index] = exponent
                log_likelihoods[threshold_index, x_min_index, x_max_index] = log_likelihood
                ks_distances[threshold_index, x_min_index, x_max_index] = ks_distance
                p_values[threshold_index, x_min_index, x_max_index] = p_value
                print(f'time {time.time() - code_start_time:.3f}, threshold {threshold_index+1}, x_min {values[x_min_index]:.3g}, x_max {values[x_max_index]:.3g} has {num_nonzero_counts_in_range} nonzero counts, exponent {exponent:.3g}, log likelihood {log_likelihood:.3g}, threshold  KS distance {ks_distance:.3g}, p-value {p_value:.3g}')
    return exponents, log_likelihoods, ks_distances, p_values

def find_power_laws(counts:torch.Tensor, num_passes:int, num_tries_per_pass:int, initial_max_exponent:float=10, min_interval_length:int=3):
    # Pass in counts with dimensions (threshold, value).
    # For each triple of (threshold, x_min, x_max) such that x_max+1 - x_min >= min_fit_points, find the exponent that maximizes the log likelihood
    # ll = -log(  sum( counts[threshold,x_min_index:x_max_index]*pow(values[x_min_index:x_max_index], -exponent) )  ) - exponent * sum( counts[threshold,x_min_index:x_max_index]*log(values[x_min_index:x_max_index]) )/sum(counts[threshold,x_min_index:x_max_index])
    # where values = [1, ..., num_values].
    # num_thresholds, num_values = counts.size()
    num_values = counts.size(dim=-1)
    num_values_plus_1 = num_values+1
    # Set the size to (1,1,num_values) so that we can broadcast it to other Tensors where
    # dim0 is threshold,
    # dim1 is interval,
    # and dim2 is a singleton to be broadcast to individual values.
    values = torch.arange(start=1, end=num_values_plus_1, step=1, dtype=counts.dtype, device=counts.device).unsqueeze(dim=0).unsqueeze(dim=0)
    value_bounds = torch.triu_indices(row=num_values_plus_1, col=num_values_plus_1, offset=min_interval_length, dtype=int_type, device=counts.device)
    values_start = value_bounds[0]
    values_end = value_bounds[1]
    # num_intervals = values_start.numel()
    value_indices_row = torch.arange(start=0, end=num_values, dtype=int_type, device=counts.device).unsqueeze(dim=0)
    mask = torch.logical_and( values_start.unsqueeze(dim=1) <= value_indices_row, value_indices_row < values_end.unsqueeze(dim=1) ).float()
    # Set up masked_counts[threshold_index,interval_index,value_index] such that,
    # if values[value_index] is in values[ values_start[interval_index]:values_end[interval_index] ],
    # then masked_counts[threshold_index,interval_index,value_index] = counts[threshold_index,value_index],
    # or, otherwise, 0.
    masked_counts = mask.unsqueeze(dim=0) * counts.unsqueeze(dim=1)
    sum_of_counts = masked_counts.sum(dim=-1, keepdim=True)
    # Some counts are 0 without masking.
    # Only keep an interval if it has 1 or more non-0 counts for 1 or more thresholds.
    interval_has_nonzero = sum_of_counts.squeeze(dim=-1).sum(dim=0) > 0
    mask = mask[interval_has_nonzero,:]
    masked_counts = masked_counts[:,interval_has_nonzero,:]
    sum_of_counts = sum_of_counts[:,interval_has_nonzero,:]
    # Add up the logs of the values found in each interval.
    sum_of_logs = torch.sum( masked_counts * values.log(), dim=-1, keepdim=True )
    # For every pairing of threshold and valid interval, we find a different power law.
    exponents = torch.zeros_like(input=sum_of_logs)
    log_likelihoods = torch.full_like( input=exponents, fill_value=-float('Inf') )
    # log_values = values.log()
    # Probability should be decreasing with size.
    min_exponents = torch.full_like(input=exponents, fill_value=0.0)
    # Due to limits on the precision of the underlying data type, 1e6 is the largest choice of exponent it is practical to test.
    max_exponents = torch.full_like(input=exponents, fill_value=initial_max_exponent)
    # We march through the values one at a time, incorporating them into sums that we need if they pass through a given mask.
    # sum_of_powers = torch.zeros_like(exponents)
    # sum_of_logs = torch.zeros_like(exponents)
    # sum_of_counts = torch.zeros_like(exponents)
    for pass_index in range(num_passes):
        delta_exponents = (max_exponents - min_exponents)/float(num_tries_per_pass-1)
        candidate_exponents = min_exponents.clone()
        for try_index in range(num_tries_per_pass):
            # sum_of_powers.zero_()
            # sum_of_logs.zero_()
            # sum_of_counts.zero_()
            # for value_index in range(num_values):
            #     value = values[value_index]
            #     # Each value has a different count for each threshold.
            #     # Each value is either included or excluded from each interval.
            #     # As such, we have a different exponent to work on for each (threshold, interval) pair.
            #     masked_count = torch.logical_and( values_start <= value_index, value_index < values_end ).float().unsqueeze(dim=0) * counts[:,value_index].unsqueeze(dim=-1)
            #     sum_of_powers += masked_count * torch.pow(value, -candidate_exponents)
            #     sum_of_logs += masked_count * log_values[value_index]
            #     sum_of_counts += masked_count
            # Replace the current best guess if we find one with a greater log likelihood.
            sum_of_powers = torch.sum( torch.pow(input=values, exponent=-candidate_exponents), dim=-1, keepdim=True )
            candidate_log_likelihoods = -torch.log(sum_of_powers) - candidate_exponents * sum_of_logs/sum_of_counts
            has_nonzero = sum_of_counts > 0
            is_better = torch.logical_and(has_nonzero, candidate_log_likelihoods > log_likelihoods)
            exponents[is_better] = candidate_exponents[is_better]
            log_likelihoods[is_better] = candidate_log_likelihoods[is_better]
            # Increment to the next set of candidate exponents.
            candidate_exponents += delta_exponents
            delta_nonzero = delta_exponents[has_nonzero]
            exponents_nonzero = exponents[has_nonzero]
            ll_nonzero = log_likelihoods[has_nonzero]
            print(f'time {time.time()-code_start_time:.3f}, pass {pass_index+1}, try {try_index+1}, delta exponents min {delta_nonzero.min():.3g} mean {delta_nonzero.mean():.3g} max {delta_nonzero.max():.3g}, exponents min {exponents_nonzero.min():.3g} mean {exponents_nonzero.mean():.3g} max {exponents_nonzero.max():.3g}, log likelihoods min {ll_nonzero.min():.3g} mean {ll_nonzero.mean():.3g} max {ll_nonzero.max():.3g}, num updated {torch.count_nonzero(is_better)}')
        # Explore a neighborhood around the best exponent we have found that lies in between the closest two exponents we have already checked.
        min_exponents = exponents - delta_exponents
        max_exponents = exponents + delta_exponents
        print(f'time {time.time()-code_start_time:.3f}, pass {pass_index+1}, delta exponents min {delta_exponents.min():.3g} mean {delta_exponents.mean():.3g} max {delta_exponents.max():.3g}, exponents min {exponents.min():.3g} mean {exponents.mean():.3g} max {exponents.max():.3g}, log likelihoods min {log_likelihoods.min():.3g} mean {log_likelihoods.mean():.3g} max {log_likelihoods.max():.3g}')
    return values, masked_counts, sum_of_counts, exponents, log_likelihoods

def get_ks_distance_batch(values:torch.Tensor, masked_counts:torch.Tensor, exponents:torch.Tensor):
    empirical_cdf = masked_counts.cumsum(dim=-1)
    empirical_cdf /= empirical_cdf[:,:,-1].unsqueeze(dim=-1)
    power_law_cdf = torch.pow(input=values, exponent=-exponents).cumsum(dim=-1)
    power_law_cdf /= power_law_cdf[:,:,-1].unsqueeze(dim=-1)
    ks_distance = torch.max( torch.abs(empirical_cdf - power_law_cdf), dim=-1, keepdim=True ).values
    ks_nonzero = ks_distance[ks_distance > 0]
    print(f'time {time.time()-code_start_time:.3f}, completed empirical-vs-power-law KS distances {ks_nonzero.min():.3g} mean {ks_nonzero.mean():.3g} max {ks_nonzero.max():.3g}')
    return power_law_cdf, ks_distance

def sample_from_power_law_batch(power_law_cdf:torch.Tensor, sum_of_counts:torch.Tensor, true_ks_distances:torch.Tensor, p_value_sample_size:int):
    num_thresholds, num_intervals, _ = power_law_cdf.size()
    sample_cdf = torch.zeros_like(power_law_cdf)
    p_values = torch.zeros_like(sum_of_counts)
    sum_of_counts_int = sum_of_counts.round().int()
    has_nonzero = sum_of_counts > 0
    for sample_index in range(p_value_sample_size):
        for threshold_index in range(num_thresholds):
            for interval_index in range(num_intervals):
                sample_cdf[threshold_index, interval_index,:] = torch.count_nonzero( torch.rand( size=(1,sum_of_counts_int[threshold_index,interval_index,:]), dtype=power_law_cdf.dtype, device=power_law_cdf.device ) < power_law_cdf[threshold_index,interval_index,:].unsqueeze(dim=-1), dim=-1 )
        p_values += (  true_ks_distances <= torch.max( torch.abs(sample_cdf - power_law_cdf), dim=-1, keepdim=True ).values  )
        p_nonzero = p_values[has_nonzero]
        print(f'time {time.time()-code_start_time:.3f}, sample {sample_index+1} of {p_value_sample_size}, bigger KS distance counts so far {p_nonzero.min():.3g} mean {p_nonzero.mean():.3g} max {p_nonzero.max():.3g}')
    p_values /= p_value_sample_size
    return p_values

def compute_and_test_power_laws(counts:torch.Tensor, num_passes:int, num_tries_per_pass:int, p_value_sample_size:int, initial_max_exponent:float=10):
    values, masked_counts, sum_of_counts, exponents, log_likelihoods = find_power_laws(counts=counts, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, initial_max_exponent=initial_max_exponent)
    power_law_cdf, ks_distances = get_ks_distance_batch(values=values, masked_counts=masked_counts, exponents=exponents)
    p_values = sample_from_power_law_batch(power_law_cdf=power_law_cdf, sum_of_counts=sum_of_counts, true_ks_distances=ks_distances, p_value_sample_size=p_value_sample_size)
    return exponents, log_likelihoods, ks_distances, p_values

def get_interval_mask(num_values:int, device:str, min_interval_length:int=3):
    num_values_plus_1 = num_values+1
    value_bounds = torch.triu_indices(row=num_values_plus_1, col=num_values_plus_1, offset=min_interval_length, dtype=int_type, device=device)
    values_start = value_bounds[0]
    values_end = value_bounds[1]
    # num_intervals = values_start.numel()
    value_indices_row = torch.arange(start=0, end=num_values, dtype=int_type, device=device).unsqueeze(dim=0)
    return torch.logical_and( values_start.unsqueeze(dim=1) <= value_indices_row, value_indices_row < values_end.unsqueeze(dim=1) ).float()

def get_threshold_and_interval_have_sufficient_nonzero(counts:torch.Tensor, interval_mask:torch.Tensor, min_num_nonzero:int):
    num_thresholds = counts.size(dim=0)
    num_intervals = interval_mask.size(dim=0)
    threshold_and_interval_have_sufficient_nonzero = torch.zeros( size=(num_thresholds, num_intervals), dtype=torch.bool, device=counts.device )
    for threshold_index in range(num_thresholds):
        threshold_and_interval_have_sufficient_nonzero[threshold_index,:] = torch.count_nonzero( counts[threshold_index,:].unsqueeze(dim=0) * interval_mask, dim=-1 ) >= min_num_nonzero
    return threshold_and_interval_have_sufficient_nonzero

def find_power_laws_serial_thresholds(values:torch.Tensor, counts:torch.Tensor, interval_mask:torch.Tensor, num_passes:int, num_tries_per_pass:int, threshold_and_interval_have_sufficient_nonzero:torch.Tensor, initial_min_exponent:float=1.0, initial_max_exponent:float=1e6):
    # Pass in counts with dimensions (threshold, value).
    # For each triple of (threshold, x_min, x_max) such that x_max+1 - x_min >= min_fit_points, find the exponent that maximizes the log likelihood
    # ll = -log(  sum( counts[threshold,x_min_index:x_max_index]*pow(values[x_min_index:x_max_index], -exponent) )  ) - exponent * sum( counts[threshold,x_min_index:x_max_index]*log(values[x_min_index:x_max_index]) )/sum(counts[threshold,x_min_index:x_max_index])
    # where values = [1, ..., num_values].
    # num_thresholds, num_values = counts.size()
    num_thresholds = counts.size(dim=0)
    num_intervals = interval_mask.size(dim=0)
    best_exponents = torch.zeros( size=(num_thresholds, num_intervals), dtype=counts.dtype, device=counts.device )
    best_log_likelihoods = torch.zeros_like(best_exponents)
    for threshold_index in range(num_thresholds):
        # Set up masked_counts[interval_index,value_index] such that,
        # if values[value_index] is in values[ values_start[interval_index]:values_end[interval_index] ],
        # then masked_counts[interval_index,value_index] = counts[threshold_index,value_index],
        # or, otherwise, 0.
        # Some counts are 0 without masking.
        # Only keep an interval if it has the required number of distinct values with non-0 counts.
        interval_has_sufficient_nonzero = threshold_and_interval_have_sufficient_nonzero[threshold_index,:]
        if interval_has_sufficient_nonzero.count_nonzero() == 0:
            continue
        masked_counts = interval_mask[interval_has_sufficient_nonzero,:] * counts[threshold_index,:].unsqueeze(dim=0)
        sum_of_counts = masked_counts.sum(dim=-1, keepdim=True)
        # Add up the logs of the values found in each interval.
        sum_of_logs = torch.sum( masked_counts * values.log(), dim=-1, keepdim=True )
        # For every pairing of threshold and valid interval, we find a different power law.
        exponents = torch.zeros_like(input=sum_of_logs)
        log_likelihoods = torch.full_like( input=exponents, fill_value=-float('Inf') )
        # log_values = values.log()
        # Probability should be decreasing with size.
        original_min_exponents = torch.full_like(input=exponents, fill_value=initial_min_exponent)
        min_exponents = original_min_exponents.clone()
        # Due to limits on the precision of the underlying data type, 1e6 is the largest choice of exponent it is practical to test.
        original_max_exponents = torch.full_like(input=exponents, fill_value=initial_max_exponent)
        max_exponents = original_max_exponents.clone()
        # We march through the values one at a time, incorporating them into sums that we need if they pass through a given mask.
        # sum_of_powers = torch.zeros_like(exponents)
        # sum_of_logs = torch.zeros_like(exponents)
        # sum_of_counts = torch.zeros_like(exponents)
        for pass_index in range(num_passes):
            candidate_exponents = min_exponents.clone()
            delta_exponents = (max_exponents - min_exponents)/float(num_tries_per_pass-1)
            updates_in_pass = torch.zeros_like(candidate_exponents)
            for _ in range(num_tries_per_pass):
                # Replace the current best guess if we find one with a greater log likelihood.
                candidate_log_likelihoods = -torch.log(  torch.sum( torch.pow(input=values, exponent=-candidate_exponents), dim=-1, keepdim=True )  ) - candidate_exponents * sum_of_logs/sum_of_counts
                is_better = candidate_log_likelihoods > log_likelihoods
                exponents[is_better] = candidate_exponents[is_better]
                log_likelihoods[is_better] = candidate_log_likelihoods[is_better]
                # Increment to the next set of candidate exponents.
                candidate_exponents += delta_exponents
                updates_in_pass += is_better.float()
                # print(f'time {time.time()-code_start_time:.3f}, pass {pass_index+1}, try {try_index+1}, delta exponents min {delta_exponents.min():.3g} mean {delta_exponents.mean():.3g} max {delta_exponents.max():.3g}, candidate exponents min {candidate_exponents.min():.3g} mean {candidate_exponents.mean():.3g} max {candidate_exponents.max():.3g}, exponents min {exponents.min():.3g} mean {exponents.mean():.3g} max {exponents.max():.3g}, log likelihoods min {log_likelihoods.min():.3g} mean {log_likelihoods.mean():.3g} max {log_likelihoods.max():.3g}, num updated {torch.count_nonzero(is_better)}')
            # Explore a neighborhood around the best exponent we have found that lies in between the closest two exponents we have already checked.
            min_exponents = torch.maximum(input=exponents - delta_exponents, other=original_min_exponents)
            max_exponents = torch.minimum(input=exponents + delta_exponents, other=original_max_exponents)
            print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1}, pass {pass_index+1}, delta exponents min {delta_exponents.min():.3g} mean {delta_exponents.mean():.3g} max {delta_exponents.max():.3g}, exponents min {exponents.min():.3g} mean {exponents.mean():.3g} max {exponents.max():.3g}, log likelihoods min {log_likelihoods.min():.3g} mean {log_likelihoods.mean():.3g} max {log_likelihoods.max():.3g}, updates min {updates_in_pass.min():.3g} mean {updates_in_pass.mean():.3g} max {updates_in_pass.max():.3g}')
        # Record the final exponents and log likelihoods.
        best_exponents[threshold_index, interval_has_sufficient_nonzero] = exponents.squeeze(dim=-1)
        best_log_likelihoods[threshold_index, interval_has_sufficient_nonzero] = log_likelihoods.squeeze(dim=-1)
    return best_exponents, best_log_likelihoods

def get_ks_distance_serial_thresholds(values:torch.Tensor, counts:torch.Tensor, interval_mask:torch.Tensor, exponents:torch.Tensor, threshold_and_interval_have_sufficient_nonzero:torch.Tensor):
    num_thresholds = counts.size(dim=0)
    all_ks_distances = torch.ones_like(exponents)
    for threshold_index in range(num_thresholds):
        interval_has_sufficient_nonzero = threshold_and_interval_have_sufficient_nonzero[threshold_index,:]
        if torch.count_nonzero(interval_has_sufficient_nonzero) == 0:
            continue
        interval_mask_nonzero = interval_mask[interval_has_sufficient_nonzero,:]
        masked_counts = interval_mask_nonzero  * counts[threshold_index,:].unsqueeze(dim=0)
        empirical_cdf = masked_counts.cumsum(dim=-1)
        empirical_cdf /= empirical_cdf[:,-1].unsqueeze(dim=-1).clone()
        power_law_cdf = (  interval_mask_nonzero  * torch.pow( input=values, exponent=-exponents[threshold_index,interval_has_sufficient_nonzero].unsqueeze(dim=-1) )  ).cumsum(dim=-1)
        power_law_cdf /= power_law_cdf[:,-1].unsqueeze(dim=-1).clone()
        ks_distance = torch.max( torch.abs(empirical_cdf - power_law_cdf), dim=-1, keepdim=False ).values
        all_ks_distances[threshold_index,interval_has_sufficient_nonzero] = ks_distance
        print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1}, empirical-vs-power-law KS distances min {ks_distance.min():.3g} mean {ks_distance.mean():.3g} max {ks_distance.max():.3g}')
    return all_ks_distances

def sample_from_power_law_serial_thresholds(values:torch.Tensor, counts:torch.Tensor, interval_mask:torch.Tensor, exponents:torch.Tensor, true_ks_distances:torch.Tensor, p_value_sample_size:int, threshold_and_interval_have_sufficient_nonzero:torch.Tensor):
    num_thresholds, num_intervals = exponents.size()
    p_values = torch.zeros_like(exponents)
    values = values.squeeze(dim=0)
    num_values = values.numel()
    bin_edges = torch.zeros( size=(num_values+1,), dtype=values.dtype, device=values.device )
    bin_edges[:num_values] = values[:] - 0.5
    bin_edges[-1] = values[-1] + 0.5
    does_not_have_enough = torch.logical_not(input=threshold_and_interval_have_sufficient_nonzero)
    sample_ksd_for_threshold = torch.zeros( size=(num_intervals, p_value_sample_size), dtype=p_values.dtype, device=p_values.device )
    for threshold_index in range(num_thresholds):
        for interval_index in range(num_intervals):
            if does_not_have_enough[threshold_index, interval_index]:
                continue
            exponent = exponents[threshold_index,interval_index]
            current_interval = interval_mask[interval_index,:]
            power_law_cdf = ( torch.pow(input=values, exponent=-exponent) * current_interval ).cumsum(dim=-1)
            power_law_cdf /= power_law_cdf[-1].clone()
            power_law_cdf = power_law_cdf.unsqueeze(dim=0)
            num_sample_points = torch.sum(counts[threshold_index,:] * current_interval).round().int()
            sample_cdf = torch.zeros( size=(p_value_sample_size, num_values), dtype=power_law_cdf.dtype, device=power_law_cdf.device )
            power_law_cdf_unsq = power_law_cdf.unsqueeze(dim=-1)
            for sample_index in range(p_value_sample_size):
                sample_cdf[sample_index,:] = torch.count_nonzero(  torch.rand( size=(1, num_sample_points), dtype=power_law_cdf.dtype, device=power_law_cdf.device ) < power_law_cdf_unsq, dim=-1  )
            sample_cdf /= num_sample_points
            sample_ks_distances = torch.max( input=torch.abs(sample_cdf - power_law_cdf), dim=-1 ).values
            sample_ksd_for_threshold[interval_index,:] = sample_ks_distances
            # sample_cdf = torch.count_nonzero(  torch.rand( size=(p_value_sample_size, 1, num_sample_points), dtype=power_law_cdf.dtype, device=power_law_cdf.device ) < power_law_cdf.unsqueeze(dim=-1), dim=-1  )/num_sample_points
            # sample_cdf = torch.distributions.Categorical( probs=power_law_cdf.diff() ).sample( sample_shape=(p_value_sample_size, num_sample_points) ).histogram(bins=bin_edges, density=True)[0].cumsum(dim=0)
            # p_value = torch.count_nonzero(  torch.max( input=torch.abs(sample_cdf - power_law_cdf), dim=-1 ).values >= true_ks_distances[threshold_index,interval_index]  )/p_value_sample_size
            # p_value = torch.count_nonzero(     torch.max(    input=torch.abs(   torch.count_nonzero(  torch.rand( size=(p_value_sample_size, 1, num_sample_points), dtype=power_law_cdf.dtype, device=power_law_cdf.device ) < power_law_cdf.unsqueeze(dim=-1), dim=-1  )/num_sample_points - power_law_cdf   ), dim=-1    ).values >= true_ks_distances[threshold_index,interval_index]     )/p_value_sample_size
            # power_law_ks_distance = true_ks_distances[threshold_index,interval_index]
            # p_value = torch.count_nonzero(sample_ks_distances >= true_ks_distances[threshold_index,interval_index])/p_value_sample_size
            p_values[threshold_index,interval_index] = torch.count_nonzero(sample_ks_distances >= true_ks_distances[threshold_index,interval_index])/p_value_sample_size
            # print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1}, interval {interval_index+1}, sample size {num_sample_points}, exponent {exponent:.3g}, true KSD {power_law_ks_distance:.3g}, sample KSD min {sample_ks_distances.min():.3g}, mean {sample_ks_distances.mean():.3g}, max {sample_ks_distances.max():.3g}, p-value {p_value:.3g}')
        exponents_for_threshold = exponents[threshold_index,:]
        ks_distances_for_threshold = true_ks_distances[threshold_index,:]
        p_values_for_threshold = p_values[threshold_index,:]
        print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1}, exponent min {exponents_for_threshold.min():.3g} mean {exponents_for_threshold.mean():.3g} max {exponents_for_threshold.max():.3g}, true KSD min {ks_distances_for_threshold.min():.3g} mean {ks_distances_for_threshold.mean():.3g} max {ks_distances_for_threshold.max():.3g}, sample KSD min {sample_ksd_for_threshold.min():.3g}, mean {sample_ksd_for_threshold.mean():.3g}, max {sample_ksd_for_threshold.max():.3g}, p-value min {p_values_for_threshold.min():.3g} mean {p_values_for_threshold.mean():.3g} max {p_values_for_threshold.max():.3g}')
    return p_values

def compute_and_test_power_laws_serial_thresholds(counts:torch.Tensor, num_passes:int, num_tries_per_pass:int, p_value_sample_size:int, initial_min_exponent:float=1.0, initial_max_exponent:float=1e6, min_interval_length:int=3):
    num_values = counts.size(dim=-1)
    device = counts.device
    values = torch.arange(start=1, end=num_values+1, step=1, dtype=float_type, device=device)
    interval_mask = get_interval_mask(num_values=num_values, device=device, min_interval_length=min_interval_length)
    threshold_and_interval_have_sufficient_nonzero = get_threshold_and_interval_have_sufficient_nonzero(counts=counts, interval_mask=interval_mask, min_num_nonzero=min_interval_length)
    exponents, log_likelihoods = find_power_laws_serial_thresholds(values=values, counts=counts, interval_mask=interval_mask, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, threshold_and_interval_have_sufficient_nonzero=threshold_and_interval_have_sufficient_nonzero, initial_min_exponent=initial_min_exponent, initial_max_exponent=initial_max_exponent)
    ks_distances = get_ks_distance_serial_thresholds(values=values, counts=counts, interval_mask=interval_mask, exponents=exponents, threshold_and_interval_have_sufficient_nonzero=threshold_and_interval_have_sufficient_nonzero)
    p_values = sample_from_power_law_serial_thresholds(values=values, counts=counts, interval_mask=interval_mask, exponents=exponents, true_ks_distances=ks_distances, p_value_sample_size=p_value_sample_size, threshold_and_interval_have_sufficient_nonzero=threshold_and_interval_have_sufficient_nonzero)
    return exponents, log_likelihoods, ks_distances, p_values



def get_interval_endpoints(num_values:int, device:str, min_interval_length:int=3):
    num_values_plus_1 = num_values+1
    value_bounds = torch.triu_indices(row=num_values_plus_1, col=num_values_plus_1, offset=min_interval_length, dtype=int_type, device=device)
    values_start = value_bounds[0]
    values_end = value_bounds[1]
    return values_start, values_end

def get_one_power_law(value:torch.Tensor, count:torch.Tensor, num_passes:int, num_tries_per_pass:int, initial_min_exponent:float, initial_max_exponent:float, precision:float):
    float_type = count.dtype
    device = count.device
    mean_log = torch.sum( count * torch.log(value) )/torch.sum(count)
    value_row = value.unsqueeze(dim=0)
    min_exponent = initial_min_exponent
    max_exponent = initial_max_exponent
    for _ in range(num_passes):
        exponents = torch.linspace(start=min_exponent, end=max_exponent, steps=num_tries_per_pass, dtype=float_type, device=device)
        log_likelihoods = -torch.log(   torch.sum(  torch.pow( input=value_row, exponent=-exponents.unsqueeze(dim=-1) ), dim=-1  )   ) - exponents * mean_log
        max_ll_index = torch.argmax(log_likelihoods)
        min_exponent_index = max(max_ll_index-1, 0)
        max_exponent_index = min(max_ll_index+1, num_tries_per_pass-1)
        min_exponent = exponents[min_exponent_index].item()
        max_exponent = exponents[max_exponent_index].item()
        if abs(max_exponent - min_exponent) < precision:
            break
    return exponents[max_ll_index], log_likelihoods[max_ll_index]

def get_one_data_cdf(count:torch.Tensor):
    cdf = torch.cumsum(input=count, dim=0)
    cdf /= cdf[-1].clone()
    return cdf

def get_one_power_law_cdf(value:torch.Tensor, exponent:torch.Tensor):
    return get_one_data_cdf( count=torch.pow(input=value, exponent=-exponent) )

def get_one_ks_distance(cdf1:torch.Tensor, cdf2:torch.Tensor):
    return torch.max( torch.abs(cdf2 - cdf1) )

def get_one_p_value(power_law_cdf:torch.Tensor, true_ks_distance:float, num_samples:int, values_per_sample:int):
    float_type = power_law_cdf.dtype
    device = power_law_cdf.device
    power_law_cdf_col = power_law_cdf.unsqueeze(dim=-1)
    worse_count = 0
    for _ in range(num_samples):
        sample_cdf = torch.count_nonzero(   input=(  torch.rand( size=(1,values_per_sample), dtype=float_type, device=device ) < power_law_cdf_col  ), dim=-1   ).float()/values_per_sample
        sample_ks_distance = get_one_ks_distance(cdf1=sample_cdf, cdf2=power_law_cdf)
        worse_count += (sample_ks_distance >= true_ks_distance)
        # worse_count += (      get_one_ks_distance(     cdf1=get_one_data_cdf(    count=torch.count_nonzero(   input=(  torch.rand( size=(1,values_per_sample), dtype=float_type, device=device ) < power_law_cdf_col  ), dim=-1   ).float()    ), cdf2=power_law_cdf     ) >= true_ks_distance      )
    return worse_count/num_samples

def compute_and_test_power_laws_serial(counts:torch.Tensor, min_nonzero_counts:int, initial_min_exponent:float, initial_max_exponent:float, num_passes:int, num_tries_per_pass:int, precision:float, num_sample_ks_distances:int):
    float_type = counts.dtype
    device = counts.device
    num_thresholds, num_values = counts.size()
    values = torch.arange(start=1, end=num_values+1, step=1, dtype=float_type, device=device)
    interval_starts, interval_ends = get_interval_endpoints(num_values=num_values, device=device, min_interval_length=min_nonzero_counts)
    num_intervals = interval_starts.numel()
    exponents = torch.zeros( size=(num_thresholds, num_intervals), dtype=float_type, device=device )
    log_likelihoods = torch.zeros_like(exponents)
    ks_distances = torch.zeros_like(exponents)
    p_values = torch.zeros_like(exponents)
    for threshold_index in range(num_thresholds):
        for interval_index in range(num_intervals):
            interval_start = interval_starts[interval_index]
            interval_end = interval_ends[interval_index]
            value = values[interval_start:interval_end]
            count = counts[threshold_index,interval_start:interval_end]
            if torch.count_nonzero(count) >= min_nonzero_counts:
                exponent, log_likelihood = get_one_power_law(value=value, count=count, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, initial_min_exponent=initial_min_exponent, initial_max_exponent=initial_max_exponent, precision=precision)
                data_cdf = get_one_data_cdf(count=count)
                power_law_cdf = get_one_power_law_cdf(value=value, exponent=exponent)
                ks_distance = get_one_ks_distance(cdf1=power_law_cdf, cdf2=data_cdf)
                p_value = get_one_p_value( power_law_cdf=power_law_cdf, true_ks_distance=ks_distance, num_samples=num_sample_ks_distances , values_per_sample=torch.sum(count).int().item() )
                exponents[threshold_index,interval_index] = exponent
                log_likelihoods[threshold_index,interval_index] = log_likelihood
                ks_distances[threshold_index,interval_index] = ks_distance
                p_values[threshold_index,interval_index] = p_value
                # print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1}, interval {interval_index+1} [{interval_start}, {interval_end}], exponent {exponent:.3g}, log likelihood {log_likelihood:.3g}, KS distance {ks_distance:.3g}, p-value {p_value:.3g}')
        exponents_for_threshold = exponents[threshold_index,:]
        log_likelihoods_for_threshold = log_likelihoods[threshold_index,:]
        ks_distances_for_threshold = ks_distances[threshold_index,:]
        p_values_for_threshold = p_values[threshold_index,:]
        print(f'time {time.time()-code_start_time:.3f}, threshold {threshold_index+1}, exponent min {exponents_for_threshold.min():.3g} mean {exponents_for_threshold.mean():.3g} max {exponents_for_threshold.max():.3g}, log likelihood min {log_likelihoods_for_threshold.min():.3g} mean {log_likelihoods_for_threshold.mean():.3g} max {log_likelihoods_for_threshold.max():.3g}, KS distance min {ks_distances_for_threshold.min():.3g} mean {ks_distances_for_threshold.mean():.3g} max {ks_distances_for_threshold.max():.3g}, p-value min {p_values_for_threshold.min():.3g} mean {p_values_for_threshold.mean():.3g} max {p_values_for_threshold.max():.3g}')
    return exponents, log_likelihoods, ks_distances, p_values

# , max_samples_per_batch:int , max_samples_per_batch=max_samples_per_batch
def compute_and_save_power_law(output_directory:str, count_name:str, out_file_suffix:str, counts:torch.Tensor, min_nonzero_counts:int=3, initial_min_exponent:float=1.0, initial_max_exponent:float=1e6, num_passes:int=1000, num_tries_per_pass:int=1000, precision:float=1e-6, num_sample_ks_distances:int=1000):
    # exponents, log_likelihoods, ks_distances, p_values = try_power_laws_one_by_one(counts=counts, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, p_value_sample_size=p_value_sample_size)
    exponents, log_likelihoods, ks_distances, p_values = compute_and_test_power_laws_serial(counts=counts, min_nonzero_counts=min_nonzero_counts, initial_min_exponent=initial_min_exponent, initial_max_exponent=initial_max_exponent, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, precision=precision, num_sample_ks_distances=num_sample_ks_distances)
    for stat_to_store, stat_name in zip([exponents, log_likelihoods, ks_distances, p_values],['exponent', 'log_likelihood', 'ks_distance', 'p_value']):
        stat_file = os.path.join(output_directory, f'{stat_name}_{count_name}_{out_file_suffix}.pt')
        torch.save(obj=stat_to_store, f=stat_file)
        print(f'time {time.time()-code_start_time:.3f}, saved {stat_file}')
    return exponents

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

with torch.no_grad():
    code_start_time = time.time()
    float_type = torch.float
    int_type = torch.int
    device = torch.device('cuda')
    # device = torch.device('cpu')

    parser = argparse.ArgumentParser(description="Load and threshold unbinarized time series data. Make and save a Tensor counts such that counts[x] is the number of times x nodes flip in a single step.")
    parser.add_argument("-a", "--input_directory", type=str, default='E:\\Ising_model_results_daai', help="containing folder of fMRI_ts_binaries/ts_[6-digit subject ID]_[1|2]_[RL|LR].bin")
    parser.add_argument("-b", "--output_directory", type=str, default='E:\\Ising_model_results_daai', help="directory to which we write the mean state and mean state product PyTorch pickle files")
    parser.add_argument("-c", "--min_threshold", type=float, default=0, help="minimum threshold in std. dev.s")
    parser.add_argument("-d", "--max_threshold", type=float, default=5, help="minimum threshold in std. dev.s")
    parser.add_argument("-e", "--num_thresholds", type=int, default=100, help="number of thresholds to try")
    parser.add_argument("-f", "--training_subject_start", type=int, default=0, help="first training subject index")
    parser.add_argument("-g", "--training_subject_end", type=int, default=670, help="one past last training subject index")
    parser.add_argument("-o", "--min_nonzero_counts", type=int, default=3, help="minimum number of non-zero counts in an interval to which we try to fit a power law")
    parser.add_argument("-m", "--initial_min_exponent", type=float, default=0, help="minimum of range in which to search for the power law exponent")
    parser.add_argument("-n", "--initial_max_exponent", type=float, default=1e6, help="maximum of range in which to search for the power law exponent")
    parser.add_argument("-i", "--num_passes", type=int, default=10, help="number of iterations of maximum log likelihood search to do to find power law")
    parser.add_argument("-j", "--num_tries_per_pass", type=int, default=10, help="number of equally spaced exponents to try per iteration")
    parser.add_argument("-k", "--precision", type=float, default=2e-6, help="stop early if the ends of a search range are within this much of each other")
    parser.add_argument("-l", "--num_sample_ks_distances", type=int, default=1000, help="number of sample KS distances to compute to estimate P(KS distance for data sampled from power law >= KS distance for actual data)")
    args = parser.parse_args()
    print('getting arguments...')
    input_directory = args.input_directory
    print(f'input_directory={input_directory}')
    output_directory = args.output_directory
    print(f'output_directory={output_directory}')
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
    min_nonzero_counts = args.min_nonzero_counts
    print(f'min_nonzero_counts={min_nonzero_counts}')
    initial_min_exponent = args.initial_min_exponent
    print(f'initial_min_exponent={initial_min_exponent}')
    initial_max_exponent = args.initial_max_exponent
    print(f'initial_max_exponent={initial_max_exponent}')
    num_passes = args.num_passes
    print(f'num_passes={num_passes}')
    num_tries_per_pass = args.num_tries_per_pass
    print(f'num_tries_per_pass={num_tries_per_pass}')
    precision = args.precision
    print(f'precision={precision}')
    num_sample_ks_distances = args.num_sample_ks_distances
    print(f'num_sample_ks_distances={num_sample_ks_distances}')

    out_file_suffix = f'training_thresholds_{num_thresholds}_from_{min_threshold:.3g}_to_{max_threshold:.3g}_min_interval_{min_nonzero_counts}_exponents_from_{initial_min_exponent:.3g}_to_{initial_max_exponent:.3g}_passes_{num_passes}_tries_{num_tries_per_pass}_pvalsamples_{num_sample_ks_distances}'

    gap_duration_counts, duration_counts, size_counts, mean_size_for_duration = get_counts(output_directory=output_directory, out_file_suffix=out_file_suffix, training_subject_start=training_subject_start, training_subject_end=training_subject_end, num_thresholds=num_thresholds, min_threshold=min_threshold, max_threshold=max_threshold)
    
    # Try fitting power laws to things.
    # , max_samples_per_batch=max_samples_per_batch
    exponents = [ compute_and_save_power_law(output_directory=output_directory, count_name=count_name, out_file_suffix=out_file_suffix, counts=counts, min_nonzero_counts=min_nonzero_counts, initial_min_exponent=initial_min_exponent, initial_max_exponent=initial_max_exponent, num_passes=num_passes, num_tries_per_pass=num_tries_per_pass, precision=precision, num_sample_ks_distances=num_sample_ks_distances) for (counts, count_name) in zip([gap_duration_counts, duration_counts, size_counts, mean_size_for_duration],['gap_duration', 'avalanche_duration', 'avalanche_size']) ]
    duration_exponent = exponents[1]
    size_exponent = exponents[2]
    mean_size_for_duration_exponent = compute_and_save_positive_power_law(counts=mean_size_for_duration, output_directory=output_directory, count_name='mean_size_for_duration', out_file_suffix=out_file_suffix)

    # See Xu et al. 2021 Eq. 9.
    # Look at all possible choices of size interval and duration interval.
    # We do not try different intervals for mean size for duration.
    diff_from_critical_point = ( duration_exponent.unsqueeze(dim=-2)-1 )/( size_exponent.unsqueeze(dim=-1)-1 ) - mean_size_for_duration_exponent.unsqueeze(dim=-1).unsqueeze(dim=-2)
    diff_from_critical_point_file = os.path.join(output_directory, f'diff_from_critical_point_{out_file_suffix}.pt')
    torch.save(obj=diff_from_critical_point, f=diff_from_critical_point_file)
    print(f'time {time.time()-code_start_time:.3f}, saved {diff_from_critical_point_file}')

    print(f'time {time.time()-code_start_time:.3f}, done')