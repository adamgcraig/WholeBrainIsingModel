# functions we use in multiple Ising-model-related scripts, slightly more memory-efficient than isingmodel.py
# based on Sida Chen's code, in turn based on
# Panas, D., Amin, H., Maccione, A., Muthmann, O., van Rossum, M., Berdondini, L., & Hennig, M. H. (2015).
# Sloppiness in spontaneously active neuronal networks. Journal of Neuroscience, 35(22), 8480-8492.
# https://www.jneurosci.org/content/35/22/8480
 
import torch
import math
import time

float_type = torch.float
int_type = torch.int

# We start with some miscellaneous methods, mostly useful for computing various means, covariances, FCs, RMSEs, and Pearson correlations.

def num_nodes_to_num_pairs(num_nodes:int):
    return ( num_nodes*(num_nodes-1) )//2

# num_pairs = num_nodes*(num_nodes-1)//2
# We can use the quadratic formula to get back an expression for num_nodes in terms of num_pairs.
def num_pairs_to_num_nodes(num_pairs:int):
    return int(  ( math.sqrt(1 + 8*num_pairs) + 1 )/2  )

# In several places, we want to get the indices for the part of a num_nodes x num_nodes matrix above the diagonal.
# To make the code cleaner, we put the relevant code snippet in this function.
def get_triu_indices_for_products(num_nodes:int, device='cpu'):
    triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, dtype=int_type, device=device)
    return triu_indices[0], triu_indices[1]

def square_to_triu_pairs(square_pairs:torch.Tensor):
    num_nodes = square_pairs.size(dim=-1)
    triu_rows, triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=square_pairs.device)
    return square_pairs[:,:,triu_rows,triu_cols]

# triu_pairs gets filled in above and below the diagonal.
# Use diag_fill to specify a value to fill in along the diagonal.
def triu_to_square_pairs(triu_pairs:torch.Tensor, diag_fill:float=0.0):
    device = triu_pairs.device
    models_per_subject, num_subjects, num_pairs = triu_pairs.size()
    num_nodes = num_pairs_to_num_nodes(num_pairs=num_pairs)
    square_pairs = torch.full( size=(models_per_subject, num_subjects, num_nodes, num_nodes), fill_value=diag_fill, dtype=triu_pairs.dtype, device=device )
    triu_rows, triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=square_pairs.device)
    square_pairs[:,:,triu_rows,triu_cols] = triu_pairs
    square_pairs[:,:,triu_cols,triu_rows] = triu_pairs
    return square_pairs

def binarize_data_ts(data_ts:torch.Tensor, threshold:float=0):
    ts_std, ts_mean = torch.std_mean(data_ts, dim=-1, keepdim=True)
    return 2*( data_ts >= (ts_mean + threshold*ts_std) ).float() - 1

# time_series is time_series_per_subject x num_subjects x num_nodes x num_steps.
# We return the state means in a Tensor of size time_series_per_subject x num_subjects x num_nodes
# and the state product means in a Tensor of size time_series_per_subject x num_subjects x num_nodes x num_nodes.
def get_time_series_mean(time_series:torch.Tensor):
    num_steps = time_series.size(dim=-1)
    return time_series.mean(dim=-1), torch.matmul( time_series, time_series.transpose(dim0=-2, dim1=-1) )/num_steps

# time_series is time_series_per_subject x num_subjects x num_nodes x num_steps.
# We return the state means in a Tensor of size time_series_per_subject x num_subjects x num_nodes
# and the state product means in a Tensor of size time_series_per_subject x num_subjects x num_nodes x num_nodes.
# Use this version if get_time_series_mean() takes up too much memory.
def get_time_series_mean_step_by_step(time_series:torch.Tensor):
    ts_per_subject, num_subjects, num_nodes, num_steps = time_series.size()
    state_product_sum = torch.zeros( size=(ts_per_subject, num_subjects, num_nodes, num_nodes), dtype=time_series.dtype, device=time_series.device )
    for step in range(num_steps):
        state = time_series[:,:,:,step]
        state_product_sum += state[:,:,:,None] * state[:,:,None,:]
    return torch.mean(time_series, dim=-1), state_product_sum/num_steps

# center the uncentered covariance
def get_cov(state_mean:torch.Tensor, state_product_mean:torch.Tensor):
    return state_product_mean - state_mean.unsqueeze(dim=-1) * state_mean.unsqueeze(dim=-2)

def get_std(state_mean:torch.Tensor, state_product_mean:torch.Tensor):
    # The standard deviation is sqrt( mean(s^2) - mean(s)^2 ).
    # We can get mean(s^2) from the diagonal of  state_product_mean.
    return torch.sqrt( torch.diagonal(state_product_mean, offset=0, dim1=-2, dim2=-1) - state_mean.square() )

def get_std_binary(state_mean:torch.Tensor):
    # The standard deviation is sqrt( mean(s^2) - mean(s)^2 ).
    # If the state is always either -1 or +1, then s^2 is always +1, so the mean of s^2 is 1.
    return torch.sqrt( 1 - state_mean.square() )

def get_fc(state_mean:torch.Tensor, state_product_mean:torch.Tensor, epsilon:float=0.0):
    # For the functional connectivity (FC), we use the Pearson correlations between pairs of nodes.
    # The correlation between nodes i and j is ( mean(s[i]*s[j]) - mean(s[i])*mean(s[j]) )/( std.dev.(s[i])*std.dev(s[j]) )
    s_std = get_std(state_mean=state_mean, state_product_mean=state_product_mean)
    return ( get_cov(state_mean=state_mean, state_product_mean=state_product_mean) + epsilon )/( s_std.unsqueeze(dim=-1) * s_std.unsqueeze(dim=-2) + epsilon)

def get_fc_binary(state_mean:torch.Tensor, state_product_mean:torch.Tensor, epsilon:float=0.0):
    # For the functional connectivity (FC), we use the Pearson correlations between pairs of nodes.
    # The correlation between nodes i and j is ( mean(s[i]*s[j]) - mean(s[i])*mean(s[j]) )/( std.dev.(s[i])*std.dev(s[j]) )
    s_std = get_std_binary(state_mean=state_mean)
    return ( get_cov(state_mean=state_mean, state_product_mean=state_product_mean) + epsilon )/( s_std.unsqueeze(dim=-1) * s_std.unsqueeze(dim=-2) + epsilon)

def get_pairwise_correlation(mat1:torch.Tensor, mat2:torch.Tensor, epsilon:float=0.0, dim:int=-1):
    # If the number of dimensions is unequal, assume that one of them is missing the 0/model dimension.
    # This is true when we compare data observables to model simulation observables.
    if len( mat1.size() ) < len( mat2.size() ):
        mat1 = mat1.unsqueeze(dim=0)
    elif len( mat2.size() ) < len( mat1.size() ):
        mat2 = mat2.unsqueeze(dim=0)
    std_1, mean_1 = torch.std_mean(mat1, dim=dim)
    std_2, mean_2 = torch.std_mean(mat2, dim=dim)
    return ( torch.mean(mat1 * mat2, dim=dim) - mean_1 * mean_2 + epsilon )/(std_1 * std_2 + epsilon)

def get_pairwise_rmse(mat1:torch.Tensor, mat2:torch.Tensor, dim:int=-1):
    # If the number of dimensions is unequal, assume that one of them is missing the 0/model dimension.
    # This is true when we compare data observables to model simulation observables.
    if len( mat1.size() ) < len( mat2.size() ):
        mat1 = mat1.unsqueeze(dim=0)
    elif len( mat2.size() ) < len( mat1.size() ):
        mat2 = mat2.unsqueeze(dim=0)
    return torch.sqrt(  torch.mean( (mat1 - mat2).square(), dim=dim )  )

def get_pairwise_correlation_2d(mat1:torch.Tensor, mat2:torch.Tensor, epsilon:float=0.0):
    return get_pairwise_correlation( mat1=mat1, mat2=mat2, epsilon=epsilon, dim=(-2,-1) )

def get_pairwise_rmse_2d(mat1:torch.Tensor, mat2:torch.Tensor):
    return get_pairwise_rmse( mat1=mat1, mat2=mat2, dim=(-2,-1) )

def get_pairwise_correlation_ut(mat1:torch.Tensor, mat2:torch.Tensor, epsilon:float=0.0):
    # If the number of dimensions is unequal, assume that one of them is missing the 0/model dimension.
    # This is true when we compare data observables to model simulation observables.
    if len( mat1.size() ) < len( mat2.size() ):
        mat1 = mat1.unsqueeze(dim=0)
    elif len( mat2.size() ) < len( mat1.size() ):
        mat2 = mat2.unsqueeze(dim=0)
    triu_rows, triu_cols = get_triu_indices_for_products( num_nodes=mat1.size(dim=-1), device=mat1.device )
    return get_pairwise_correlation( mat1=mat1[:,:,triu_rows,triu_cols], mat2=mat2[:,:,triu_rows,triu_cols], epsilon=epsilon, dim=-1 )

def get_pairwise_rmse_ut(mat1:torch.Tensor, mat2:torch.Tensor):
    # If the number of dimensions is unequal, assume that one of them is missing the 0/model dimension.
    # This is true when we compare data observables to model simulation observables.
    if len( mat1.size() ) < len( mat2.size() ):
        mat1 = mat1.unsqueeze(dim=0)
    elif len( mat2.size() ) < len( mat1.size() ):
        mat2 = mat2.unsqueeze(dim=0)
    triu_rows, triu_cols = get_triu_indices_for_products( num_nodes=mat1.size(dim=-1), device=mat1.device )
    return get_pairwise_rmse( mat1=mat1[:,:,triu_rows,triu_cols], mat2=mat2[:,:,triu_rows,triu_cols], dim=-1 )

# Use these methods to create the beta, h, J, and s Tensors we need in order to initialize the IsingModel module.

# Make a beta Tensor with the same range of values for each fitting target.
def get_linspace_beta(models_per_subject:int, num_subjects:int, dtype=float_type, device='cpu', min_beta:float=10e-10, max_beta:float=1.0):
    return torch.linspace(start=min_beta, end=max_beta, steps=models_per_subject, dtype=dtype, device=device).unsqueeze(dim=-1).repeat( repeats=(1,num_subjects) )

def get_linspace_beta_like(input:torch.Tensor, min_beta:float=10e-10, max_beta:float=1.0):
    models_per_subject, num_subjects = input.size()
    return get_linspace_beta(models_per_subject=models_per_subject, num_subjects=num_subjects, dtype=input.dtype, device=input.device, min_beta=min_beta, max_beta=max_beta)

# Replace the diagonal of each 2D Tensor in the stack with 0.
def zero_diag(square_mat:torch.Tensor):
    return square_mat - torch.diag_embed( input=torch.diagonal(input=square_mat, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )

# Create an initial guess for J by zeroing the diagonal of each target uncentered covariance and replicating it models_per_subject times. 
def get_J_from_means(models_per_subject:int, mean_state_product:torch.Tensor):
    return zero_diag(mean_state_product).unsqueeze(dim=0).repeat( repeats=(models_per_subject, 1, 1, 1) )

# Creat an initial guess for h by replicating each target mean models_per_subject times.
def get_h_from_means(models_per_subject:int, mean_state:torch.Tensor):
    return mean_state.unsqueeze(dim=0).repeat( repeats=(models_per_subject, 1, 1) )

# Create random initial states with each node having . 
def get_random_state(models_per_subject:int, num_subjects:int, num_nodes:int, dtype=float_type, device='cpu', prob_up:float=0.5):
    return 2.0*( torch.rand( size=(models_per_subject, num_subjects, num_nodes), dtype=dtype, device=device ) <= prob_up ).float() - 1.0

# Create a Tensor with the same shape, data type, and device as input but filled with randomly selected -1 and +1 values.
def get_random_state_like(input:torch.Tensor, prob_up:float=0.5):
    return 2 * ( torch.rand_like(input=input) <= prob_up ).float() - 1

# Create a batched state vector where all nodes are in state -1.
# In the pooled group HCP data parcellated with the Glasser Atlas and binarized at all thresholds >= 0, this is the most common state.
# As such, it is a convenient starting point for the simulation.
def get_neg_state(models_per_subject:int, num_subjects:int, num_nodes:int, dtype=float_type, device='cpu'):
    return torch.full( size=(models_per_subject, num_subjects, num_nodes), fill_value=-1, dtype=dtype, device=device )

# Create a Tensor with the same shape, data type, and device as input but filled with -1.
def get_neg_state_like(input:torch.Tensor):
    return torch.full_like(input, fill_value=-1)

# General notes on the IsingModel module
# Because the Metropolis simulation only works right when we compute the entropy change of flipping a single node at a time,
# we cannot use GPU arithmetic to operate on all nodes at once.
# Instead, we amortize the cost of performing a simulation by operating on a batch of Ising models, fitting and simulating each one independently in parallel.
# We have two batch dimensions: models_per_subject and num_subjects.
# The first, models_per_subject, allows us to perform multiple fittings to the same target data.
# The second, num_subjects, allows us to fit to multiple targets.
# We use the models_per_subject in two ways:
# First, we simulate the same initial guess model at multiple temperatures when optimizing the temperature.
# Second, once we arrive at the optimal temperature, we independently optimize the parameters of each instance using Boltzmann learning.
# This allows us to estimate the reliability of the Boltzmann learning process.
# To make it easier to work with the model, we package it into a PyTorch Module.
# However, it does not work with PyTorch gradient tracking, since we modify the state in place.
# When using this class, you can place the code in a torch.no_grad() block to reduce memory usage.
# The constructor is trivial, because we provide functions for constructing the four persistent entities:
# variable name, alternate name, dimensions
# beta, inverse temperature, models_per_subject x num_subjects
# J, coupling, models_per_subject x num_subjects x num_nodes x num_nodes
# h, external field, models_per_subject x num_subjects x num_nodes
# s, state, models_per_subject x num_subjects x num_nodes
# The methods use a balanced Metropolis simulation with fixed order to sample states, meaning that we iterate over the nodes, giving each a chance to update, before recording the next state.
class IsingModel(torch.nn.Module):

    def __init__(self, beta:torch.Tensor, J:torch.Tensor, h:torch.Tensor, s:torch.Tensor):
        super(IsingModel, self).__init__()
        self.beta = beta
        self.J = J
        self.h = h
        self.s = s
    
    def forward(self, time_series:torch.Tensor):
        return self.get_entropy_of_ts(time_series)
    
    # time_series is of size models_per_subject x num_subjects x num_nodes x num_time_points.
    # We return a stacked matrix of size models_per_subject x num_subjects (optionally x 1) x num_time_points
    # with the entropy of each time point
    def get_entropy_of_ts(self, time_series:torch.Tensor, keepdim:bool=False):
        return torch.sum(  time_series * ( self.h.unsqueeze(dim=-1) + 0.5 * torch.matmul(self.J, time_series) ), dim=-2, keepdim=keepdim  )
    
    # Run a Metropolis simulation.
    # Record the full time series of states.
    # Use this if you need something other than just the means and uncentered covariances.
    # For simplicity, this version assumes that you have not pre-allocated any of the temporary variables that it needs.
    def simulate_and_record_time_series(self, num_steps:int):
        models_per_subject, num_subjects, num_nodes = self.s.size()
        time_series = torch.zeros( size=(models_per_subject, num_subjects, num_nodes, num_steps), dtype=self.s.dtype, device=self.s.device )
        rand_choice = torch.zeros_like(self.s)
        hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
        Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
        for step in range(num_steps):
            torch.rand( size=rand_choice.size(), out=rand_choice )
            rand_choice.log_()
            for node in range(num_nodes):
                self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
            time_series[:,:,:,step] = self.s
        return time_series
    
    # Run a Metropolis simulation.
    # Record just the means and uncentered covariances.
    # This is more memory efficient than recording the full time series,
    # especially if you are running multiple simulations in sequence and have pre-allocated memory for the outputs and some other temporary variables.
    def simulate_and_record_means(self, num_steps:int=1200, mean_state:torch.Tensor=None, mean_state_product:torch.Tensor=None, rand_choice:torch.Tensor=None, s_product:torch.Tensor=None, hbeta:torch.Tensor=None, Jbeta:torch.Tensor=None):
        # If we do not have pre-allocated Tensors, allocate them here.
        if type(mean_state) == type(None):
            mean_state = torch.zeros_like(self.h)
        else:
            # If we are reusing memory, reset the sum.
            mean_state.zero_()
        if type(mean_state_product) == type(None):
            mean_state_product = torch.zeros_like(self.J)
        else:
            # If we are reusing memory, reset the sum.
            mean_state_product.zero_()
        if type(rand_choice) == type(None):
            rand_choice = torch.zeros_like(self.h)
        if type(s_product) == type(None):
            s_product = torch.zeros_like(self.J)
        if type(hbeta) == type(None):
            hbeta = self.h.clone()
        else:
            hbeta.copy_(self.h)
        if type(Jbeta) == type(None):
            Jbeta = self.J.clone()
        else:
            Jbeta.copy_(self.J)
        # Get the number of nodes.
        num_nodes = self.s.size(dim=-1)
        # Pre-multiply h and J by beta.
        # Pre-multiply in -2, since we would otherwise need to divide log(rand_choice) by -2 each time we generated it.
        beta_3d = (-2 * self.beta).unsqueeze(dim=-1)
        hbeta *= beta_3d
        Jbeta *= beta_3d.unsqueeze(dim=-1)
        # Run the requested number of steps.
        for _ in range(num_steps):
            # For each node, pre-allocate a random number that will help determine whether it flips.
            # To make better use of GPU parallelization,
            # instead of taking exp() of each individual entropy change to get the flip probability one at a time,
            # we take log() of the randomly generated numbers all at once.
            torch.rand( size=rand_choice.size(), out=rand_choice )
            rand_choice.log_()
            # Iterate over the nodes to decide which ones to flip.
            for node in range(num_nodes):
                self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
            mean_state += self.s# B x N x 1
            torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            mean_state_product += s_product
        mean_state /= num_steps
        mean_state_product /= num_steps
        return mean_state, mean_state_product

    # Perform an iterative grid-based search for the beta value that minimizes the RMSE between the data and model covariance matrices.
    # We assume that this step occurs between initialization and parameter fitting, so that all models for a subject are copies of the initial guess model.
    # We use these copies to test multiple beta choices in parallel, so we need models_per_subject to be > 1.
    # In general, the more models_per_subject, the faster and more reliably we will find the optimum beta value.
    # If you are using the maximum models_per_subject that memory limitations allow and still get poor convergence, try using a smaller max_beta.
    def optimize_beta(self, target_cov:torch.Tensor, num_updates:int, num_steps:int, min_beta:float=10e-10, max_beta:float=1.0, epsilon:float=10e-10, verbose:bool=False):
        opt_start_time = time.time()
        target_cov = target_cov.unsqueeze(dim=0)
        # Initialize all beta ranges to [epsilon, 1], but make a separate copy for every target so that we can arrive at different optimal values.
        models_per_subject, num_subjects, _ = self.s.size()
        dtype = self.s.dtype
        device = self.s.device
        # Pre-allocate variables that we will re-use on every iteration of the loop.
        # We need this arange() Tensor for indexing later.
        subject_index = torch.arange(start=0, end=num_subjects, step=1, dtype=int_type, device=device)
        beta_steps = torch.arange(start=0, end=models_per_subject, step=1, dtype=dtype, device=device)
        beta_start = torch.full( size=(num_subjects,), fill_value=min_beta, dtype=dtype, device=device )
        beta_end = torch.full_like(beta_start, fill_value=max_beta)
        mean_state = torch.zeros_like(self.h)
        mean_state_product = torch.zeros_like(self.J)
        rand_choice = torch.zeros_like(self.h)
        s_product = torch.zeros_like(self.J)
        hbeta = torch.zeros_like(self.h)
        Jbeta = torch.zeros_like(self.J)
        # print( 'initialized beta with size ', beta.size() )
        num_updates_completed = 0
        for iteration in range(num_updates):
            self.beta = beta_start.unsqueeze(dim=0) + ( (beta_end - beta_start)/(models_per_subject-1) ).unsqueeze(dim=0) * beta_steps.unsqueeze(dim=1)
            # Run the Metropolis simulation, and record mean_state and mean_state_product.
            self.simulate_and_record_means(num_steps=num_steps, mean_state=mean_state, mean_state_product=mean_state_product, rand_choice=rand_choice, s_product=s_product, hbeta=hbeta, Jbeta=Jbeta)
            # Compute the centered covariance.
            sim_state_cov = get_cov(state_mean=mean_state, state_product_mean=mean_state_product)
            # Take the RMSE over all node pairs for each model.
            cov_rmse = get_pairwise_rmse_2d(mat1=target_cov, mat2=sim_state_cov)
            # Find the minimum RMSE over all models for each target.
            _, best_beta_index = torch.min(cov_rmse, dim=0)
            best_beta = self.beta[best_beta_index,subject_index]
            range_width = self.beta[-1,:] - self.beta[0,:]
            half_range_width = range_width/2
            # If the best value is epsilon, make the new range from epsilon to the next highest value.
            # Since epsilon is the first value, this will scale the range by 1/models_per_subject.
            best_is_epsilon = best_beta <= epsilon
            beta_start[best_is_epsilon] = epsilon
            beta_end[best_is_epsilon] = self.beta[1,best_is_epsilon]
            # If the best value is the lowest value and not epsilon,
            # keep the range width the same and shift the range so that what was previously the lowest value is in the middle.
            # Truncate it if this makes it extend below 0.
            best_is_least = best_beta_index == 0
            best_is_least_and_gt_epsilon = torch.logical_and( best_is_least, torch.logical_not(best_is_epsilon) )
            best_is_least_best_beta = best_beta[best_is_least_and_gt_epsilon]
            best_is_least_half_range_width = half_range_width[best_is_least_and_gt_epsilon]
            beta_start[best_is_least_and_gt_epsilon] = torch.clamp_min(best_is_least_best_beta - best_is_least_half_range_width, min=epsilon)
            beta_end[best_is_least_and_gt_epsilon] = best_is_least_best_beta + best_is_least_half_range_width
            # If the best value is the highest value,
            # keep the range width the same and shift the range so that what was previously the highest value is in the middle.
            # We do not impose any upper limit on beta.
            best_is_most = best_beta_index == (models_per_subject-1)
            best_is_most_best_beta = best_beta[best_is_most]
            best_is_most_half_range_width = half_range_width[best_is_most]
            beta_start[best_is_most] = best_is_most_best_beta - best_is_most_half_range_width
            beta_end[best_is_most] = best_is_most_best_beta + best_is_most_half_range_width
            # If the best value is neither the highest nor the lowest,
            # make the new range between the values immediately above and below it. 
            best_is_middle = torch.logical_and( torch.logical_not(best_is_least), torch.logical_not(best_is_most) )
            best_is_middle_best_beta_index = best_beta_index[best_is_middle]
            best_is_middle_subject_index = subject_index[best_is_middle]
            beta_start[best_is_middle] = self.beta[best_is_middle_best_beta_index-1, best_is_middle_subject_index]
            beta_end[best_is_middle] = self.beta[best_is_middle_best_beta_index+1, best_is_middle_subject_index]
            # print(f'interval calculation time {time.time()-interval_calc_start_time:.3f}')
            if verbose:
                print(f'time {time.time()-opt_start_time:.3f}, iteration {iteration+1}, best beta min {best_beta.min():.3g}, mean {best_beta.mean():.3g}, max {best_beta.max():.3g}, beta range min {range_width.min():.3g}, mean {range_width.mean():.3g}, max {range_width.max():.3g}, cov RMSE min {cov_rmse.min():.3g}, mean {cov_rmse.mean():.3g}, max {cov_rmse.max():.3g}')
            num_updates_completed += 1
            # Stop if we have homed in on the best possible beta to within machine precision.
            if range_width.max() == 0:
                if verbose:
                    print('betas have all converged to a single value per subject')
                break
        return num_updates_completed
    
    # Perform Boltzmann learning with Metropolis simulation.
    # This is an iterative process wherein we run the model to get sim_mean_state and sim_mean_state_product,
    # and then update the model parameters with the rules
    # h <- h + learning_rate*(target_mean_state - sim_mean_state)
    # J <- J + learning_rate*(target_mean_state_product - sim_mean_state_product)
    def fit_by_simulation(self, target_mean_state:torch.Tensor, target_mean_state_product:torch.Tensor, num_updates:int=100000, steps_per_update:int=1200, learning_rate:float=0.01, verbose:bool=False):
        opt_start_time = time.time()
        # Give the target an additional dimension so that we can broadcast it to all replica models of the same target.
        target_mean_state = target_mean_state.unsqueeze(dim=0)
        target_mean_state_product = target_mean_state_product.unsqueeze(dim=0)
        # Pre-allocate variables that we will re-use on each iteration.
        mean_state = torch.zeros_like(self.h)
        mean_state_product = torch.zeros_like(self.J)
        rand_choice = torch.zeros_like(self.h)
        s_product = torch.zeros_like(self.J)
        hbeta = torch.zeros_like(self.h)
        Jbeta = torch.zeros_like(self.J)
        neg_lr = -learning_rate
        for update in range(num_updates):
            self.simulate_and_record_means(num_steps=steps_per_update, mean_state=mean_state, mean_state_product=mean_state_product, rand_choice=rand_choice, s_product=s_product, hbeta=hbeta, Jbeta=Jbeta)
            # Since the product of any state with itself is +1, the diagonals of both the target and sim product means are 1.
            # Consequently, the difference between the diagonals is 0, leading to no change in J, which is initialized to 0.
            # This is intentional, since no node should have a coupling to itsef.
            # We split up h <- h+lr*(data_mean - sim_mean) into h+=(lr*data_mean) and h-=(lr*sim_mean)
            # and J <- J+lr*(data_mean_product - sim_mean_product) into J+=(lr*data_mean_product) and J-=(lr*sim_mean_product)
            # so that we can do these steps in-place without allocating memory for the differences.
            self.h.add_(other=target_mean_state, alpha=learning_rate)
            self.h.add_(other=mean_state, alpha=neg_lr)
            self.J.add_(other=target_mean_state_product, alpha=learning_rate)
            self.J.add_(other=mean_state_product, alpha=neg_lr)
            if verbose:
                rmse_state_mean = get_pairwise_rmse(mat1=target_mean_state, mat2=mean_state)
                rmse_state_product_mean = get_pairwise_rmse_2d(mat1=target_mean_state_product, mat2=mean_state_product)
                print(f'time {time.time()-opt_start_time:.3f}, update {update+1}, state mean RMSE  min {rmse_state_mean.min():.3g}, mean {rmse_state_mean.mean():.3g}, max {rmse_state_mean.max():.3g}, state product mean RMSE min {rmse_state_product_mean.min():.3g}, mean {rmse_state_product_mean.mean():.3g}, max {rmse_state_product_mean.max():.3g}')    