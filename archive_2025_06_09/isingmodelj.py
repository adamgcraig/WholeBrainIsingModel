# functions we use in multiple Ising-model-related scripts
# based on Sida Chen's code, in turn based on
# Panas, D., Amin, H., Maccione, A., Muthmann, O., van Rossum, M., Berdondini, L., & Hennig, M. H. (2015).
# Sloppiness in spontaneously active neuronal networks. Journal of Neuroscience, 35(22), 8480-8492.
# https://www.jneurosci.org/content/35/22/8480
# This is a simplified Ising model where we omit h and premultiply beta into J.
# As such, we only need to store the J matrix.
# We always store J with dimensions models_per_subject x num_subjects x num_nodes x num_nodes,
# but we include methods to export the upper triangular part.

import torch
import math

float_type = torch.float
int_type = torch.int

def num_nodes_to_num_pairs(num_nodes:int):
    return ( num_nodes*(num_nodes-1) )//2

# num_pairs = num_nodes*(num_nodes-1)//2
# We can use the quadratic formula to get back an expression for num_nodes in terms of num_pairs.
def num_pairs_to_num_nodes(num_pairs:int):
    return int(  ( math.sqrt(1 + 8*num_pairs) + 1 )/2  )

def get_random_J(models_per_subject:int, num_subjects:int, num_nodes:int, dtype=float_type, device='cpu'):
    return torch.randn( size=(models_per_subject, num_subjects, num_nodes, num_nodes), dtype=dtype, device=device )/math.sqrt(num_nodes)

# Create random initial states for batch_size independent Ising models, each with num_nodes nodes. 
def get_random_state(models_per_subject:int, num_subjects:int, num_nodes:int, dtype=float_type, device='cpu'):
    return 2.0*torch.randint( low=0, high=2, size=(models_per_subject, num_subjects, num_nodes), dtype=dtype, device=device ) - 1.0

# Create a Tensor with the same shape, data type, and device as input but filled with randomly selected -1 and +1 values.
def get_random_state_like(input:torch.Tensor):
    return 2 * torch.randint_like(input=input, high=2) - 1

# Create a batched state vector where all nodes are in state -1. 
def get_neg_state(models_per_subject:int, num_subjects:int, num_nodes:int, dtype=float_type, device='cpu'):
    return torch.full( size=(models_per_subject, num_subjects, num_nodes), fill_value=-1, dtype=dtype, device=device )

def get_neg_state_like(input:torch.Tensor):
    return torch.full_like(input, fill_value=-1)

# Compute the entropy of an Ising model with external fields h, couplings J, and state s.
# s and h can have an arbitrary number of leading batch dimensions but must have the same number of dimensions and same size in the last dimension.
# J should have the number of dimensions in s and h + 1.
# Let s', h', and J' be slices of s, h, and J respectively representing an individual model state, external fields, and couplings.
# s' and h' have num_nodes elements, and J' is num_nodes x num_nodes with J'[i,j] representing the coupling from node j to node i.
# We assume J[i,i] = 0.
def get_entropy(s:torch.Tensor, J:torch.Tensor):
    return 0.5 * torch.matmul(  s.unsqueeze(dim=-2), torch.matmul( J, s.unsqueeze(dim=-1) )  )

# Compute the change in entropy of an Ising model with external fields h, couplings J, and state s if we flip node node.
# s and h can have an arbitrary number of leading batch dimensions but must have the same number of dimensions and same size in the last dimension.
# J should have the number of dimensions in s and h + 1.
# Let s', h', and J' be slices of s, h, and J respectively representing an individual model state, external fields, and couplings.
# s' and h' have num_nodes elements, and J' is num_nodes x num_nodes with J'[i,j] representing the coupling from node j to node i.
# We assume J[i,i] = 0.
# For illustrative purposes, consider two unbatched states, s_before and s_after,
# representing the state of the unbatched model (h,J) before and after we flip node i.
# For j != i, s_after[j] = s_before[j], so s_after[j] - s_before[j] = 0.
# s_after[i] = -1*s_before[i], so s_after[i] - s_before[i] = -1*s_before[i] - s_before[i] = -2*s_before[i].
# Consequently, torch.sum( h*(s_after-s_before) ) = -2*h[i]*s_before[i].
# For j != i and k != i, s_after[j] = s_before[j], and s_after[k] = s_before[k],
# so s_after[j]*s_after[k] = s_before[j]*s_before[k],
# and s_after[j]*s_after[k] - s_before[j]*s_before[k] = 0.
# For j != i, s_after[j] = s_before[j], and s_after[i] = -1*s_before[i],
# so s_after[j]*s_after[i] = -1*s_before[j]*s_before[i],
# and s_after[j]*s_after[i] - s_before[j]*s_before[i] = -2*s_before[j]*s_before[k].
# s_after[i] = -1*s_before[i],
# so s_after[i]*s_after[i] = -1*(-1)*s_before[i]*s_before[i] = s_before[i]*s_before[i],
# and s_after[i]*s_after[i] - s_before[i]*s_before[i] = 0.
# Consequently, torch.sum(  J * ( s_after[:,None] * s_after[None,:] - s_before[:,None] * s_before[None,:] ), dim=(-1,-2)  )
# = torch.sum(-2 * J[i,:] * s_before * s_before[i]) (relying on J[i,i] = 0)
# S_before = torch.sum(h*s_before, dim=-1) + 0.5 * torch.sum( J * s_before[:,None] * s_before[None,:], dim=(-1,-2) )
# S_after = torch.sum(h*s_after, dim=-1) + 0.5 * torch.sum( J * s_after[:,None] * s_after[None,:], dim=(-1,-2) )
# S_diff = S_after - S_before
#        = torch.sum(h*s_after, dim=-1) + 0.5 * torch.sum( J * s_after[:,None] * s_after[None,:], dim=(-1,-2) )
#        - torch.sum(h*s_before, dim=-1) + 0.5 * torch.sum( J * s_before[:,None] * s_before[None,:], dim=(-1,-2) )
#        = torch.sum(h*(s_after-s_before), dim=-1) + 0.5 * torch.sum( J * (s_after[:,None]*s_after[None,:] - s_before[:,None]*s_before[None,:]), dim=(-1,-2) )
#        = -2*h[i]*s_before[i] + 0.5 * torch.sum(-2 * J[i,:] * s_before * s_before[i])
#        = -2*( h[i]*s_before[i] + 0.5 * torch.sum(J[i,:] * s_before * s_before[i]) )
#        = -2*s_before[i]*( sh[i] + 0.5 * torch.sum(J[i,:] * s_before) )
def get_entropy_change(s:torch.Tensor, J:torch.Tensor, node:int):
    return -1 * s.index_select(dim=-1, index=node) * torch.matmul( J.index_select(dim=-2, index=node).unsqueeze(dim=-2), s.unsqueeze(dim=-1) )

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
def triu_to_square_pairs(triu_pairs:torch.Tensor, diag_fill:torch.float=0.0):
    device = triu_pairs.device
    models_per_subject, num_subjects, num_pairs = triu_pairs.size()
    num_nodes = num_pairs_to_num_nodes(num_pairs=num_pairs)
    square_pairs = torch.full( size=(models_per_subject, num_subjects, num_nodes, num_nodes), fill_value=diag_fill, dtype=triu_pairs.dtype, device=device )
    triu_rows, triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=square_pairs.device)
    square_pairs[:,:,triu_rows,triu_cols] = triu_pairs
    square_pairs[:,:,triu_cols,triu_rows] = triu_pairs
    return square_pairs

def binarize_data_ts(data_ts:torch.Tensor):
    # First, threshold each region time series so that
    # anything below the median maps to -1,
    # anything above the median maps to +1,
    # anything exactly equal to the median maps to 0.
    step_dim = -1
    sign_threshold = torch.median(input=data_ts, dim=step_dim, keepdim=True).values
    binary_ts = torch.sign(data_ts - sign_threshold)
    # Next, for each individual region time series, fill in any 0s in such a way as to make the counts of -1 and +1 as even as possible.
    num_neg = torch.count_nonzero(binary_ts == -1, dim=step_dim)
    num_pos = torch.count_nonzero(binary_ts == 1, dim=step_dim)
    is_zero = binary_ts == 0
    num_zero = torch.count_nonzero(is_zero, dim=step_dim)
    ts_per_subject, num_subjects, num_nodes, _ = binary_ts.size()
    for ts_index in range(ts_per_subject):
        for subject_index in range(num_subjects):
            for node_index in range(num_nodes):
                num_neg_here = num_neg[ts_index, subject_index, node_index]
                num_pos_here = num_pos[ts_index, subject_index, node_index]
                is_zero_here = is_zero[ts_index, subject_index, node_index, :]
                num_zero_here = num_zero[ts_index, subject_index, node_index]
                zero_fills = torch.zeros( size=(num_zero_here,), dtype=binary_ts.dtype, device=binary_ts.device )
                for zero_index in range(num_zero_here):
                    if num_pos_here < num_neg_here:
                        zero_fills[zero_index] = 1
                        num_pos_here += 1
                    else:
                        zero_fills[zero_index] = -1
                        num_neg_here += 1
                binary_ts[ts_index, subject_index, node_index, is_zero_here] = zero_fills
    return binary_ts

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

def get_fc_binary(state_mean:torch.Tensor, state_product_mean:torch.Tensor, epsilon:torch.float=10e-10):
    # For the functional connectivity (FC), we use the Pearson correlations between pairs of nodes.
    # The correlation between nodes i and j is ( mean(s[i]*s[j]) - mean(s[i])*mean(s[j]) )/( std.dev.(s[i])*std.dev(s[j]) )
    # The standard deviation is sqrt( mean(s^2) - mean(s)^2 ).
    # If the state is always either -1 or +1, then s^2 is always +1, so the mean of s^2 is 1.
    s_std = torch.sqrt( 1 - state_mean.square() )
    return ( state_product_mean - state_mean.unsqueeze(dim=-1) * state_mean.unsqueeze(dim=-2) )/( s_std.unsqueeze(dim=-1) * s_std.unsqueeze(dim=-2) + epsilon)

def get_pairwise_correlation(mat1:torch.Tensor, mat2:torch.Tensor, epsilon:torch.float=10e-10, dim:int=-1):
    std_1, mean_1 = torch.std_mean(mat1, dim=dim)
    std_2, mean_2 = torch.std_mean(mat2, dim=dim)
    return ( torch.mean(mat1 * mat2, dim=dim) - mean_1 * mean_2 + epsilon )/(std_1 * std_2 + epsilon)

def get_pairwise_rmse(mat1:torch.Tensor, mat2:torch.Tensor, dim:int=-1):
    return torch.sqrt(  torch.mean( (mat1 - mat2).square(), dim=dim )  )

def get_pairwise_correlation_2d(mat1:torch.Tensor, mat2:torch.Tensor, epsilon:torch.float=10e-10):
    return get_pairwise_correlation( mat1=mat1, mat2=mat2, epsilon=epsilon, dim=(-2,-1) )

def get_pairwise_rmse_2d(mat1:torch.Tensor, mat2:torch.Tensor):
    return get_pairwise_rmse( mat1=mat1, mat2=mat2, dim=(-2,-1) )

# We can simulate the evolution of the system over time in one of two ways.
# 1. The standard Metropolis algorithm, wherein, at each time step, we randomly select one node and decide whether or not to flip it.
# 2. The balanced Metropolis algorithm, wherein, at each time step, we iterate over all nodes and decide whether to flip each one in turn.
# In our version of the balanced algorithm, we randomize the order of the nodes, but this is optional.
# In both cases, we need to consider one node at a time, because each flip changes the probabilities of subsequent flips.
# In both cases, the probability of a given flip is 1 if the entropy change is >= 0 and in the range (0, 1) if the entropy change is < 0.
# The inverse temperature beta rescales the probabilities of energetically unfavorable (negative-entropy) flips.
# For the code below, we use the balanced sim.
# For our use case, it is unrealistic for only one value per time step to change.

# To make it easier to work with the model, we package it into a PyTorch Module.
# However, it does not work with PyTorch gradient tracking, since we modify the state in place.
# When using this class, place the code in a torch.no_grad() block.
# Since keeping track of multiple batch dimensions makes the code more complicated, we assume a single batch dimension.
# Since evaluation of each model in the batch is independent, the calling code can just flatten over all batch dimensions before passing data in and unflatten after getting data out.
class IsingModelJ(torch.nn.Module):

    def __init__(self, initial_J:torch.Tensor, initial_s:torch.Tensor):
        super(IsingModelJ, self).__init__()
        self.J = initial_J.clone()
        self.s = initial_s.clone()
    
    def forward(self, time_series:torch.Tensor):
        return self.get_entropy_of_ts(time_series)
    
    # time_series is of size models_per_subject x num_subjects x num_nodes x num_time_points.
    # We return a stacked matrix of size models_per_subject x num_subjects (optionally x 1) x num_time_points
    # with the entropy of each time point
    def get_entropy_of_ts(self, time_series:torch.Tensor, keepdim:bool=False):
        return 0.5 * torch.sum( time_series * torch.matmul(self.J, time_series), dim=-2, keepdim=keepdim )
    
    def get_entropy(self):
        return 0.5 * torch.matmul(  self.s.unsqueeze(dim=-2), torch.matmul( self.J, self.s.unsqueeze(dim=-1) )  )
    
    #   0.5 * torch.matmul(  s_new.unsqueeze(dim=-2), torch.matmul( self.J, s_new.unsqueeze(dim=-1) )  )
    # - 0.5 * torch.matmul(  s_old.unsqueeze(dim=-2), torch.matmul( self.J, s_old.unsqueeze(dim=-1) )  )
    # =
    #   0.5 * torch.sum(  s_new * torch.sum( self.J * s_new.unsqueeze(dim=-2), dim=-1 ), dim=-1  )
    # - 0.5 * torch.sum(  s_old * torch.sum( self.J * s_old.unsqueeze(dim=-2), dim=-1 ), dim=-1  )
    # =
    #   0.5 * torch.sum(  s_new * torch.sum( self.J * s_new.unsqueeze(dim=-2), dim=-1 ) - s_old * torch.sum( self.J * s_old.unsqueeze(dim=-2), dim=-1 ), dim=-1  )
    # =
    #   0.5 * torch.sum(  torch.sum( s_new.unsqueeze(dim=-1) * self.J * s_new.unsqueeze(dim=-2), dim=-1 ) - torch.sum( s_old.unsqueeze(dim=-1) * self.J * s_old.unsqueeze(dim=-2), dim=-1 ), dim=-1  )
    # =
    #   0.5 * torch.sum(  torch.sum( self.J * s_new.unsqueeze(dim=-1) * s_new.unsqueeze(dim=-2), dim=-1 ) - torch.sum( self.J * s_old.unsqueeze(dim=-1) * s_old.unsqueeze(dim=-2), dim=-1 ), dim=-1  )
    # =
    #   0.5 * torch.sum(  torch.sum( self.J * s_new.unsqueeze(dim=-1) * s_new.unsqueeze(dim=-2) - self.J * s_old.unsqueeze(dim=-1) * s_old.unsqueeze(dim=-2), dim=-1 ), dim=-1  )
    # =
    #   0.5 * torch.sum(   torch.sum(  self.J * ( s_new.unsqueeze(dim=-1) * s_new.unsqueeze(dim=-2) - s_old.unsqueeze(dim=-1) * s_old.unsqueeze(dim=-2) ), dim=-1  ), dim=-1   )
    # Let i be the flipped node and j and k be any other nodes.
    # s_new[:,:,i] == -1*s_old[:,:,i], and s_new[:,:,j] == s_old[:,:,j].
    # s_new[:,:,j]*s_new[:,:,k] == s_old[:,:,j]*s_old[:,:,k], so s_new[:,:,j]*s_new[:,:,k] - s_old[:,:,j]*s_old[:,:,k] == 0.
    # s_new[:,:,j]*s_new[:,:,i] == s_old[:,:,j]*(-1)*s_old[:,:,i], so s_new[:,:,j]*s_new[:,:,i] - s_old[:,:,j]*s_old[:,:,i] == -2*s_old[:,:,j]*s_old[:,:,i].
    # s_new[:,:,i]*s_new[:,:,k] == -1*s_old[:,:,i]*s_old[:,:,k], so s_new[:,:,i]*s_new[:,:,k] - s_old[:,:,i]*s_old[:,:,k] == -2*s_old[:,:,i]*s_old[:,:,k].
    # s_new[:,:,i]*s_new[:,:,i] == (-1)*s_old[:,:,i]*(-1)*s_old[:,:,i], so s_new[:,:,i]*s_new[:,:,i] - s_old[:,:,i]*s_old[:,:,i] == 0.
    # We assume that J[:,:,i,i] == 0 so that the product at (i,i) does not matter
    # and that J is symmetric so that we can use either the row J[:,:,i,:] or the column J[:,:,:,i] for both -2*s_old[:,:,j]*s_old[:,:,i] and -2*s_old[:,:,i]*s_old[:,:,k].
    # Use J[:,:,i,:] for consistency with torch.matmul(J, s).
    # = 0.5 * torch.sum(   torch.sum(  self.J[:,:,i,:] * ( -2*s_old[:,:,:]*s_old[:,:,i] + -2*s_old[:,:,i]*s_old[:,:,:] ), dim=-1  ), dim=-1   )
    # = 0.5 * torch.sum(   torch.sum(  self.J[:,:,i,:] * ( -4*s_old[:,:,:]*s_old[:,:,i] ), dim=-1  ), dim=-1   )
    # = 0.5 * torch.sum(  torch.sum( -4*s_old[:,:,i]*self.J[:,:,i,:]*s_old[:,:,:], dim=-1 ), dim=-1  )
    # = 0.5 * torch.sum(  -4*s_old[:,:,i]*torch.sum( self.J[:,:,i,:]*s_old[:,:,:], dim=-1 ), dim=-1  )
    # = 0.5 * (-4) * torch.sum(  s_old[:,:,i]*torch.sum( self.J[:,:,i,:]*s_old[:,:,:], dim=-1 ), dim=-1  )
    # = -2 * torch.sum(  s_old[:,:,i]*torch.sum( self.J[:,:,i,:]*s_old[:,:,:], dim=-1 ), dim=-1  )
    def get_entropy_change(self, node:int):
        return -2 * self.s.index_select(dim=-1, index=node) * torch.sum( self.J.index_select(dim=-2, index=node) * self.s, dim=-1 )
    
    def do_balanced_metropolis_step(self):
        # For performance reasons, instead of doing
        # flip if rand_choice < exp( -2*s[i]*dot(J[i,:],s) ),
        # we do
        # flip if -1/2*log(rand_choice) > s[i]*dot(J[i,:],s).
        rand_choice = -0.5 * torch.rand_like(input=self.s).log()
        for node in torch.randperm( n=self.s.size(dim=-1), dtype=int_type, device=self.s.device ):
            self.s[:,:,node] *= 1.0 - 2.0*(  rand_choice[:,:,node] > ( self.s[:,:,node] * torch.sum(self.J[:,:,node,:] * self.s, dim=-1) )  ).float()

    # Simulate the Ising model.
    # Record the means of the states and products of states.
    # Record the means of state products in a square matrix.
    # This is faster but includes redundant elements that take up memory.
    def simulate_and_record_means(self, num_steps:int):
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            s_sum += self.s# B x N x 1
            torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            s_product_sum += s_product
        return s_sum/num_steps, s_product_sum/num_steps
    
    # For every subject, we have a set number (B) of models.
    # Assume they all start out with the same J.
    # Search along the interval [0, 1] for a scaling factor (beta, also called the inverse temperature) by which to multiply J
    # that minimizes the RMSE between target_state_product_means and the state product means observed in the Ising model simulation.
    # target_state_product_means should have dimensions 1 x num_subjects x num_nodes x num_nodes so that we have one target per subject.
    # At each step, we try B different values for each subject.
    # Over multiple steps, we home in more tightly on the optimal value of beta.
    def rescale_J(self, target_state_product_means:torch.Tensor, num_updates:int, num_steps:int, verbose:bool=False):
        num_targets = target_state_product_means.size(dim=1)
        # Append a singleton dimension along which to prodcast the same target to every replica of a given subject.
        # target_state_product_means = target_state_product_means.unsqueeze(dim=0)
        original_J = self.J.clone()
        num_beta = self.J.size(dim=0)
        # Initialize all beta ranges to [0, 1], but make a separate copy for every target so that we can arrive at different optimal values.
        beta = torch.linspace(start=0, end=1, steps=num_beta, dtype=self.J.dtype, device=self.J.device).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat( (1,num_targets,1,1) )
        # print( 'initialized beta with size ', beta.size() )
        self.J = beta * original_J
        num_updates_completed = 0
        for iteration in range(num_updates):
            _, sim_state_product_means = self.simulate_and_record_means(num_steps=num_steps)
            # Take the RMSE over all node pairs for each model.
            rmse_state_product_means = get_pairwise_rmse_2d(mat1=target_state_product_means, mat2=sim_state_product_means)
            # Take the minimum RMSE over all models for each target.
            _, best_beta_index = torch.min(rmse_state_product_means, dim=0)
            range_width = beta[-1,:,0,0] - beta[0,:,0,0]
            if verbose:
                print(f'iteration {iteration+1}, beta range min {range_width.min():.3g}, mean {range_width.mean():.3g}, max {range_width.max():.3g}, RMSE min {rmse_state_product_means.min():.3g}, mean {rmse_state_product_means.mean():.3g}, max {rmse_state_product_means.max():.3g}')
            for target_index in range(num_targets):
                best_beta_index_for_target = best_beta_index[target_index]
                betas_for_target = beta[:,target_index,0,0]
                best_beta_for_target = betas_for_target[best_beta_index_for_target]
                range_width_for_target = range_width[target_index]
                if best_beta_index_for_target == 0:
                    beta_start = 0
                    if best_beta_for_target == 0:
                        # If the best value we tried is 0, try values in a smaller range around 0.
                        # There might be better values that are close to but not exactly 0.
                        beta_start = 0
                        beta_end = betas_for_target[1]
                    else:
                        # If the lowest value we tried is the best and is not 0, try values less than that value.
                        # An even lower value might be even better.
                        # Preserve the current range width so long as it does not take us below 0.
                        beta_start = max(0.0, best_beta_for_target-range_width_for_target)
                        beta_end = best_beta_for_target
                elif best_beta_index_for_target == (num_beta-1):
                    # If the highest value we tried is the best, try values greater than that value.
                    # There might be even higher values that are better.
                    # Preserve the current range width.
                    beta_start = best_beta_for_target
                    beta_end = best_beta_for_target + range_width_for_target
                else:
                    # If the best value was between two other values, search the interval between those values at higher resolution.
                    beta_start = betas_for_target[best_beta_index_for_target-1]
                    beta_end = betas_for_target[best_beta_index_for_target+1]
                beta[:,target_index,0,0] = torch.linspace(start=beta_start, end=beta_end, steps=num_beta, dtype=beta.dtype, device=beta.device)
            # print( f'{sim+1}th beta[0]', beta[:,0].tolist() )
            self.J = beta * original_J
            num_updates_completed += 1
            # Stop if we have homed in on the best possible beta to within machine precision.
            if range_width.max() == 0:
                if verbose:
                    print('new betas have all converged to a single value per subject')
                break
        return beta, num_updates_completed
    
    # target_state_product_mean is of size 1 x num_subjects x num_nodes x num_nodes.
    def fit_by_simulation(self, target_state_product_mean:torch.Tensor, num_updates:torch.int, steps_per_update:torch.int=50, learning_rate:torch.float=0.001, verbose:bool=False):
        # Give the target an additional dimension so that we can broadcast it to all replica models of the same target.
        # target_state_product_mean = target_state_product_mean.unsqueeze(dim=-1)
        for update in range(num_updates):
            _, sim_state_product_mean = self.simulate_and_record_means(num_steps=steps_per_update)
            # Since the product of any state with itself is +1, the diagonals of both the target and sim product means are 1.
            # Consequently, the difference between the diagonals is 0, leading to no change in J, which is initialized to 0.
            # This is intentional, since no node should have a coupling to itsef.
            self.J += learning_rate * (target_state_product_mean - sim_state_product_mean)
            if verbose:
                rmse_state_product_mean = get_pairwise_rmse_2d(mat1=target_state_product_mean, mat2=sim_state_product_mean)
                print(f'update {update+1}, RMSE min {rmse_state_product_mean.min():.3g}, mean {rmse_state_product_mean.mean():.3g}, max {rmse_state_product_mean.max():.3g},')
    
    # target_state_product_mean is of size num_subjects x num_nodes x num_nodes.
    def fit_by_pseudolikelihood(self, num_updates:int, target_ts:torch.Tensor, target_state_product_means:torch.Tensor=None, learning_rate:torch.float=0.001, get_means_step_by_step:bool=False):
        num_time_points = target_ts.size(dim=-1)
        if type(target_state_product_means) == type(None):
            if get_means_step_by_step:
                _, target_state_product_means = get_time_series_mean_step_by_step(target_ts)
            else:
                _, target_state_product_means = get_time_series_mean(target_ts)
        for _ in range(num_updates):
            # models_per_subject x num_subjects x num_nodes x num_nodes @ 1 x num_subjects x num_nodes x num_steps -> models_per_subject x num_subjects x num_nodes x num_steps
            # mean_field = torch.tanh( torch.matmul(self.J, target_ts) )
            # 1 x num_subjects x num_nodes x num_steps @ models_per_subject x num_subjects x num_steps x num_nodes -> models_per_subject x num_subjects x num_nodes x num_nodes
            # mean_field_outer_product = torch.matmul(  target_ts, torch.tanh( torch.matmul(self.J, target_ts) ).transpose(dim0=-2, dim1=-1)  )/num_time_points
            self.J += learning_rate * (   target_state_product_means - torch.matmul(  target_ts, torch.tanh( torch.matmul(self.J, target_ts) ).transpose(dim0=-2, dim1=-1)  )/num_time_points   )
    
    # Run a simulation, and record and return the functional connectivity matrix (FC).
    # The FC has dimensions B x N x N where fc[b,i,j] is the Pearson correlation between nodes i and j in model b.
    # We replace any 0s in the denominator of the division at the end with the small number epsilon to avoid division by 0.
    # Since the FC is symmetric with 1 along the diagonal, we only return the upper triangular part above the diagonal.
    def simulate_and_record_fc(self, num_steps:int, epsilon:torch.float=10e-10):
        s_mean, s_product_mean = self.simulate_and_record_means(num_steps=num_steps)
        return get_fc_binary(state_mean=s_mean, state_product_mean=s_product_mean, epsilon=epsilon)
    
    # Run a simulation, and record and return the Fisher Information Matrix (FIM) estimated from the observed states.
    # Since the FIM is symmetric, we only retain the upper triangular part, including the diagonal.
    def simulate_and_record_fim(self, num_steps:int):
        dtype = self.s.dtype
        device = self.s.device
        models_per_subject, num_subjects, num_nodes = self.s.size()
        state_triu_rows, state_triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=device)
        # Unlike with the products matrix, the diagonal of the FIM is meaningful.
        num_observables = state_triu_rows.numel()
        fim_triu_indices = torch.triu_indices(row=num_observables, col=num_observables, offset=0, dtype=int_type, device=device)
        fim_triu_rows = fim_triu_indices[0]
        fim_triu_cols = fim_triu_indices[1]
        num_fim_triu_elements = fim_triu_rows.numel()
        observables = torch.zeros( (models_per_subject, num_subjects, num_observables), dtype=dtype, device=device )
        observables_mean = torch.zeros_like(observables)
        observable_product_sum = torch.zeros( (models_per_subject, num_subjects, num_fim_triu_elements), dtype=dtype, device=device )
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            observables[:,:,:] = self.s[:,:,state_triu_rows] * self.s[:,:,state_triu_cols]
            observables_mean += observables
            observable_product_sum += observables[:,:,fim_triu_rows] * observables[:,:,fim_triu_cols]
        observables_mean /= num_steps
        return observable_product_sum/num_steps - observables_mean[:,:,fim_triu_rows] * observables_mean[:,:,fim_triu_cols]
    
    # Run a simulation, and record and return the Fisher Information Matrix (FIM) estimated from the observed states.
    # Since the FIM is symmetric, we only retain the upper triangular part, including the diagonal.
    # In this version, we combine observations for all models into a single FIM.
    def simulate_and_record_fim(self, num_steps:int):
        dtype = self.s.dtype
        device = self.s.device
        models_per_subject, num_subjects, num_nodes = self.s.size()
        state_triu_rows, state_triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=device)
        # Unlike with the products matrix, the diagonal of the FIM is meaningful.
        num_observables = state_triu_rows.numel()
        fim_triu_indices = torch.triu_indices(row=num_observables, col=num_observables, offset=0, dtype=int_type, device=device)
        fim_triu_rows = fim_triu_indices[0]
        fim_triu_cols = fim_triu_indices[1]
        num_fim_triu_elements = fim_triu_rows.numel()
        observables = torch.zeros( (models_per_subject, num_subjects, num_observables), dtype=dtype, device=device )
        observables_mean = torch.zeros( (num_observables,), dtype=dtype, device=device )
        observable_product_sum = torch.zeros( (num_fim_triu_elements,), dtype=dtype, device=device )
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            observables[:,:,:] = self.s[:,:,state_triu_rows] * self.s[:,:,state_triu_cols]
            observables_mean += observables.sum( dim=(0,1) )
            observable_product_sum += torch.sum( observables[:,:,fim_triu_rows] * observables[:,:,fim_triu_cols], dim=(0,1) )
        num_observations = models_per_subject * num_subjects * num_steps
        observables_mean /= num_observations
        return observable_product_sum/num_observations - observables_mean[fim_triu_rows] * observables_mean[fim_triu_cols]

    # Simulate the Ising model.
    # Record the full time series of states.
    def simulate_and_record_time_series(self, num_steps:int):
        models_per_subject, num_subjects, num_nodes = self.s.size()
        time_series = torch.zeros( size=(models_per_subject, num_subjects, num_nodes, num_steps), dtype=self.s.dtype, device=self.s.device )
        for step in range(num_steps):
            self.do_balanced_metropolis_step()
            time_series[:,:,:,step] = self.s
        return time_series
        
    def get_triu_indices_for_products(self):
        return get_triu_indices_for_products( num_nodes=self.s.size(dim=-1), device=self.s.device )