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

def num_nodes_to_num_pairs(num_nodes:int):
    return ( num_nodes*(num_nodes-1) )//2

# num_pairs = num_nodes*(num_nodes-1)//2
# We can use the quadratic formula to get back an expression for num_nodes in terms of num_pairs.
def num_pairs_to_num_nodes(num_pairs:int):
    return int(  ( math.sqrt(1 + 8*num_pairs) + 1 )/2  )

def get_unit_beta(models_per_subject:int, num_subjects:int, dtype=float_type, device='cpu'):
    return torch.ones( size=(models_per_subject, num_subjects), dtype=dtype, device=device )

def get_linspace_beta(models_per_subject:int, num_subjects:int, dtype=float_type, device='cpu', min_beta:float=10e-10, max_beta:float=1.0):
    return torch.linspace(start=min_beta, end=max_beta, steps=models_per_subject, dtype=dtype, device=device).unsqueeze(dim=-1).repeat( repeats=(1,num_subjects) )

def zero_diag(square_mat:torch.Tensor):
    return square_mat - torch.diag_embed( input=torch.diagonal(input=square_mat, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )

def get_J_from_means(models_per_subject:int, mean_state_product:torch.Tensor):
    return zero_diag(mean_state_product).unsqueeze(dim=0).repeat( repeats=(models_per_subject, 1, 1, 1) )

def get_h_from_means(models_per_subject:int, mean_state:torch.Tensor):
    return mean_state.unsqueeze(dim=0).repeat( repeats=(models_per_subject, 1, 1) )

def get_random_J(models_per_subject:int, num_subjects:int, num_nodes:int, dtype=float_type, device='cpu'):
    return zero_diag(  torch.randn( size=(models_per_subject, num_subjects, num_nodes, num_nodes), dtype=dtype, device=device )/math.sqrt(num_nodes)  )

def get_random_h(models_per_subject:int, num_subjects:int, num_nodes:int, dtype=float_type, device='cpu'):
    return torch.randn( size=(models_per_subject, num_subjects, num_nodes), dtype=dtype, device=device )

def get_zero_h(models_per_subject:int, num_subjects:int, num_nodes:int, dtype=float_type, device='cpu'):
    return torch.zeros( size=(models_per_subject, num_subjects, num_nodes), dtype=dtype, device=device )

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
# We halve torch.matmul( J, s.unsqueeze(dim=-1) ), because it includes both
# J[i,j]*s[j], which later gets multiplied by s[i], and
#  J[j,i]*s[j,i], which later gets multiplied by s[j], even though we should count each coupling only once.
def get_entropy(s:torch.Tensor, h:torch.Tensor, J:torch.Tensor):
    return torch.matmul(  s.unsqueeze(dim=-2), h.unsqueeze(dim=-1) + 0.5*torch.matmul( J, s.unsqueeze(dim=-1) )  )

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
#        = -2*s_before[i]*( h[i] + 0.5 * torch.sum(J[i,:] * s_before) )
def get_entropy_change(s:torch.Tensor, h:torch.Tensor, J:torch.Tensor, node:int):
    return -2 * s.index_select(dim=-1, index=node) * (  h.index_select(dim=-1, index=node) + 0.5 * torch.matmul( J.index_select(dim=-2, index=node).unsqueeze(dim=-2), s.unsqueeze(dim=-1) )  )

# The time series has size models_per_subject x num_subjects x num_nodes x num_steps.
# As such, we need to index into it differently from how we would a single state.
# We return a Tensor of size models_per_subject x num_subjects x num_steps.
def get_entropy_of_time_series(ts:torch.Tensor, h:torch.Tensor, J:torch.Tensor):
    return torch.matmul(  ts.unsqueeze(dim=-2), h.unsqueeze(dim=-1) + 0.5*torch.matmul( J, ts )  )

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

def binarize_data_ts_z(data_ts:torch.Tensor, threshold:float=0):
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

def get_fc(state_mean:torch.Tensor, state_product_mean:torch.Tensor, epsilon:float=10e-10):
    # For the functional connectivity (FC), we use the Pearson correlations between pairs of nodes.
    # The correlation between nodes i and j is ( mean(s[i]*s[j]) - mean(s[i])*mean(s[j]) )/( std.dev.(s[i])*std.dev(s[j]) )
    s_std = get_std(state_mean=state_mean, state_product_mean=state_product_mean)
    return ( get_cov(state_mean=state_mean, state_product_mean=state_product_mean) + epsilon )/( s_std.unsqueeze(dim=-1) * s_std.unsqueeze(dim=-2) + epsilon)

def get_fc_binary(state_mean:torch.Tensor, state_product_mean:torch.Tensor, epsilon:float=10e-10):
    # For the functional connectivity (FC), we use the Pearson correlations between pairs of nodes.
    # The correlation between nodes i and j is ( mean(s[i]*s[j]) - mean(s[i])*mean(s[j]) )/( std.dev.(s[i])*std.dev(s[j]) )
    s_std = get_std_binary(state_mean=state_mean)
    return ( get_cov(state_mean=state_mean, state_product_mean=state_product_mean) + epsilon )/( s_std.unsqueeze(dim=-1) * s_std.unsqueeze(dim=-2) + epsilon)

def get_pairwise_correlation(mat1:torch.Tensor, mat2:torch.Tensor, epsilon:float=10e-10, dim:int=-1):
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

def get_pairwise_r_squared(target:torch.Tensor, prediction:torch.Tensor, dim:int=-1):
    # If the number of dimensions is unequal, assume that one of them is missing the 0/model dimension.
    # This is true when we compare data observables to model simulation observables.
    if len( target.size() ) < len( prediction.size() ):
        target = target.unsqueeze(dim=0)
    elif len( prediction.size() ) < len( target.size() ):
        prediction = prediction.unsqueeze(dim=0)
    return 1.0 - torch.mean( (target - prediction).square(), dim=dim )/torch.var(target, dim=dim)

def get_pairwise_correlation_2d(mat1:torch.Tensor, mat2:torch.Tensor, epsilon:float=10e-10):
    return get_pairwise_correlation( mat1=mat1, mat2=mat2, epsilon=epsilon, dim=(-2,-1) )

def get_pairwise_rmse_2d(mat1:torch.Tensor, mat2:torch.Tensor):
    return get_pairwise_rmse( mat1=mat1, mat2=mat2, dim=(-2,-1) )

def get_pairwise_correlation_ut(mat1:torch.Tensor, mat2:torch.Tensor, epsilon:float=10e-10):
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

# @torch.compile
def get_next_state(s:torch.Tensor, h2:torch.Tensor, J:torch.Tensor, log_rand_choice_over_beta:torch.Tensor, node_index:int):
    return s[:,:,node_index] * (   1.0 - 2.0*(  log_rand_choice_over_beta[:,:,node_index] > ( s[:,:,node_index] * ( h2[:,:,node_index] + torch.sum(J[:,:,node_index,:] * s, dim=-1) ) )  ).float()   )

# @torch.compile
def get_next_state_preindexed(s_i:torch.Tensor, s:torch.Tensor, h2_i:torch.Tensor, J_i:torch.Tensor, log_rand_choice_over_beta_i:torch.Tensor):
    return s_i * (   1.0 - 2.0*(  log_rand_choice_over_beta_i > ( s_i * ( h2_i + torch.sum(J_i * s, dim=-1) ) )  ).float()   )

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
class IsingModelLight(torch.nn.Module):

    def __init__(self, beta:torch.Tensor, J:torch.Tensor, h:torch.Tensor, s:torch.Tensor):
        super(IsingModelLight, self).__init__()
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
    
    def get_entropy(self):
        return get_entropy(s=self.s, h=self.h, J=self.J)
    
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
        return get_entropy_change(s=self.s, h=self.h, J=self.J, node=node)
    
    def do_balanced_metropolis_step(self):
        # flip if rand_choice < exp(  -2*s[i]*( h+0.5*dot(J[i,:],s) )  ) =
        # flip if log(rand_choice) < -2*s[i]*( h+0.5*dot(J[i,:],s) ) =
        # flip if -0.5*log(rand_choice) > s[i]*( h+0.5*dot(J[i,:],s) ) =
        # flip if -1*log(rand_choice) > 2*s[i]*( h+0.5*dot(J[i,:],s) ) =
        # flip if -1*log(rand_choice) > s[i]*( 2*h+dot(J[i,:],s) ).
        rand_choice = -1 * torch.rand_like(input=self.s).log() / self.beta.unsqueeze(dim=-1)
        h2 = 2 * self.h
        for node in torch.randperm( n=self.s.size(dim=-1), dtype=int_type, device=self.s.device ):
            self.s[:,:,node] = get_next_state(s=self.s, h2=h2, J=self.J, log_rand_choice_over_beta=rand_choice, node_index=node)
            # self.s[:,:,node] *= 1.0 - 2.0*(  rand_choice[:,:,node] > ( self.s[:,:,node] * ( h2[:,:,node] + torch.sum(self.J[:,:,node,:] * self.s, dim=-1) ) )  ).float()

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
        s_sum /= num_steps
        s_product_sum /= num_steps
        return s_sum, s_product_sum
    
    # Simulate the Ising model.
    # Record the means of the states and products of states.
    # Record the means of state products in a square matrix.
    # This is faster but includes redundant elements that take up memory.
    # @torch.compile
    def simulate_and_record_means_faster(self, num_steps:int):
        num_nodes = self.s.size(dim=-1)
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        h2 = 2 * self.h
        for _ in range(num_steps):
            rand_choice = -1 * torch.rand_like(input=self.s).log() / self.beta.unsqueeze(dim=-1)
            for node in range(num_nodes):
                self.s[:,:,node] *= 1.0 - 2.0*(  rand_choice[:,:,node] > ( self.s[:,:,node] * ( h2[:,:,node] + torch.sum(self.J[:,:,node,:] * self.s, dim=-1) ) )  ).float()
            s_sum += self.s# B x N x 1
            torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            s_product_sum += s_product
        s_sum /= num_steps
        s_product_sum /= num_steps
        return s_sum, s_product_sum
    
    # Run a simulation, and record and return the covariance matrix.
    # The covariance has dimensions B x N x N where cov[b,i,j] is the covariance between nodes i and j in model b.
    # We replace any 0s in the denominator of the division at the end with the small number epsilon to avoid division by 0.
    def simulate_and_record_cov(self, num_steps:int):
        s_mean, s_product_mean = self.simulate_and_record_means(num_steps=num_steps)
        return get_cov(state_mean=s_mean, state_product_mean=s_product_mean)
    
    # Run a simulation, and record and return the functional connectivity matrix (FC).
    # The FC has dimensions B x N x N where fc[b,i,j] is the Pearson correlation between nodes i and j in model b.
    # We replace any 0s in the denominator of the division at the end with the small number epsilon to avoid division by 0.
    # Since the FC is symmetric with 1 along the diagonal, we only return the upper triangular part above the diagonal.
    def simulate_and_record_fc(self, num_steps:int, epsilon:float=10e-10):
        s_mean, s_product_mean = self.simulate_and_record_means(num_steps=num_steps)
        return get_fc_binary(state_mean=s_mean, state_product_mean=s_product_mean, epsilon=epsilon)
    
    # For every subject, we have a set number (B) of models.
    # Assume they all start out with the same J.
    # Search along the interval [0, 1] for a scaling factor (beta, also called the inverse temperature) by which to multiply J
    # that minimizes the RMSE between target_fc and the FC observed in the Ising model simulation.
    # target_state_product_means should have dimensions 1 x num_subjects x num_nodes x num_nodes so that we have one target per subject.
    # At each step, we try B different values for each subject.
    # Over multiple steps, we home in more tightly on the optimal value of beta.
    # @torch.compile
    def optimize_beta(self, target_cov:torch.Tensor, num_updates:int, num_steps:int, verbose:bool=False, epsilon:float=10e-10):
        target_cov = target_cov.unsqueeze(dim=0)
        # Initialize all beta ranges to [epsilon, 1], but make a separate copy for every target so that we can arrive at different optimal values.
        models_per_subject, num_subjects, _ = self.s.size()
        dtype = self.s.dtype
        device = self.s.device
        # We need this arange() Tensor for indexing later.
        subject_index = torch.arange(start=0, end=num_subjects, step=1, dtype=int_type, device=device)
        beta_steps = torch.arange(start=0, end=models_per_subject, step=1, dtype=dtype, device=device)
        beta_start = torch.full( size=(num_subjects,), fill_value=epsilon, dtype=dtype, device=device )
        beta_end = torch.ones_like(beta_start)
        # print( 'initialized beta with size ', beta.size() )
        num_updates_completed = 0
        for iteration in range(num_updates):
            interval_init_time = time.time()
            self.beta = beta_start.unsqueeze(dim=0) + ( (beta_end - beta_start)/(models_per_subject-1) ).unsqueeze(dim=0) * beta_steps.unsqueeze(dim=1)
            print(f'interval init time {time.time()-interval_init_time:.3f}')
            sim_start_time = time.time()
            # sim_state_cov, flip_rate = self.simulate_and_record_cov_and_flip_rate(num_steps=num_steps)
            sim_state_cov = self.simulate_and_record_cov(num_steps=num_steps)
            print(f'sim time {time.time()-sim_start_time:.3f}')
            interval_calc_start_time = time.time()
            # Take the RMSE over all node pairs for each model.
            cov_rmse = get_pairwise_rmse_2d(mat1=target_cov, mat2=sim_state_cov)
            # Take the minimum RMSE over all models for each target.
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
            print(f'interval calculation time {time.time()-interval_calc_start_time:.3f}')
            if verbose:
                print(f'iteration {iteration+1}, best beta min {best_beta.min():.3g}, mean {best_beta.mean():.3g}, max {best_beta.max():.3g}, beta range min {range_width.min():.3g}, mean {range_width.mean():.3g}, max {range_width.max():.3g}, RMSE min {cov_rmse.min():.3g}, mean {cov_rmse.mean():.3g}, max {cov_rmse.max():.3g}')
                # print('beta')
                # print( self.beta )
                # print('range width')
                # print( range_width )
                # print('flip_rate')
                # print( flip_rate )
                # print('cov_rmse')
                # print( cov_rmse )
                # print('cov_rmse range')
                # print(  ( cov_rmse.max(dim=0).values - cov_rmse.min(dim=0).values )  )
                # print('best_rmse')
                # print( best_rmse )
                # print('best beta index')
                # print( best_beta_index )
                # print('best beta')
                # print( best_beta )
                # print('best_is_epsilon')
                # print( best_is_epsilon )
                # print('best_is_least')
                # print( best_is_least )
                # print('best_is_most')
                # print( best_is_most )
                # print('best_is_middle')
                # print( best_is_middle )
                # print('beta_start')
                # print( beta_start )
                # print('beta_end')
                # print( beta_end )
                # print(f'iteration {iteration+1}, best beta min {best_beta.min():.3g}, mean {best_beta.mean():.3g}, max {best_beta.max():.3g}, best beta index min {best_beta_index.float().min():.3g}, mean {best_beta_index.float().mean():.3g}, max {best_beta_index.float().max():.3g}, beta range min {range_width.min():.3g}, mean {range_width.mean():.3g}, max {range_width.max():.3g}, RMSE min {cov_rmse.min():.3g}, mean {cov_rmse.mean():.3g}, max {cov_rmse.max():.3g}, flip rate min {flip_rate.min():.3g}, mean {flip_rate.mean():.3g}, max {flip_rate.max():.3g}')
            num_updates_completed += 1
            # Stop if we have homed in on the best possible beta to within machine precision.
            if range_width.max() == 0:
                if verbose:
                    print('betas have all converged to a single value per subject')
                break
        return num_updates_completed
    
    # @torch.compile
    def optimize_beta_faster(self, target_cov:torch.Tensor, num_updates:int, num_steps:int, verbose:bool=False, epsilon:float=10e-10, min_beta:float=10e-10, max_beta:float=1.0):
        opt_start_time = time.time()
        target_cov = target_cov.unsqueeze(dim=0)
        # Initialize all beta ranges to [epsilon, 1], but make a separate copy for every target so that we can arrive at different optimal values.
        models_per_subject, num_subjects, num_nodes = self.s.size()
        dtype = self.s.dtype
        device = self.s.device
        # We need this arange() Tensor for indexing later.
        subject_index = torch.arange(start=0, end=num_subjects, step=1, dtype=int_type, device=device)
        beta_steps = torch.arange(start=0, end=models_per_subject, step=1, dtype=dtype, device=device)
        beta_start = torch.full( size=(num_subjects,), fill_value=min_beta, dtype=dtype, device=device )
        beta_end = torch.full_like(beta_start, fill_value=max_beta)
        # We only need to set h2 once, because we do not change h in this method.
        h2 = 2 * self.h
        rand_choice = torch.zeros_like(self.s)
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        # print( 'initialized beta with size ', beta.size() )
        num_updates_completed = 0
        for iteration in range(num_updates):
            # interval_init_time = time.time()
            self.beta = beta_start.unsqueeze(dim=0) + ( (beta_end - beta_start)/(models_per_subject-1) ).unsqueeze(dim=0) * beta_steps.unsqueeze(dim=1)
            # print(f'interval init time {time.time()-interval_init_time:.3f}')
            s_sum.zero_()
            s_product_sum.zero_()
            # sim_start_time = time.time()
            neg_over_beta_3d = -1 / self.beta.unsqueeze(dim=-1)
            # sim_state_cov, flip_rate = self.simulate_and_record_cov_and_flip_rate(num_steps=num_steps)
            # sim_state_cov = self.simulate_and_record_cov(num_steps=num_steps)
            for _ in range(num_steps):
                torch.mul( neg_over_beta_3d, torch.rand_like(input=self.s).log(), out=rand_choice )
                for node in range(num_nodes):
                    # self.s[:,:,node] = get_next_state_preindexed(s_i=self.s[:,:,node], s=self.s, h2_i=h2[:,:,node], J_i=self.J[:,:,node,:], log_rand_choice_over_beta_i=rand_choice[:,:,node])
                    # self.s[:,:,node] = get_next_state(s=self.s, h2=h2, J=self.J, log_rand_choice_over_beta=rand_choice, node_index=node)
                    self.s[:,:,node] *= (   1.0 - 2.0*(  rand_choice[:,:,node] > ( self.s[:,:,node] * ( h2[:,:,node] + torch.sum(self.J[:,:,node,:] * self.s, dim=-1) ) )  ).float()   )
                s_sum += self.s# B x N x 1
                torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
                s_product_sum += s_product
            sim_state_cov = get_cov(state_mean=s_sum/num_steps, state_product_mean=s_product_sum/num_steps)
            # print(f'sim time {time.time()-sim_start_time:.3f}')
            # interval_calc_start_time = time.time()
            # Take the RMSE over all node pairs for each model.
            cov_rmse = get_pairwise_rmse_2d(mat1=target_cov, mat2=sim_state_cov)
            # Take the minimum RMSE over all models for each target.
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
    
    # @torch.compile
    def optimize_beta_msp(self, target_mean_state_product:torch.Tensor, num_updates:int, num_steps:int, min_beta:float=10e-10, max_beta:float=1.0, epsilon:float=10e-10, verbose:bool=False):
        opt_start_time = time.time()
        target_mean_state_product = target_mean_state_product.unsqueeze(dim=0)
        # Initialize all beta ranges to [epsilon, 1], but make a separate copy for every target so that we can arrive at different optimal values.
        models_per_subject, num_subjects, num_nodes = self.s.size()
        dtype = self.s.dtype
        device = self.s.device
        # We need this arange() Tensor for indexing later.
        subject_index = torch.arange(start=0, end=num_subjects, step=1, dtype=int_type, device=device)
        beta_steps = torch.arange(start=0, end=models_per_subject, step=1, dtype=dtype, device=device)
        beta_start = torch.full( size=(num_subjects,), fill_value=min_beta, dtype=dtype, device=device )
        beta_end = torch.full_like(beta_start, fill_value=max_beta)
        # We only need to set h2 once, because we do not change h in this method.
        rand_choice = torch.zeros_like(self.s)
        # s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        # print( 'initialized beta with size ', beta.size() )
        num_updates_completed = 0
        for iteration in range(num_updates):
            # interval_init_time = time.time()
            self.beta = beta_start.unsqueeze(dim=0) + ( (beta_end - beta_start)/(models_per_subject-1) ).unsqueeze(dim=0) * beta_steps.unsqueeze(dim=1)
            hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
            Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
            # print(f'interval init time {time.time()-interval_init_time:.3f}')
            # s_sum.zero_()
            s_product_sum.zero_()
            # sim_start_time = time.time()
            for _ in range(num_steps):
                torch.rand( size=rand_choice.size(), out=rand_choice )
                rand_choice.log_()
                for node in range(num_nodes):
                    self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
                # s_sum += self.s# B x N x 1
                torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
                s_product_sum += s_product
            s_product_sum /= num_steps
            # print(f'sim time {time.time()-sim_start_time:.3f}')
            # interval_calc_start_time = time.time()
            # Take the RMSE over all node pairs for each model.
            msp_rmse = get_pairwise_rmse_2d(mat1=target_mean_state_product, mat2=s_product_sum)
            # Take the minimum RMSE over all models for each target.
            _, best_beta_index = torch.min(msp_rmse, dim=0)
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
                print(f'time {time.time()-opt_start_time:.3f}, iteration {iteration+1}, best beta min {best_beta.min():.3g}, mean {best_beta.mean():.3g}, max {best_beta.max():.3g}, beta range min {range_width.min():.3g}, mean {range_width.mean():.3g}, max {range_width.max():.3g}, cov RMSE min {msp_rmse.min():.3g}, mean {msp_rmse.mean():.3g}, max {msp_rmse.max():.3g}')
            num_updates_completed += 1
            # Stop if we have homed in on the best possible beta to within machine precision.
            if range_width.max() == 0:
                if verbose:
                    print('betas have all converged to a single value per subject')
                break
        return num_updates_completed
    
    # @torch.compile
    def optimize_beta_pmb(self, target_cov:torch.Tensor, num_updates:int, num_steps:int, min_beta:float=10e-10, max_beta:float=1.0, epsilon:float=10e-10, verbose:bool=False):
        opt_start_time = time.time()
        target_cov = target_cov.unsqueeze(dim=0)
        # Initialize all beta ranges to [epsilon, 1], but make a separate copy for every target so that we can arrive at different optimal values.
        models_per_subject, num_subjects, num_nodes = self.s.size()
        dtype = self.s.dtype
        device = self.s.device
        # We need this arange() Tensor for indexing later.
        subject_index = torch.arange(start=0, end=num_subjects, step=1, dtype=int_type, device=device)
        beta_steps = torch.arange(start=0, end=models_per_subject, step=1, dtype=dtype, device=device)
        beta_start = torch.full( size=(num_subjects,), fill_value=min_beta, dtype=dtype, device=device )
        beta_end = torch.full_like(beta_start, fill_value=max_beta)
        rand_choice = torch.zeros_like(self.s)
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        # print( 'initialized beta with size ', beta.size() )
        num_updates_completed = 0
        for iteration in range(num_updates):
            # interval_init_time = time.time()
            self.beta = beta_start.unsqueeze(dim=0) + ( (beta_end - beta_start)/(models_per_subject-1) ).unsqueeze(dim=0) * beta_steps.unsqueeze(dim=1)
            hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
            Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
            # print(f'interval init time {time.time()-interval_init_time:.3f}')
            s_sum.zero_()
            s_product_sum.zero_()
            # sim_start_time = time.time()
            for _ in range(num_steps):
                torch.rand( size=rand_choice.size(), out=rand_choice )
                rand_choice.log_()
                for node in range(num_nodes):
                    self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
                s_sum += self.s# B x N x 1
                torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
                s_product_sum += s_product
            sim_state_cov = get_cov(state_mean=s_sum/num_steps, state_product_mean=s_product_sum/num_steps)
            # print(f'sim time {time.time()-sim_start_time:.3f}')
            # interval_calc_start_time = time.time()
            # Take the RMSE over all node pairs for each model.
            cov_rmse = get_pairwise_rmse_2d(mat1=target_cov, mat2=sim_state_cov)
            # Take the minimum RMSE over all models for each target.
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
    
    # @torch.compile
    def optimize_beta_for_fc_corr_pmb(self, target_fc:torch.Tensor, num_updates:int, num_steps:int, min_beta:float=10e-10, max_beta:float=1.0, epsilon:float=10e-10, verbose:bool=False):
        opt_start_time = time.time()
        target_fc = target_fc.unsqueeze(dim=0)
        # Initialize all beta ranges to [epsilon, 1], but make a separate copy for every target so that we can arrive at different optimal values.
        models_per_subject, num_subjects, num_nodes = self.s.size()
        dtype = self.s.dtype
        device = self.s.device
        # We need this arange() Tensor for indexing later.
        subject_index = torch.arange(start=0, end=num_subjects, step=1, dtype=int_type, device=device)
        beta_steps = torch.arange(start=0, end=models_per_subject, step=1, dtype=dtype, device=device)
        beta_start = torch.full( size=(num_subjects,), fill_value=min_beta, dtype=dtype, device=device )
        beta_end = torch.full_like(beta_start, fill_value=max_beta)
        rand_choice = torch.zeros_like(self.s)
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        # print( 'initialized beta with size ', beta.size() )
        num_updates_completed = 0
        for iteration in range(num_updates):
            # interval_init_time = time.time()
            self.beta = beta_start.unsqueeze(dim=0) + ( (beta_end - beta_start)/(models_per_subject-1) ).unsqueeze(dim=0) * beta_steps.unsqueeze(dim=1)
            hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
            Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
            # print(f'interval init time {time.time()-interval_init_time:.3f}')
            s_sum.zero_()
            s_product_sum.zero_()
            # sim_start_time = time.time()
            for _ in range(num_steps):
                torch.rand( size=rand_choice.size(), out=rand_choice )
                rand_choice.log_()
                for node in range(num_nodes):
                    self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
                s_sum += self.s# B x N x 1
                torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
                s_product_sum += s_product
            s_sum /= num_steps
            s_product_sum /= num_steps
            sim_fc = get_fc(state_mean=s_sum, state_product_mean=s_product_sum)
            # print(f'sim time {time.time()-sim_start_time:.3f}')
            # interval_calc_start_time = time.time()
            # Take the FC correlation over all node pairs above the diagonal for each model.
            fc_corr = get_pairwise_correlation_ut(mat1=target_fc, mat2=sim_fc)
            # If FC correlation is NaN or Inf, set it to -1 so that we do not select it.
            fc_corr[ torch.logical_or( torch.isnan(fc_corr), torch.isinf(fc_corr) ) ] = -1.0
            # Take the maximum correlation over all temperatures for each target.
            _, best_beta_index = torch.max(fc_corr, dim=0)
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
                print(f'time {time.time()-opt_start_time:.3f}, iteration {iteration+1}, best beta min {best_beta.min():.3g}, mean {best_beta.mean():.3g}, max {best_beta.max():.3g}, beta range min {range_width.min():.3g}, mean {range_width.mean():.3g}, max {range_width.max():.3g}, FC correlation min {fc_corr.min():.3g}, mean {fc_corr.mean():.3g}, max {fc_corr.max():.3g}')
            num_updates_completed += 1
            # Stop if we have homed in on the best possible beta to within machine precision.
            if range_width.max() == 0:
                if verbose:
                    print('betas have all converged to a single value per subject')
                break
        return num_updates_completed, fc_corr
    
    # @torch.compile
    def optimize_beta_for_flip_rate_pmb(self, target_flip_rate:float, num_updates:int, num_steps:int, min_beta:float=10e-10, max_beta:float=1.0, epsilon:float=10e-10, verbose:bool=False):
        opt_start_time = time.time()
        # Initialize all beta ranges to [epsilon, 1], but make a separate copy for every target so that we can arrive at different optimal values.
        models_per_subject, num_subjects, num_nodes = self.s.size()
        dtype = self.s.dtype
        device = self.s.device
        # We need this arange() Tensor for indexing later.
        subject_index = torch.arange(start=0, end=num_subjects, step=1, dtype=int_type, device=device)
        beta_steps = torch.arange(start=0, end=models_per_subject, step=1, dtype=dtype, device=device)
        beta_start = torch.full( size=(num_subjects,), fill_value=min_beta, dtype=dtype, device=device )
        beta_end = torch.full_like(beta_start, fill_value=max_beta)
        rand_choice = torch.zeros_like(self.s)
        s_previous = self.s.clone()
        flip_count = torch.zeros_like(s_previous)
        # print( 'initialized beta with size ', beta.size() )
        num_updates_completed = 0
        for iteration in range(num_updates):
            # interval_init_time = time.time()
            self.beta = beta_start.unsqueeze(dim=0) + ( (beta_end - beta_start)/(models_per_subject-1) ).unsqueeze(dim=0) * beta_steps.unsqueeze(dim=1)
            hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
            Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
            flip_count.zero_()
            # sim_start_time = time.time()
            for _ in range(num_steps):
                torch.rand( size=rand_choice.size(), out=rand_choice )
                rand_choice.log_()
                for node in range(num_nodes):
                    self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
                flip_count += torch.abs(self.s - s_previous)/2
                s_previous[:,:,:] = self.s[:,:,:]
            flip_rate = torch.mean(flip_count/num_steps, dim=-1)
            diff_from_target = torch.abs(flip_rate - target_flip_rate)
            # Take the minimum difference over all temperatures for each target.
            _, best_beta_index = torch.min(diff_from_target, dim=0)
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
                print(f'time {time.time()-opt_start_time:.3f}, iteration {iteration+1}, best beta min {best_beta.min():.3g}, mean {best_beta.mean():.3g}, max {best_beta.max():.3g}, beta range min {range_width.min():.3g}, mean {range_width.mean():.3g}, max {range_width.max():.3g}, flip rate min {flip_rate.min():.3g}, mean {flip_rate.mean():.3g}, max {flip_rate.max():.3g}')
            num_updates_completed += 1
            # Stop if we have homed in on the best possible beta to within machine precision.
            if range_width.max() == 0:
                if verbose:
                    print('betas have all converged to a single value per subject')
                break
        return num_updates_completed
    
    # @torch.compile
    def optimize_beta_for_flip_rate_faster(self, target_flip_rate:float, num_updates:int, num_steps:int, verbose:bool=False, epsilon:float=10e-10, min_beta:float=10e-10, max_beta:float=1.0):
        opt_start_time = time.time()
        # Initialize all beta ranges to [epsilon, 1], but make a separate copy for every target so that we can arrive at different optimal values.
        models_per_subject, num_subjects, num_nodes = self.s.size()
        dtype = self.s.dtype
        device = self.s.device
        # We need this arange() Tensor for indexing later.
        subject_index = torch.arange(start=0, end=num_subjects, step=1, dtype=int_type, device=device)
        beta_steps = torch.arange(start=0, end=models_per_subject, step=1, dtype=dtype, device=device)
        beta_start = torch.full( size=(num_subjects,), fill_value=min_beta, dtype=dtype, device=device )
        beta_end = torch.full_like(beta_start, fill_value=max_beta)
        # We only need to set h2 once, because we do not change h in this method.
        h2 = 2 * self.h
        rand_choice = torch.zeros_like(self.s)
        s_previous = torch.zeros_like(self.s)
        flip_count = torch.zeros_like(self.s)
        # print( 'initialized beta with size ', beta.size() )
        num_updates_completed = 0
        for iteration in range(num_updates):
            self.beta = beta_start.unsqueeze(dim=0) + ( (beta_end - beta_start)/(models_per_subject-1) ).unsqueeze(dim=0) * beta_steps.unsqueeze(dim=1)
            neg_over_beta_3d = -1 / self.beta.unsqueeze(dim=-1)
            flip_count.zero_()
            for _ in range(num_steps):
                s_previous[:,:,:] = self.s
                torch.mul( neg_over_beta_3d, torch.rand_like(input=self.s).log(), out=rand_choice )
                for node in range(num_nodes):
                    # self.s[:,:,node] = get_next_state_preindexed(s_i=self.s[:,:,node], s=self.s, h2_i=h2[:,:,node], J_i=self.J[:,:,node,:], log_rand_choice_over_beta_i=rand_choice[:,:,node])
                    # self.s[:,:,node] = get_next_state(s=self.s, h2=h2, J=self.J, log_rand_choice_over_beta=rand_choice, node_index=node)
                    self.s[:,:,node] *= (   1.0 - 2.0*(  rand_choice[:,:,node] > ( self.s[:,:,node] * ( h2[:,:,node] + torch.sum(self.J[:,:,node,:] * self.s, dim=-1) ) )  ).float()   )
                flip_count += torch.abs(self.s - s_previous)/2
            flip_count /= num_steps
            # Get one flip rate RMSE value per model by taking the mean over nodes.
            flip_rate_rmse = torch.mean( torch.square(flip_count - target_flip_rate), dim=-1 )
            # Take the minimum RMSE over all models for each target.
            _, best_beta_index = torch.min(flip_rate_rmse, dim=0)
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
                print(f'time {time.time()-opt_start_time:.3f}, iteration {iteration+1}, best beta min {best_beta.min():.3g}, mean {best_beta.mean():.3g}, max {best_beta.max():.3g}, beta range min {range_width.min():.3g}, mean {range_width.mean():.3g}, max {range_width.max():.3g}, flip rate RMSE min {flip_rate_rmse.min():.3g}, mean {flip_rate_rmse.mean():.3g}, max {flip_rate_rmse.max():.3g}')
            num_updates_completed += 1
            # Stop if we have homed in on the best possible beta to within machine precision.
            if range_width.max() == 0:
                if verbose:
                    print('betas have all converged to a single value per subject')
                break
        return num_updates_completed
    
    # target_state_product_mean is of size 1 x num_subjects x num_nodes x num_nodes.
    def fit_by_simulation(self, target_state_mean:torch.Tensor, target_state_product_mean:torch.Tensor, num_updates:int, steps_per_update:int=50, learning_rate:float=0.001, verbose:bool=False):
        target_state_mean = target_state_mean.unsqueeze(dim=0)
        target_state_product_mean = target_state_product_mean.unsqueeze(dim=0)
        # Give the target an additional dimension so that we can broadcast it to all replica models of the same target.
        for update in range(num_updates):
            sim_state_mean, sim_state_product_mean = self.simulate_and_record_means(num_steps=steps_per_update)
            # Since the product of any state with itself is +1, the diagonals of both the target and sim product means are 1.
            # Consequently, the difference between the diagonals is 0, leading to no change in J, which is initialized to 0.
            # This is intentional, since no node should have a coupling to itsef.
            self.h += learning_rate * (target_state_mean - sim_state_mean)
            self.J += learning_rate * (target_state_product_mean - sim_state_product_mean)
            if verbose:
                rmse_state_mean = get_pairwise_rmse(mat1=target_state_mean, mat2=sim_state_mean)
                rmse_state_product_mean = get_pairwise_rmse_2d(mat1=target_state_product_mean, mat2=sim_state_product_mean)
                print(f'update {update+1}, state mean RMSE  min {rmse_state_mean.min():.3g}, mean {rmse_state_mean.mean():.3g}, max {rmse_state_mean.max():.3g}, state product mean RMSE min {rmse_state_product_mean.min():.3g}, mean {rmse_state_product_mean.mean():.3g}, max {rmse_state_product_mean.max():.3g}')
    
    # target_state_product_mean is of size 1 x num_subjects x num_nodes x num_nodes.
    # @torch.compile
    def fit_by_simulation_faster(self, target_state_mean:torch.Tensor, target_state_product_mean:torch.Tensor, num_updates:int, steps_per_update:int=50, learning_rate:float=0.001, verbose:bool=False):
        opt_start_time = time.time()
        # Give the target an additional dimension so that we can broadcast it to all replica models of the same target.
        target_state_mean = target_state_mean.unsqueeze(dim=0)
        target_state_product_mean = target_state_product_mean.unsqueeze(dim=0)
        # Pre-allocate space for the sums that we use to calculate the means.
        rand_choice = torch.zeros_like(self.s)
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        num_nodes = self.s.size(dim=-1)
        # We do not update beta in this method, so we only need to do this once.
        neg_over_beta_3d = -1 / self.beta.unsqueeze(dim=-1)
        for update in range(num_updates):
            s_sum.zero_()
            s_product_sum.zero_()
            h2 = 2 * self.h
            # sim_start_time = time.time()
            for _ in range(steps_per_update):
                torch.mul( neg_over_beta_3d, torch.rand_like(input=self.s).log(), out=rand_choice )
                for node in range(num_nodes):
                    # self.s[:,:,node] = get_next_state_preindexed(s_i=self.s[:,:,node], s=self.s, h2_i=h2[:,:,node], J_i=self.J[:,:,node,:], log_rand_choice_over_beta_i=rand_choice[:,:,node])
                    # self.s[:,:,node] = get_next_state(s=self.s, h2=h2, J=self.J, log_rand_choice_over_beta=rand_choice, node_index=node)
                    self.s[:,:,node] *= (   1.0 - 2.0*(  rand_choice[:,:,node] > ( self.s[:,:,node] * ( h2[:,:,node] + torch.sum(self.J[:,:,node,:] * self.s, dim=-1) ) )  ).float()   )
                s_sum += self.s# B x N x 1
                torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
                s_product_sum += s_product
            # print(f'sim time {time.time()-sim_start_time:.3f}')
            # param_update_start_time = time.time()
            s_sum /= steps_per_update
            s_product_sum /= steps_per_update
            # Since the product of any state with itself is +1, the diagonals of both the target and sim product means are 1.
            # Consequently, the difference between the diagonals is 0, leading to no change in J, which is initialized to 0.
            # This is intentional, since no node should have a coupling to itsef.
            self.h += learning_rate * (target_state_mean - s_sum)
            self.J += learning_rate * (target_state_product_mean - s_product_sum)
            # print(f'param update time {time.time()-param_update_start_time:.3f}')
            if verbose:
                rmse_state_mean = get_pairwise_rmse(mat1=target_state_mean, mat2=s_sum)
                rmse_state_product_mean = get_pairwise_rmse_2d(mat1=target_state_product_mean, mat2=s_product_sum)
                print(f'time {time.time()-opt_start_time:.3f}, update {update+1}, state mean RMSE  min {rmse_state_mean.min():.3g}, mean {rmse_state_mean.mean():.3g}, max {rmse_state_mean.max():.3g}, state product mean RMSE min {rmse_state_product_mean.min():.3g}, mean {rmse_state_product_mean.mean():.3g}, max {rmse_state_product_mean.max():.3g}')
    
    # target_state_product_mean is of size 1 x num_subjects x num_nodes x num_nodes.
    # @torch.compile
    def fit_by_simulation_pmb(self, target_state_mean:torch.Tensor, target_state_product_mean:torch.Tensor, num_updates:int, steps_per_update:int=50, learning_rate:float=0.001, verbose:bool=False):
        opt_start_time = time.time()
        # Give the target an additional dimension so that we can broadcast it to all replica models of the same target.
        target_state_mean = target_state_mean.unsqueeze(dim=0)
        target_state_product_mean = target_state_product_mean.unsqueeze(dim=0)
        # Pre-allocate space for the sums that we use to calculate the means.
        rand_choice = torch.zeros_like(self.s)
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        num_nodes = self.s.size(dim=-1)
        # We do not update beta in this method, so we only need to do this once.
        rand_choice = torch.zeros_like(self.s)
        neg_lr_over_steps = -learning_rate/steps_per_update
        for update in range(num_updates):
            s_sum.zero_()
            s_product_sum.zero_()
            hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
            Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
            # sim_start_time = time.time()
            for _ in range(steps_per_update):
                torch.rand( size=rand_choice.size(), out=rand_choice )
                rand_choice.log_()
                for node in range(num_nodes):
                    self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
                s_sum += self.s# B x N x 1
                torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
                s_product_sum += s_product
            # print(f'sim time {time.time()-sim_start_time:.3f}')
            # param_update_start_time = time.time()
            # Since the product of any state with itself is +1, the diagonals of both the target and sim product means are 1.
            # Consequently, the difference between the diagonals is 0, leading to no change in J, which is initialized to 0.
            # This is intentional, since no node should have a coupling to itsef.
            self.h.add_(other=target_state_mean, alpha=learning_rate)
            self.h.add_(other=s_sum, alpha=neg_lr_over_steps)
            self.J.add_(other=target_state_product_mean, alpha=learning_rate)
            self.J.add_(other=s_product_sum, alpha=neg_lr_over_steps)
            # s_sum /= steps_per_update
            # s_product_sum /= steps_per_update
            # self.h += learning_rate * (target_state_mean - s_sum)
            # self.J += learning_rate * (target_state_product_mean - s_product_sum)
            # print(f'param update time {time.time()-param_update_start_time:.3f}')
            if verbose:
                rmse_state_mean = get_pairwise_rmse(mat1=target_state_mean, mat2=s_sum/steps_per_update)
                rmse_state_product_mean = get_pairwise_rmse_2d(mat1=target_state_product_mean, mat2=s_product_sum/steps_per_update)
                print(f'time {time.time()-opt_start_time:.3f}, update {update+1}, state mean RMSE  min {rmse_state_mean.min():.3g}, mean {rmse_state_mean.mean():.3g}, max {rmse_state_mean.max():.3g}, state product mean RMSE min {rmse_state_product_mean.min():.3g}, mean {rmse_state_product_mean.mean():.3g}, max {rmse_state_product_mean.max():.3g}')
    
    # target_state_product_mean is of size 1 x num_subjects x num_nodes x num_nodes.
    # learning_rate is a Tensor of size num_reps x num_targets, same as beta.
    # @torch.compile
    def fit_by_simulation_pmb_multi_learning_rate(self, target_state_mean:torch.Tensor, target_state_product_mean:torch.Tensor, num_updates:int, steps_per_update:int, learning_rate:torch.Tensor, verbose:bool=False):
        opt_start_time = time.time()
        # Give the target an additional dimension so that we can broadcast it to all replica models of the same target.
        scaled_target_state_mean = learning_rate.unsqueeze(dim=-1) * target_state_mean.unsqueeze(dim=0)
        scaled_target_state_product_mean = learning_rate.unsqueeze(dim=-1).unsqueeze(dim=-1) * target_state_product_mean.unsqueeze(dim=0)
        # Pre-allocate space for the sums that we use to calculate the means.
        rand_choice = torch.zeros_like(self.s)
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        num_nodes = self.s.size(dim=-1)
        # We do not update beta in this method, so we only need to do this once.
        rand_choice = torch.zeros_like(self.s)
        neg_lr_over_steps = -learning_rate/steps_per_update
        neg_lr_over_steps_nodewise = neg_lr_over_steps.unsqueeze(dim=-1)
        neg_lr_over_steps_pairwise = neg_lr_over_steps_nodewise.unsqueeze(dim=-1)
        for update in range(num_updates):
            s_sum.zero_()
            s_product_sum.zero_()
            hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
            Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
            # sim_start_time = time.time()
            for _ in range(steps_per_update):
                torch.rand( size=rand_choice.size(), out=rand_choice )
                rand_choice.log_()
                for node in range(num_nodes):
                    self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
                s_sum += self.s# B x N x 1
                torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
                s_product_sum += s_product
            # print(f'sim time {time.time()-sim_start_time:.3f}')
            # param_update_start_time = time.time()
            # Since the product of any state with itself is +1, the diagonals of both the target and sim product means are 1.
            # Consequently, the difference between the diagonals is 0, leading to no change in J, which is initialized to 0.
            # This is intentional, since no node should have a coupling to itsef.
            s_sum *= neg_lr_over_steps_nodewise
            s_sum += scaled_target_state_mean
            self.h += s_sum
            s_product_sum *= neg_lr_over_steps_pairwise
            s_product_sum += scaled_target_state_product_mean
            self.J += s_product_sum
            # s_sum /= steps_per_update
            # s_product_sum /= steps_per_update
            # self.h += learning_rate * (target_state_mean - s_sum)
            # self.J += learning_rate * (target_state_product_mean - s_product_sum)
            # print(f'param update time {time.time()-param_update_start_time:.3f}')
            if verbose:
                rmse_state_mean = get_pairwise_rmse(mat1=target_state_mean, mat2=s_sum/steps_per_update)
                rmse_state_product_mean = get_pairwise_rmse_2d(mat1=target_state_product_mean, mat2=s_product_sum/steps_per_update)
                print(f'time {time.time()-opt_start_time:.3f}, update {update+1}, state mean RMSE  min {rmse_state_mean.min():.3g}, mean {rmse_state_mean.mean():.3g}, max {rmse_state_mean.max():.3g}, state product mean RMSE min {rmse_state_product_mean.min():.3g}, mean {rmse_state_product_mean.mean():.3g}, max {rmse_state_product_mean.max():.3g}')
    
    # target_state_product_mean is of size 1 x num_subjects x num_nodes x num_nodes.
    # @torch.compile
    def fit_by_simulation_and_save(self, target_state_mean:torch.Tensor, target_state_product_mean:torch.Tensor, save_file:str, num_saves:int=1000, updates_per_save:int=1000, steps_per_update:int=50, learning_rate:float=0.001, verbose:bool=False):
        opt_start_time = time.time()
        # Give the target an additional dimension so that we can broadcast it to all replica models of the same target.
        target_state_mean = target_state_mean.unsqueeze(dim=0)
        target_state_product_mean = target_state_product_mean.unsqueeze(dim=0)
        # Pre-allocate space for the sums that we use to calculate the means.
        rand_choice = torch.zeros_like(self.s)
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        num_nodes = self.s.size(dim=-1)
        # We do not update beta in this method, so we only need to do this once.
        neg_over_beta_3d = -1 / self.beta.unsqueeze(dim=-1)
        for save in range(num_saves):
            for update in range(updates_per_save):
                s_sum.zero_()
                s_product_sum.zero_()
                h2 = 2 * self.h
                # sim_start_time = time.time()
                for _ in range(steps_per_update):
                    torch.mul( neg_over_beta_3d, torch.rand_like(input=self.s).log(), out=rand_choice )
                    for node in range(num_nodes):
                        # self.s[:,:,node] = get_next_state_preindexed(s_i=self.s[:,:,node], s=self.s, h2_i=h2[:,:,node], J_i=self.J[:,:,node,:], log_rand_choice_over_beta_i=rand_choice[:,:,node])
                        # self.s[:,:,node] = get_next_state(s=self.s, h2=h2, J=self.J, log_rand_choice_over_beta=rand_choice, node_index=node)
                        self.s[:,:,node] *= (   1.0 - 2.0*(  rand_choice[:,:,node] > ( self.s[:,:,node] * ( h2[:,:,node] + torch.sum(self.J[:,:,node,:] * self.s, dim=-1) ) )  ).float()   )
                    s_sum += self.s# B x N x 1
                    torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
                    s_product_sum += s_product
                # print(f'sim time {time.time()-sim_start_time:.3f}')
                # param_update_start_time = time.time()
                s_sum /= steps_per_update
                s_product_sum /= steps_per_update
                # Since the product of any state with itself is +1, the diagonals of both the target and sim product means are 1.
                # Consequently, the difference between the diagonals is 0, leading to no change in J, which is initialized to 0.
                # This is intentional, since no node should have a coupling to itsef.
                self.h += learning_rate * (target_state_mean - s_sum)
                self.J += learning_rate * (target_state_product_mean - s_product_sum)
                # print(f'param update time {time.time()-param_update_start_time:.3f}')
                if verbose:
                    rmse_state_mean = get_pairwise_rmse(mat1=target_state_mean, mat2=s_sum)
                    rmse_state_product_mean = get_pairwise_rmse_2d(mat1=target_state_product_mean, mat2=s_product_sum)
                    print(f'time {time.time()-opt_start_time:.3f}, update {update+1}, state mean RMSE  min {rmse_state_mean.min():.3g}, mean {rmse_state_mean.mean():.3g}, max {rmse_state_mean.max():.3g}, state product mean RMSE min {rmse_state_product_mean.min():.3g}, mean {rmse_state_product_mean.mean():.3g}, max {rmse_state_product_mean.max():.3g}')
            torch.save(obj=self, f=save_file)
            if verbose:
                print(f'save {save+1} to {save_file}')

    # target_state_product_mean is of size num_subjects x num_nodes x num_nodes.
    def fit_by_pseudolikelihood_step_by_step(self, num_updates:int, target_ts:torch.Tensor, target_state_means:torch.Tensor=None, target_state_product_means:torch.Tensor=None, learning_rate:float=0.001, get_means_step_by_step:bool=False, verbose:bool=True):
        if type(target_state_product_means) == type(None):
            if get_means_step_by_step:
                target_state_means, target_state_product_means = get_time_series_mean_step_by_step(target_ts)
            else:
                target_state_means, target_state_product_means = get_time_series_mean(target_ts)
        elif type(target_state_means) == type(None):
            target_state_means = target_ts.mean(dim=-1)
        target_state_means = target_state_means[None,:,:]
        target_state_product_means = target_state_product_means[None,:,:,:]
        num_time_points = target_ts.size(dim=-1)
        model_state_sum = torch.zeros_like(self.h)
        model_state_product_sum = torch.zeros_like(self.J)
        if verbose:
            start_time = time.time()
            print('starting pseudo-likelihood maximization')
        for update in range(num_updates):
            model_state_sum.zero_()
            model_state_product_sum.zero_()
            for t in range(num_time_points):
                state = target_ts[:,:,t].unsqueeze(dim=0)
                # mean_field = torch.tanh( self.h + torch.matmul(self.J, state) )
                mean_field = torch.tanh( self.h + torch.sum(self.J * state[:,:,None,:], dim=-1) )
                model_state_sum += mean_field
                # model_state_product_sum += torch.matmul( state, mean_field.transpose(dim0=-2, dim1=-1) )
                model_state_product_sum += (state[:,:,:,None] * mean_field[:,:,None,:])
            h_update = target_state_means - model_state_sum/num_time_points
            J_update = target_state_product_means - model_state_product_sum/num_time_points
            self.h += learning_rate * h_update
            self.J += learning_rate * J_update
            if verbose:
                print(f'time {time.time()-start_time:.3f}, {update}, h_update range [{h_update.min():.3g}, {h_update.max():.3g}], J_update range [{J_update.min():.3g}, {J_update.max():.3g}]')

    # target_state_product_mean is of size num_subjects x num_nodes x num_nodes.
    def fit_by_pseudolikelihood_block_by_block(self, num_updates:int, target_ts:torch.Tensor, target_state_means:torch.Tensor=None, target_state_product_means:torch.Tensor=None, learning_rate:float=0.001, get_means_step_by_step:bool=False, block_length:int=1200, verbose:bool=True):
        if type(target_state_product_means) == type(None):
            if get_means_step_by_step:
                target_state_means, target_state_product_means = get_time_series_mean_step_by_step(target_ts)
            else:
                target_state_means, target_state_product_means = get_time_series_mean(target_ts)
        elif type(target_state_means) == type(None):
            target_state_means = target_ts.mean(dim=-1)
        target_state_means = target_state_means[None,:,:]
        target_state_product_means = target_state_product_means[None,:,:,:]
        num_time_points = target_ts.size(dim=-1)
        num_blocks = num_time_points//block_length
        num_time_points_used = num_blocks * block_length
        h_update = torch.zeros_like(self.h)
        J_update = torch.zeros_like(self.J)
        num_models, num_subjects, num_nodes = self.h.size()
        mean_field = torch.zeros( (num_models, num_subjects, num_nodes, block_length), dtype=self.h.dtype, device=self.h.device )
        state_cross_mean_field = torch.zeros_like(self.J)
        # J_t = self.J.unsqueeze(dim=-1)
        h_t = self.h.unsqueeze(dim=-1)
        if verbose:
            start_time = time.time()
            print('starting pseudo-likelihood maximization')
        for update in range(num_updates):
            h_update.zero_()
            J_update.zero_()
            block_start = 0
            block_end = block_length
            for _ in range(num_blocks):
                state_block = target_ts[None,:,:,block_start:block_end]
                torch.matmul(self.J, state_block, out=mean_field)
                mean_field += h_t
                mean_field.tanh_()
                # mean_field = torch.tanh( h_t + torch.matmul(self.J, state_block) )
                # mean_field = torch.tanh( h_t + torch.sum(J_t * state_block[:,:,None,:,:], dim=-1) )
                h_update += mean_field.sum(dim=-1)
                torch.matmul( state_block, mean_field.transpose(dim0=-2, dim1=-1), out=state_cross_mean_field )
                J_update += state_cross_mean_field
                # model_state_product_sum += (state_block[:,:,:,None,:] * mean_field[:,:,None,:,:]).sum(dim=-1)
                block_start += block_length
                block_end += block_length
            h_update /= num_time_points_used
            h_update -= target_state_means
            h_update *= learning_rate
            self.h -= h_update
            # h_update = target_state_means - model_state_sum/num_time_points_used
            J_update /= num_time_points_used
            J_update -= target_state_product_means
            J_update *= learning_rate
            self.J -= J_update
            # J_update = target_state_product_means - model_state_product_sum/num_time_points_used
            if verbose:
                print(f'time {time.time()-start_time:.3f}, {update}, h_update range [{h_update.min():.3g}, {h_update.max():.3g}], J_update range [{J_update.min():.3g}, {J_update.max():.3g}]')
    
    # target_state_product_mean is of size num_subjects x num_nodes x num_nodes.
    def fit_by_pseudolikelihood_model_by_model(self, num_updates:int, target_ts:torch.Tensor, target_state_means:torch.Tensor=None, target_state_product_means:torch.Tensor=None, learning_rate:float=0.001, get_means_step_by_step:bool=False, verbose:bool=True):
        if type(target_state_product_means) == type(None):
            if get_means_step_by_step:
                target_state_means, target_state_product_means = get_time_series_mean_step_by_step(target_ts)
            else:
                target_state_means, target_state_product_means = get_time_series_mean(target_ts)
        elif type(target_state_means) == type(None):
            target_state_means = target_ts.mean(dim=-1)
        num_models, num_subjects, _ = self.s.size()
        mean_field = torch.zeros_like(target_ts[0,:,:])
        start_time = time.time()
        J_diff = torch.zeros_like( self.J[0,0,:,:] )
        num_steps = target_ts.size(dim=-1)
        if verbose:
            print('starting pseudolikelihood maximization...')
        for subject in range(num_subjects):
            current_ts = target_ts[subject,:,:]
            current_mean_state = target_state_means[subject,:].unsqueeze(dim=-1)
            current_mean_state_product = target_state_product_means[subject,:,:]
            for model in range(num_models):
                current_h = self.h[model,subject,:].unsqueeze(dim=-1)
                current_J = self.J[model,subject,:,:]
                for update in range(num_updates):
                    torch.addmm( input=current_h.expand( size=(-1,num_steps) ), mat1=current_J, mat2=current_ts, beta=1, alpha=1, out=mean_field )
                    mean_field.tanh_()
                    h_diff = current_mean_state - mean_field.mean(dim=-1, keepdim=True)
                    current_h.add_(other=h_diff, alpha=learning_rate)
                    torch.addmm(input=current_mean_state_product, mat1=mean_field, mat2=current_ts.transpose(dim0=0,dim1=1), beta=1, alpha=-1/num_steps, out=J_diff )
                    J_diff -= torch.diag_embed( input=torch.diagonal(input=J_diff, offset=0, dim1=-2, dim2=-1), offset=0, dim1=-2, dim2=-1 )
                    current_J.add_(other=J_diff, alpha=learning_rate)
                    if verbose:
                        print(f'model {model}, subject {subject}, time {time.time()-start_time:.3f}, update {update+1}, h_diff min {h_diff.min():.3g}, mean {h_diff.mean():.3g}, max {h_diff.max():.3g}, J diff min {J_diff.min():.3g}, mean {J_diff.mean():.3g}, max {J_diff.max():.3g}')
                        # print(f'model {model}, subject {subject}, time {time.time()-start_time:.3f}, update {update+1}, h_diff min {h_diff.min():.3g}, mean {h_diff.mean():.3g}, max {h_diff.max():.3g}, J diff min {J_diff.min():.3g}, mean {J_diff.mean():.3g}, max {J_diff.max():.3g}')
    
        # target_state_product_mean is of size num_subjects x num_nodes x num_nodes.
    def fit_by_pseudolikelihood_model_blocks(self, num_updates:int, target_ts:torch.Tensor, target_state_means:torch.Tensor=None, target_state_product_means:torch.Tensor=None, learning_rate:float=0.001, get_means_step_by_step:bool=False, block_length:int=1, verbose:bool=True):
        if type(target_state_product_means) == type(None):
            if get_means_step_by_step:
                target_state_means, target_state_product_means = get_time_series_mean_step_by_step(target_ts)
            else:
                target_state_means, target_state_product_means = get_time_series_mean(target_ts)
        elif type(target_state_means) == type(None):
            target_state_means = target_ts.mean(dim=-1)
        num_models, num_subjects, num_nodes = self.s.size()
        num_steps = target_ts.size(dim=-1)
        num_model_blocks = num_models//block_length
        mean_field = torch.zeros( size=(block_length, num_nodes, num_steps), dtype=target_ts.dtype, device=target_ts.device )
        target_state_means *= learning_rate
        target_state_product_means *= learning_rate
        neg_lr_over_steps = -learning_rate/target_ts.size(dim=-1)
        if verbose:
            start_time = time.time()
            print('starting pseudolikelihood maximization...')
        for subject in range(num_subjects):
            current_ts = target_ts[subject,:,:].unsqueeze(dim=0).expand( size=(block_length,-1,-1) )# B x N x T
            current_mean_state = target_state_means[subject,:].unsqueeze(dim=-1).unsqueeze(dim=0).expand( size=(block_length,-1,-1) )# B x N x 1
            current_mean_state_product = target_state_product_means[subject,:,:].expand( size=(block_length,-1,-1) )# B x N x N
            block_start = 0
            block_end = block_length
            for block in range(num_model_blocks):
                current_h = self.h[block_start:block_end,subject,:].unsqueeze(dim=-1)# B x N x 1
                current_J = self.J[block_start:block_end,subject,:,:]# B x N x N
                for _ in range(num_updates):
                    torch.bmm(input=current_J, mat2=current_ts, out=mean_field)
                    mean_field += current_h
                    mean_field.tanh_()
                    current_h += current_mean_state
                    current_h.add_( other=mean_field.sum(dim=-1, keepdim=True), alpha=neg_lr_over_steps )
                    current_J += current_mean_state_product
                    current_J.baddbmm_( batch1=current_ts, batch2=mean_field.transpose(dim0=-2,dim1=-1), beta=1, alpha=neg_lr_over_steps )
                    # print(f'model block {block+1}, from {block_start} to {block_end}, subject {subject}, time {time.time()-start_time:.3f}, update {update+1}, h min {current_h.min():.3g}, mean {current_h.mean():.3g}, max {current_h.max():.3g}, J min {current_J.min():.3g}, mean {current_J.mean():.3g}, max {current_J.max():.3g}')
                if verbose:
                    print(f'model block {block+1}, from {block_start} to {block_end}, subject {subject}, time {time.time()-start_time:.3f}, h min {current_h.min():.3g}, mean {current_h.mean():.3g}, max {current_h.max():.3g}, J min {current_J.min():.3g}, mean {current_J.mean():.3g}, max {current_J.max():.3g}')
                block_start += block_length
                block_end += block_length

    # target_state_product_mean is of size num_subjects x num_nodes x num_nodes.
    def fit_by_pseudolikelihood(self, num_updates:int, target_ts:torch.Tensor, target_state_means:torch.Tensor=None, target_state_product_means:torch.Tensor=None, learning_rate:float=0.001, get_means_step_by_step:bool=False):
        num_time_points = target_ts.size(dim=-1)
        if type(target_state_product_means) == type(None):
            if get_means_step_by_step:
                target_state_means, target_state_product_means = get_time_series_mean_step_by_step(target_ts)
            else:
                target_state_means, target_state_product_means = get_time_series_mean(target_ts)
        elif type(target_state_means) == type(None):
            target_state_means = target_ts.mean(dim=-1)
        target_ts = target_ts.unsqueeze(dim=0)
        target_state_means = target_state_means.unsqueeze(dim=0)
        target_state_product_means = target_state_product_means.unsqueeze(dim=0)
        for _ in range(num_updates):
            mean_field = torch.tanh( self.h.unsqueeze(dim=-1) + torch.matmul(self.J, target_ts) )
            self.h += learning_rate * ( target_state_means - torch.mean(input=mean_field, dim=-1) )
            self.J += learning_rate * (  target_state_product_means - torch.matmul( target_ts, mean_field.transpose(dim0=-2, dim1=-1) )/num_time_points  )
    
    # Run a simulation, and record and return the Fisher Information Matrix (FIM) estimated from the observed states.
    # Since the FIM is symmetric, we only retain the upper triangular part, including the diagonal.
    def simulate_and_record_fim(self, num_steps:int):
        dtype = self.s.dtype
        device = self.s.device
        models_per_subject, num_subjects, num_nodes = self.s.size()
        state_triu_rows, state_triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=device)
        # Unlike with the products matrix, the diagonal of the FIM is meaningful.
        num_observables = num_nodes + state_triu_rows.numel()
        fim_triu_indices = torch.triu_indices(row=num_observables, col=num_observables, offset=0, dtype=int_type, device=device)
        fim_triu_rows = fim_triu_indices[0]
        fim_triu_cols = fim_triu_indices[1]
        num_fim_triu_elements = fim_triu_rows.numel()
        observables = torch.zeros( (models_per_subject, num_subjects, num_observables), dtype=dtype, device=device )
        observables_mean = torch.zeros_like(observables)
        observable_product_sum = torch.zeros( (models_per_subject, num_subjects, num_fim_triu_elements), dtype=dtype, device=device )
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            observables[:,:,:num_nodes] = self.s
            observables[:,:,num_nodes:] = self.s[:,:,state_triu_rows] * self.s[:,:,state_triu_cols]
            observables_mean += observables
            observable_product_sum += observables[:,:,fim_triu_rows] * observables[:,:,fim_triu_cols]
        observables_mean /= num_steps
        observable_product_sum /= num_steps
        observable_product_sum -= (observables_mean[:,:,fim_triu_rows] * observables_mean[:,:,fim_triu_cols])
        return observable_product_sum
    
    # Run a simulation, and record and return the Fisher Information Matrix (FIM) estimated from the observed states for one of the models.
    # We return the fim as a square matrix, because maintaining the upper triangular matrix of values plus the two upper triangular indexing variables takes up more memory than just retaining a square matrix.
    def simulate_and_record_one_fim(self, num_steps:int, model_index:int, subject_index:int):
        dtype = self.s.dtype
        device = self.s.device
        num_nodes = self.s.size(dim=-1)
        state_triu_rows, state_triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=device)
        # Unlike with the products matrix, the diagonal of the FIM is meaningful.
        num_observables = num_nodes + state_triu_rows.numel()
        observables = torch.zeros( (num_observables,), dtype=dtype, device=device )
        observables_mean = torch.zeros_like(observables)
        observable_product_sum = torch.zeros( (num_observables, num_observables), dtype=dtype, device=device )
        rand_choice = torch.zeros_like(self.s)
        neg_over_beta_3d = -1 / self.beta.unsqueeze(dim=-1)
        h2 = 2 * self.h
        for _ in range(num_steps):
            torch.mul( neg_over_beta_3d, torch.rand_like(input=self.s).log(), out=rand_choice )
            for node in range(num_nodes):
                self.s[:,:,node] *= (   1.0 - 2.0*(  rand_choice[:,:,node] > ( self.s[:,:,node] * ( h2[:,:,node] + torch.sum(self.J[:,:,node,:] * self.s, dim=-1) ) )  ).float()   )
            model_state = self.s[model_index,subject_index,:]
            observables[:num_nodes] = model_state
            observables[num_nodes:] = model_state[state_triu_rows] * model_state[state_triu_cols]
            observables_mean += observables
            observable_product_sum += observables[:,None] * observables[None,:]
        observables_mean /= num_steps
        observable_product_sum /= num_steps
        observable_product_sum -= (observables_mean[:,None] * observables_mean[None,:])
        return observable_product_sum
    
    # Simulate the Ising model.
    # Record the means of the states and products of states.
    # Record the means of state products in a square matrix.
    # This is faster but includes redundant elements that take up memory.
    def simulate_and_record_flip_rate(self, num_steps:int):
        s_previous = self.s.clone()
        flip_count = torch.zeros_like(self.s)
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            flip_count += (self.s != s_previous).float()# B x N x 1
            s_previous[:,:,:] = self.s
        flip_count /= num_steps
        return flip_count
    
    def simulate_and_record_means_and_flip_rate(self, num_steps:int):
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        s_previous = self.s.clone()
        flip_count = torch.zeros_like(self.s)
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            s_sum += self.s# B x N x 1
            torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            s_product_sum += s_product
            flip_count += (self.s != s_previous).float()# B x N x 1
            s_previous[:,:,:] = self.s
        s_sum /= num_steps
        s_product_sum /= num_steps
        flip_count /= num_steps
        return s_sum, s_product_sum, flip_count
    
    def simulate_and_record_means_and_flip_rate_faster(self, num_steps:int):
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        s_previous = self.s.clone()
        flip_count = torch.zeros_like(self.s)
        rand_choice = torch.zeros_like(self.s)
        neg_over_beta_3d = -1 / self.beta.unsqueeze(dim=-1)
        h2 = 2 * self.h
        num_nodes = self.s.size(dim=-1)
        for _ in range(num_steps):
            torch.mul( neg_over_beta_3d, torch.rand_like(input=self.s).log(), out=rand_choice )
            for node in range(num_nodes):
                self.s[:,:,node] *= (   1.0 - 2.0*(  rand_choice[:,:,node] > ( self.s[:,:,node] * ( h2[:,:,node] + torch.sum(self.J[:,:,node,:] * self.s, dim=-1) ) )  ).float()   )
            s_sum += self.s# B x N x 1
            torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            s_product_sum += s_product
            flip_count += (self.s != s_previous).float()# B x N x 1
            s_previous[:,:,:] = self.s
        s_sum /= num_steps
        s_product_sum /= num_steps
        flip_count /= num_steps
        return s_sum, s_product_sum, flip_count
    
    def simulate_and_record_means_and_flip_rate_pmb(self, num_steps:int):
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        s_previous = self.s.clone()
        flip_count = torch.zeros_like(self.s)
        rand_choice = torch.zeros_like(self.s)
        num_nodes = self.s.size(dim=-1)
        hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
        Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
        for _ in range(num_steps):
            torch.rand( size=rand_choice.size(), out=rand_choice )
            rand_choice.log_()
            for node in range(num_nodes):
                self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
            s_sum += self.s# B x N x 1
            torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            s_product_sum += s_product
            flip_count += (self.s != s_previous).float()# B x N x 1
            s_previous[:,:,:] = self.s
        s_sum /= num_steps
        s_product_sum /= num_steps
        flip_count /= num_steps
        return s_sum, s_product_sum, flip_count
    
    def simulate_and_record_time_series_pmb(self, num_steps:int):
        num_reps, num_targets, num_nodes = self.s.size()
        time_series = torch.zeros( size=(num_reps, num_targets, num_nodes, num_steps), dtype=self.s.dtype, device=self.s.device )
        rand_choice = torch.zeros_like(self.s)
        hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
        Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
        for t in range(num_steps):
            torch.rand( size=rand_choice.size(), out=rand_choice )
            rand_choice.log_()
            for node in range(num_nodes):
                self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
            time_series[:,:,:,t] = self.s
        return time_series
    
    def simulate_and_record_means_pmb(self, num_steps:int):
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        rand_choice = torch.zeros_like(self.s)
        num_nodes = self.s.size(dim=-1)
        hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
        Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
        for _ in range(num_steps):
            torch.rand( size=rand_choice.size(), out=rand_choice )
            rand_choice.log_()
            for node in range(num_nodes):
                self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
            s_sum += self.s# B x N x 1
            torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            s_product_sum += s_product
        s_sum /= num_steps
        s_product_sum /= num_steps
        return s_sum, s_product_sum
    
    def simulate_and_record_magnetization(self, num_steps:int):
        num_models, num_subjects, num_nodes = self.s.size()
        magnetization = torch.zeros( size=(num_models, num_subjects, num_steps), dtype=self.s.dtype, device=self.s.device )
        rand_choice = torch.zeros_like(self.s)
        hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
        Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
        for step in range(num_steps):
            torch.rand( size=rand_choice.size(), out=rand_choice )
            rand_choice.log_()
            for node in range(num_nodes):
                self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
            magnetization[:,:,step] = self.s.mean(dim=-1, keepdim=False)
        return magnetization

    def simulate_and_record_avalanche_counts(self, num_steps:int):
        s_previous = self.s.clone()
        rand_choice = torch.zeros_like(self.s)
        neg_over_beta_3d = -1 / self.beta.unsqueeze(dim=-1)
        h2 = 2 * self.h
        models_per_subject, num_subjects, num_nodes = self.s.size()
        avalanche_counts = torch.zeros( size=(models_per_subject, num_subjects, num_nodes+1), dtype=self.s.dtype, device=self.s.device )
        for _ in range(num_steps):
            torch.mul( neg_over_beta_3d, torch.rand_like(input=self.s).log(), out=rand_choice )
            for node in range(num_nodes):
                self.s[:,:,node] *= (   1.0 - 2.0*(  rand_choice[:,:,node] > ( self.s[:,:,node] * ( h2[:,:,node] + torch.sum(self.J[:,:,node,:] * self.s, dim=-1) ) )  ).float()   )
            num_flips = torch.count_nonzero(self.s != s_previous, dim=-1)
            for model_index in range(models_per_subject):
                for subject_index in range(num_subjects):
                    avalanche_counts[ model_index, subject_index, num_flips[model_index, subject_index] ] += 1
            s_previous[:,:,:] = self.s
        return avalanche_counts

    def simulate_and_record_point_process_stats_pmb(self, num_steps:int):
        s_previous = self.s.clone()
        # print( 's_previous size', s_previous.size() )
        rand_choice = torch.zeros_like(self.s)
        # print( 'rand_choice size', rand_choice.size() )
        num_ts, num_subjects, num_nodes = self.s.size()
        max_duration = num_steps-1
        max_size = max_duration * num_nodes
        num_activations_previous = torch.zeros( size=(num_ts, num_subjects), dtype=int_type, device=self.s.device )
        # print( 'num_activations_previous size', num_activations_previous.size() )
        is_in_gap_previous = torch.ones( size=(num_ts, num_subjects), dtype=torch.bool, device=self.s.device )
        # print( 'is_in_gap_previous size', is_in_gap_previous.size() )
        is_in_avalanche_previous = torch.logical_not(is_in_gap_previous)
        # print( 'is_in_avalanche_previous size', is_in_avalanche_previous.size() )
        gap_start_time = torch.zeros( size=(num_ts, num_subjects), dtype=int_type, device=self.s.device )
        # print( 'gap_start_time size', gap_start_time.size() )
        avalanche_start_time = torch.zeros_like(gap_start_time)
        # print( 'avalanche_start_time size', avalanche_start_time.size() )
        avalanche_in_progress_size = torch.zeros_like(gap_start_time)
        # print( 'avalanche_in_progress_size size', avalanche_in_progress_size.size() )
        # all_durations = torch.arange(start=1, end=max_duration+1, dtype=int_type, device=self.s.device).unsqueeze(dim=-1)
        # print( 'all_durations size', all_durations.size() )
        # all_sizes = torch.arange(start=1, end=max_size+1, dtype=int_type, device=self.s.device).unsqueeze(dim=-1)
        # print( 'all_sizes size', all_sizes.size() )
        branching_parameter = torch.zeros( size=(num_ts, num_subjects), dtype=float_type, device=self.s.device )
        # print( 'branching_parameter size', branching_parameter.size() )
        gap_duration_count = torch.zeros( size=(max_duration,), dtype=int_type, device=self.s.device )
        # print( 'gap_duration_count size', gap_duration_count.size() )
        avalanche_duration_count = torch.zeros_like(gap_duration_count)
        # print( 'avalanche_duration_count size', avalanche_duration_count.size() )
        avalanche_size_count = torch.zeros( size=(max_size,), dtype=int_type, device=self.s.device )
        # print( 'avalanche_size_count size', avalanche_size_count.size() )
        avalanche_mean_size_for_duration = torch.zeros( size=(max_duration,), dtype=float_type, device=self.s.device )
        # print( 'avalanche_mean_size_for_duration size', avalanche_mean_size_for_duration.size() )
        hbeta = -2 * self.beta.unsqueeze(dim=-1) * self.h
        # print( 'hbeta size', hbeta.size() )
        Jbeta = -2 * self.beta.unsqueeze(dim=-1).unsqueeze(dim=-1) * self.J
        # print( 'Jbeta size', Jbeta.size() )
        for step in range(num_steps):
            # print(f'step {step}')
            torch.rand( size=rand_choice.size(), out=rand_choice )
            rand_choice.log_()
            for node in range(num_nodes):
                self.s[:,:,node] *= (    1.0 - 2.0*(   rand_choice[:,:,node] < (  self.s[:,:,node] * ( hbeta[:,:,node] + torch.sum(Jbeta[:,:,node,:] * self.s, dim=-1) )  )   ).float()    )
            num_activations = torch.count_nonzero(self.s > s_previous, dim=-1)# B x N x 1
            # print(f'num_activations range [{num_activations.min()} {num_activations.max()}]')
            # print( 'num_activations size', num_activations.size() )
            is_in_gap = num_activations == 0
            # print(f'num in gap {torch.count_nonzero(is_in_gap)}')
            # print( 'is_in_gap size', is_in_gap.size() )
            is_in_avalanche = torch.logical_not(is_in_gap)
            # print(f'num in avalanche {torch.count_nonzero(is_in_avalanche)}')
            # print( 'is_in_avalanche size', is_in_avalanche.size() )
            is_gap_start = torch.logical_and(is_in_avalanche_previous, is_in_gap)
            # print(f'num at gap start {torch.count_nonzero(is_gap_start)}')
            # print( 'is_gap_start size', is_gap_start.size() )
            is_avalanche_start = torch.logical_and(is_in_gap_previous, is_in_avalanche)
            # print(f'num at avalanche start {torch.count_nonzero(is_avalanche_start)}')
            # print( 'is_avalanche_start size', is_avalanche_start.size() )
            # Reset the gap start time counter at the start of the gap.
            gap_start_time[is_gap_start] = step
            # Reset the size and start time counters at the start of the avalanche.
            avalanche_start_time[is_avalanche_start] = step
            avalanche_in_progress_size[is_avalanche_start] = 0
            avalanche_in_progress_size += num_activations
            # The start of an avalanche is the end of a gap, so we can now check its duration.
            gap_durations = step - gap_start_time[is_avalanche_start]
            # print( 'gap_durations size', gap_durations.size() )
            # Creating the rectangular boolean Tensor may take up too much memory.
            # gap_duration_count += torch.count_nonzero( all_durations == gap_durations.unsqueeze(dim=0), dim=-1 )
            for gap_duration_index in (gap_durations-1):
                gap_duration_count[gap_duration_index] += 1
            # The start of a gap is the end of an avalanche, so we can now check its duration and size.
            avalanche_durations = step - avalanche_start_time[is_gap_start]
            # print( 'avalanche_durations size', avalanche_durations.size() )
            avalanche_sizes = avalanche_in_progress_size[is_gap_start]
            # is_size and is_duration sometimes take up too much memory.
            # print( 'avalanche_sizes size', avalanche_sizes.size() )
            # is_duration = all_durations == avalanche_durations.unsqueeze(dim=0)
            # print( 'is_duration size', is_duration.size() )
            # is_size = all_sizes == avalanche_sizes.unsqueeze(dim=0)
            # print( 'is_size size', is_size.size() )
            # avalanche_duration_count += torch.count_nonzero(is_duration, dim=-1)
            # avalanche_size_count += torch.count_nonzero(is_size, dim=-1)
            # avalanche_mean_size_for_duration += torch.matmul( is_duration.float(), avalanche_sizes.float().unsqueeze(dim=-1) ).squeeze(dim=-1)
            for (duration_index, size_index, a_size) in zip(avalanche_durations-1, avalanche_sizes-1, avalanche_sizes):
                avalanche_duration_count[duration_index] += 1
                avalanche_size_count[size_index] += 1
                avalanche_mean_size_for_duration[duration_index] += a_size
            branching_parameter[is_in_avalanche_previous] += num_activations[is_in_avalanche_previous]/num_activations_previous[is_in_avalanche_previous]
            s_previous[:,:,:] = self.s
            num_activations_previous[:,:] = num_activations
            is_in_gap_previous[:,:] = is_in_gap
            is_in_avalanche_previous[:,:] = is_in_avalanche
        branching_parameter /= num_steps
        duration_has_count = avalanche_duration_count != 0
        avalanche_mean_size_for_duration[duration_has_count] /= avalanche_duration_count[duration_has_count]
        return branching_parameter, gap_duration_count, avalanche_duration_count, avalanche_size_count, avalanche_mean_size_for_duration
    
    def simulate_and_record_cov_and_flip_rate(self, num_steps:int):
        s_mean, s_product_mean, flip_rate = self.simulate_and_record_means_and_flip_rate(num_steps=num_steps)
        cov = get_cov(state_mean=s_mean, state_product_mean=s_product_mean)
        return cov, flip_rate
    
    def simulate_and_record_fc_and_flip_rate(self, num_steps:int):
        s_mean, s_product_mean, flip_rate = self.simulate_and_record_means_and_flip_rate(num_steps=num_steps)
        fc = get_fc_binary(state_mean=s_mean, state_product_mean=s_product_mean)
        return fc, flip_rate

    # Simulate the Ising model.
    # Record the full time series of states.
    def simulate_and_record_time_series(self, num_steps:int):
        models_per_subject, num_subjects, num_nodes = self.s.size()
        time_series = torch.zeros( size=(models_per_subject, num_subjects, num_nodes, num_steps), dtype=self.s.dtype, device=self.s.device )
        for step in range(num_steps):
            self.do_balanced_metropolis_step()
            time_series[:,:,:,step] = self.s
        return time_series
    
    # Simulate the Ising model.
    # Record the means of the states and products of states.
    # Record the means of state products in a square matrix.
    # This is faster but includes redundant elements that take up memory.
    def simulate_and_record_time_series_and_cov(self, num_steps:int):
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        models_per_subject, num_subjects, num_nodes = self.s.size()
        time_series = torch.zeros( size=(models_per_subject, num_subjects, num_nodes, num_steps), dtype=self.s.dtype, device=self.s.device )
        for step in range(num_steps):
            self.do_balanced_metropolis_step()
            time_series[:,:,:,step] = self.s
            s_sum += self.s# B x N x 1
            torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            s_product_sum += s_product
        cov = get_cov(state_mean=s_sum/num_steps, state_product_mean=s_product_sum/num_steps)
        return time_series, cov
    
    # Simulate the Ising model.
    # Record the means of the states and products of states.
    # Record the means of state products in a square matrix.
    # This is faster but includes redundant elements that take up memory.
    def simulate_and_record_time_series_and_fc(self, num_steps:int, epsilon:float=10e-10):
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        models_per_subject, num_subjects, num_nodes = self.s.size()
        time_series = torch.zeros( size=(models_per_subject, num_subjects, num_nodes, num_steps), dtype=self.s.dtype, device=self.s.device )
        for step in range(num_steps):
            self.do_balanced_metropolis_step()
            time_series[:,:,:,step] = self.s
            s_sum += self.s# B x N x 1
            torch.mul(input=self.s[:,:,:,None], other=self.s[:,:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            s_product_sum += s_product
        fc = get_fc_binary(state_mean=s_sum/num_steps, state_product_mean=s_product_sum/num_steps, epsilon=epsilon)
        return time_series, fc
        
    def get_triu_indices_for_products(self):
        return get_triu_indices_for_products( num_nodes=self.s.size(dim=-1), device=self.s.device )

# @torch.compile
def update_model_state(model:IsingModelLight, log_rand_choice_over_beta:torch.Tensor, node_index:int):
    model.s[:,:,node_index] *= 1.0 - 2.0*(  log_rand_choice_over_beta[:,:,node_index] > ( model.s[:,:,node_index] * ( 2*model.h[:,:,node_index] + torch.sum(model.J[:,:,node_index,:] * model.s, dim=-1) ) )  ).float()

