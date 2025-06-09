# functions we use in multiple Ising-model-related scripts
# based on Sida Chen's code, in turn based on
# Panas, D., Amin, H., Maccione, A., Muthmann, O., van Rossum, M., Berdondini, L., & Hennig, M. H. (2015).
# Sloppiness in spontaneously active neuronal networks. Journal of Neuroscience, 35(22), 8480-8492.
# https://www.jneurosci.org/content/35/22/8480
 
import torch
import math

float_type = torch.float
int_type = torch.int

def num_nodes_to_num_pairs(num_nodes:int):
    return ( num_nodes*(num_nodes-1) )//2

def num_nodes_to_num_params(num_nodes:int):
    return num_nodes + num_nodes_to_num_pairs(num_nodes=num_nodes)

# num_pairs = num_nodes*(num_nodes-1)//2
# We can use the quadratic formula to get back an expression for num_nodes in terms of num_pairs.
def num_pairs_to_num_nodes(num_pairs:int):
    return int(  ( math.sqrt(1 + 8*num_pairs) + 1 )/2  )

# num_params = num_nodes + num_nodes*(num_nodes-1)//2
# We can use the quadratic formula to get back an expression for num_nodes in terms of num_params.
def num_params_to_num_nodes(num_params:int):
    return int(  ( math.sqrt(1 + 8*num_params) - 1 )/2  )

# Create random initial states for batch_size independent Ising models, each with num_nodes nodes. 
def get_random_state(batch_size:int, num_nodes:int, dtype=float_type, device='cpu'):
    s = 2.0*torch.randint( 2, (batch_size, num_nodes), dtype=dtype, device=device ) - 1.0
    return s

# Create a Tensor with the same shape, data type, and device as input but filled with randomly selected -1 and +1 values.
def get_random_state_like(input:torch.Tensor):
    return 2 * torch.randint_like(input=input, high=2) - 1

# Compute the entropy of an Ising model with external fields h, couplings J, and state s.
# s and h can have an arbitrary number of leading batch dimensions but must have the same number of dimensions and same size in the last dimension.
# J should have the number of dimensions in s and h + 1.
# Let s', h', and J' be slices of s, h, and J respectively representing an individual model state, external fields, and couplings.
# s' and h' have num_nodes elements, and J' is num_nodes x num_nodes with J'[i,j] representing the coupling from node j to node i.
# We assume J[i,i] = 0.
def get_entropy(s:torch.Tensor, h:torch.Tensor, J:torch.Tensor):
    return torch.sum(h*s, dim=-1) + 0.5 * torch.sum( J * s.unsqueeze(dim=-2) * s.unsqueeze(dim=-1), dim=(-1,-2) )

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
def get_entropy_change(s:torch.Tensor, h:torch.Tensor, J:torch.Tensor, node:int):
    return -2 * s.index_select(dim=-1, index=node) * (  h.index_select(dim=-1, index=node) + 0.5 * torch.sum( J.index_select(dim=-2, index=node) * s, dim=-1 )  )

# In several places, we want to get the indices for the part of a num_nodes x num_nodes matrix above the diagonal.
# To make the code cleaner, we put the relevant code snippet in this function.
def get_triu_indices_for_products(num_nodes:int, device='cpu'):
    triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, offset=1, dtype=int_type, device=device)
    return triu_indices[0], triu_indices[1]

def square_to_triu_pairs(square_pairs:torch.Tensor):
    num_nodes = square_pairs.size(dim=-1)
    triu_rows, triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=square_pairs.device)
    return square_pairs[:,triu_rows,triu_cols]

# triu_pairs gets filled in above and below the diagonal.
# Use diag_fill to specify a value to fill in along the diagonal.
def triu_to_square_pairs(triu_pairs:torch.Tensor, diag_fill:torch.float=0.0):
    device = triu_pairs.device
    batch_size, num_pairs = triu_pairs.size()
    num_nodes = num_pairs_to_num_nodes(num_pairs=num_pairs)
    square_pairs = torch.full( size=(batch_size, num_nodes, num_nodes), fill_value=diag_fill, dtype=triu_pairs.dtype, device=device )
    triu_rows, triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=square_pairs.device)
    square_pairs[:,triu_rows,triu_cols] = triu_pairs
    square_pairs[:,triu_cols,triu_rows] = triu_pairs
    return square_pairs

def binarize_data_ts(data_ts:torch.Tensor):
    # First, threshold each region time series so that
    # anything below the median maps to -1,
    # anything above the median maps to +1,
    # anything exactly equal to the median maps to 0.
    step_dim = -1
    batch_size, num_nodes, _ = data_ts.size()
    sign_threshold = torch.median(input=data_ts, dim=step_dim, keepdim=True).values
    binary_ts = torch.sign(data_ts - sign_threshold)
    # Next, fill in the 0s so as to make the number of -1s and the number of +1s as nearly equal as possible.
    num_neg = torch.count_nonzero(binary_ts == -1, dim=step_dim)
    num_pos = torch.count_nonzero(binary_ts == 1, dim=step_dim)
    is_zero = binary_ts == 0
    num_zero = torch.count_nonzero(is_zero, dim=step_dim)
    for batch_index in range(batch_size):
        for node_index in range(num_nodes):
            num_zero_here = num_zero[batch_index,node_index]
            num_neg_here = num_neg[batch_index,node_index]
            num_pos_here = num_pos[batch_index,node_index]
            zero_fills = torch.zeros( size=(num_zero_here,), dtype=binary_ts.dtype, device=binary_ts.device )
            for zero_step in range(num_zero_here):
                if num_pos_here < num_neg_here:
                    zero_fills[zero_step] = 1
                    num_pos_here += 1
                else:
                    zero_fills[zero_step] = -1
                    num_neg_here += 1
            binary_ts[ batch_index,node_index,is_zero[batch_index,node_index,:] ] = zero_fills
    return binary_ts

def get_time_series_mean(time_series:torch.Tensor):
    state_mean = torch.mean(time_series, dim=-1)
    batch_size, num_nodes, num_steps = time_series.size()
    triu_rows, triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=time_series.device)
    num_products = triu_rows.numel()
    state_product_sum = torch.zeros( size=(batch_size, num_products), dtype=time_series.dtype, device=time_series.device )
    for step in range(num_steps):
        state = time_series[:,:,step]
        state_product_sum += state[:,triu_rows] * state[:,triu_cols]
    return state_mean, state_product_sum/num_steps

def get_fc_binary(s_mean:torch.Tensor, s_product_mean:torch.Tensor, epsilon:torch.float=10e-10):
    # For the functional connectivity (FC), we use the Pearson correlations between pairs of nodes.
    # The correlation between nodes i and j is ( mean(s[i]*s[j]) - mean(s[i])*mean(s[j]) )/( std.dev.(s[i])*std.dev(s[j]) )
    # The standard deviation is sqrt( mean(s^2) - mean(s)^2 ).
    # If the state is always either -1 or +1, then s^2 is always +1, so the mean of s^2 is 1.
    triu_rows, triu_cols = get_triu_indices_for_products( num_nodes=s_mean.size(dim=-1), device=s_mean.device )
    s_std = torch.sqrt( 1 - s_mean.square() )
    return (s_product_mean - s_mean[:,triu_rows] * s_mean[:,triu_cols])/(s_std[:,triu_rows] * s_std[:,triu_cols] + epsilon)

def get_pairwise_correlation(mat1:torch.Tensor, mat2:torch.Tensor, epsilon:torch.float=10e-10):
    std_1, mean_1 = torch.std_mean(mat1, dim=-1)
    std_2, mean_2 = torch.std_mean(mat2, dim=-1)
    return ( torch.mean(mat1 * mat2, dim=-1) - mean_1 * mean_2 + epsilon )/(std_1 * std_2 + epsilon)

def get_pairwise_rmse(mat1:torch.Tensor, mat2:torch.Tensor):
    return torch.sqrt(  torch.mean( (mat1 - mat2).square(), dim=-1 )  )

def fold_nodes(node_params:torch.Tensor, num_folds:int):
    num_nodes = node_params.size(dim=-1)
    nodes_per_fold = num_nodes//num_folds
    num_nodes_to_keep = nodes_per_fold * num_folds
    # Folding the per-node values is straight-forward.
    folded_node_params = torch.flatten(  torch.unflatten( input=node_params[:,:num_nodes_to_keep], dim=-1, sizes=(num_folds, nodes_per_fold) ), start_dim=0, end_dim=1  )
    return folded_node_params

def fold_node_pairs(pair_params:torch.Tensor, num_nodes:int, num_folds:int):
    dtype = pair_params.dtype
    device = pair_params.device
    batch_size = pair_params.size(dim=0)
    nodes_per_fold = num_nodes//num_folds
    # Folding the per-pair values is more complex.
    # The actual number of values decreases, since we drop pairs of nodes that are placed into different folds.
    triu_rows, triu_cols = get_triu_indices_for_products(num_nodes=num_nodes, device=device)
    square_mat = torch.zeros( size=(batch_size, num_nodes, num_nodes), dtype=dtype, device=device )
    square_mat[:,triu_rows,triu_cols] = pair_params
    square_mat[:,triu_cols,triu_rows] = pair_params
    folded_square_mat = torch.zeros( size=(batch_size, num_folds, nodes_per_fold, nodes_per_fold), dtype=dtype, device=device )
    for fold in range(num_folds):
        fold_start = fold * nodes_per_fold
        fold_end = fold_start + nodes_per_fold
        folded_square_mat[:,fold,:,:] = square_mat[:,fold_start:fold_end,fold_start:fold_end]
    folded_triu_rows, folded_triu_cols = get_triu_indices_for_products(num_nodes=nodes_per_fold, device=device)
    folded_pair_params = folded_square_mat[:,:,folded_triu_rows,folded_triu_cols].flatten(start_dim=0, end_dim=1)
    return folded_pair_params

def init_with_means_and_num_betas(mean_state:torch.Tensor, mean_state_product:torch.Tensor, num_betas_per_model:int):
    # Initialize the model.
    dtype = mean_state.dtype
    device = mean_state.device
    batch_size, num_nodes = mean_state.size()
    num_models = batch_size//num_betas_per_model
    beta = torch.linspace(start=0.0, end=1.0, steps=num_betas_per_model, dtype=dtype, device=device).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat( (1,num_models,1) )
    # print( '0th beta[0]', beta[:,0].tolist() )
    model = IsingModel( batch_size=batch_size, num_nodes=num_nodes, beta=beta.flatten(start_dim=0,end_dim=1), dtype=dtype, device=device )
    model.set_params(h=mean_state, J_triu=mean_state_product)
    model.set_target_means(target_node_means=mean_state, target_node_product_means=mean_state_product)
    return model

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
class IsingModel(torch.nn.Module):

    def __init__(self, target_state_means:torch.Tensor, target_state_product_means:torch.Tensor, num_betas_per_target:int, random_init:bool=False, optimizer_name:str=None, loss_fn:torch.nn.MSELoss=None, learning_rate:torch.float=0.001):
        super(IsingModel, self).__init__()
        # Keep track of both how large the model is along each dimension and which dimension is which.
        dtype = target_state_means.dtype
        device = target_state_means.device
        num_targets = target_state_means.size(dim=0)
        # For every set of target means, create num_betas_per_target replicate models.
        # Assign each model of the same target a different beta value, initially evenly spaced throughout the range from 0 to 1.
        if num_betas_per_target > 1:
            self.beta = torch.linspace(start=0.0, end=1.0, steps=num_betas_per_target, dtype=dtype, device=device).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat( (1,num_targets,1) ).flatten(start_dim=0, end_dim=1)
        else:
            # If we only have one beta per target, set them all to 1, not 0.
            self.beta = torch.ones( size=(num_targets, 1), dtype=dtype, device=device )
        # Save copies of these target means to use for fitting later.
        self.target_state_means = target_state_means.unsqueeze(dim=0).repeat( (num_betas_per_target, 1, 1) ).flatten(start_dim=0, end_dim=1)
        self.target_state_product_means = triu_to_square_pairs(triu_pairs=target_state_product_means, diag_fill=1.0).unsqueeze(dim=0).repeat( (num_betas_per_target, 1, 1, 1) ).flatten(start_dim=0, end_dim=1)
        # We can initialize the parameters and state either randomly or with the means and a fixed default state.
        if random_init:
            self.h = torch.randn_like(self.target_state_means)
            self.J = torch.rand_like(self.target_state_product_means)
            self.s = 2.0 * torch.randint_like(input=self.target_state_means, low=0.0, high=2.0) - 1.0
        else:
            self.h = self.target_state_means.clone()
            self.J = self.target_state_product_means.clone()
            # Initialize the state of each model to all -1s.
            # In the binarized HCP fMRI data, this is the most common state.
            self.s = torch.full_like( input=self.h, fill_value=-1.0 )
        # There is one key difference between target means and parameters.
        # The product of any state with itself is 1, so the diagonals of both the simulated and data mean states are 1.
        # However, when we compute the entropy, a node should not be coupled to itself, so the diagonal of J must be 0.
        self.daig_mask = self.get_diag_mask()# We also use diag_mask as a constant in forward().
        self.J *= self.daig_mask
        # Keep track of the number of distinct targets and number of betas per target so that we can unflatten the batch dimension later.
        self.num_targets = num_targets
        self.num_betas_per_target = num_betas_per_target
        # Keep track of how many times we have updated beta.
        self.num_beta_updates = 0
        # Keep track of how many times we have updated h and J.
        self.num_param_updates = 0
        # If the caller passed in an optimizer, save it, and make the parameters explicitly Autograd trackable.
        if type(optimizer_name) != type(None):
            self.h = torch.nn.Parameter(self.h)
            self.J = torch.nn.Parameter(self.J)
            optimizer_name_lower = optimizer_name.lower()
            if optimizer_name_lower == 'adam':
                self.optimizer = torch.optim.Adam( params=self.parameters(), lr=learning_rate )
            elif optimizer_name_lower == 'sgd':
                self.optimizer = torch.optim.SGD( params=self.parameters(), lr=learning_rate )
            else:
                self.optimizer = None
                print(f'IsingModel did not expect to use optimizer {optimizer_name}. Add your own code to instantiate it.')
            if type(loss_fn) != type(None):
                self.loss_fn = loss_fn
            else:
                self.loss_fn = torch.nn.MSELoss()
    
    # Have the forward function output the pseudolikelihood of a batch of states in a num_models x num_nodes x num_time_points Tensor.
    def forward(self, target_ts:torch.Tensor):
        mean_field = torch.tanh( self.h.unsqueeze(dim=-1) + torch.matmul(self.J, target_ts) )# S x N x 1 + (S x N x N @ S x N x T) -> S x N x T
        mean_field_mean = torch.mean(input=mean_field, dim=-1, keepdim=False)# mean(S x N x T, dim=-1, keepdim=True) -> S x N x 1
        mean_field_outer_product = torch.matmul( target_ts, mean_field.transpose(dim0=-2, dim1=-1) )/target_ts.size(dim=-1)# S x N x T @ transpose(S x N x T) -> S x N x T @ S x T x N -> S x N x N
        mean_field_outer_product_sym = 0.5 * square_to_triu_pairs( mean_field_outer_product + mean_field_outer_product.transpose(dim0=-2, dim1=-1) )
        return torch.cat( ( mean_field_mean, mean_field_outer_product_sym ), dim=-1 )
    
    def do_balanced_metropolis_step(self):
        # deltaE[i] = -2*beta*s1[i]*( h[i] + torch.sum(J[i,:]*s1) )
        # P_flip[i] = torch.exp( beta*deltaE[i] )
        # If deltaE < 0, then P_flip > 1, so it will definitely flip.
        # If deltaE > 0, the 0 < P_flip < 1, smaller for larger deltaE.
        # We can model this by randomly generating a number c between 0 and 1 and flipping if c < P_flip.
        # Note that we do not need to clamp P_flip to the range (0, 1].
        # It is faster to just pre-generate all our randomly selected floats at once.
        # When using GPU acceleration, we can also save some time using the following trick:
        # c < P_flip <=> c < exp(beta*deltaE)
        #            <=> log(c) < beta*deltaE
        #            <=> log(c)/beta < deltaE
        #            <=> log(c)/beta < -2*s[i]*( h[i] + torch.sum(J[i,:]*s) )
        #            <=> log(c)/(-2*beta) > s[i]*( h[i] + torch.sum(J[i,:]*s) )
        # We assume beta > 0.
        rand_choice = torch.rand_like(input=self.s).log()/(-2*self.beta)# B x N
        for node in torch.randperm( n=self.s.size(dim=-1), dtype=int_type, device=self.s.device ):
            self.s[:,node] *= 1.0 - 2.0*(   rand_choice[:,node] > (  self.s[:,node] * ( self.h[:,node] + torch.sum(self.J[:,node,:]*self.s, dim=-1) )  )   ).float()
    
    def get_triu_indices_for_products(self):
        return get_triu_indices_for_products( num_nodes=self.s.size(dim=-1), device=self.s.device )
    
    def get_target_means(self):
        return self.target_state_means, square_to_triu_pairs(square_pairs=self.target_state_product_means)
    
    # Simulate the Ising model.
    # Record the means of the states and products of states.
    # In this version, we compute the mean product as a square matrix, which is faster.
    # We then convert it to a more memory-efficient form by extracting the upper triangular part.
    def simulate_and_record_means(self, num_steps:int):
        s_mean, s_product_mean = self.simulate_and_record_means_square(num_steps=num_steps)
        return s_mean, square_to_triu_pairs(square_pairs=s_product_mean)

    # Simulate the Ising model.
    # Record the means of the states and products of states.
    # Record the means of state products in a square matrix.
    # This is faster but includes redundant elements that take up memory.
    def simulate_and_record_means_square(self, num_steps:int):
        s_sum = torch.zeros_like(self.s)
        s_product = torch.zeros_like(self.J)
        s_product_sum = torch.zeros_like(self.J)
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            s_sum += self.s# B x N x 1
            torch.mul(input=self.s[:,:,None], other=self.s[:,None,:], out=s_product)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            s_product_sum += s_product
        return s_sum/num_steps, s_product_sum/num_steps

    # Simulate the Ising model.
    # Record the means of the states and products of states.
    # Only calculate the products above the diagonal.
    # This is more memory efficient but a little slower due to indexing.
    def simulate_and_record_means_triu_every_step(self, num_steps:int):
        s_sum = torch.zeros_like(self.s)
        batch_size = s_sum.size(dim=0)
        triu_rows, triu_cols = self.get_triu_indices_for_products()
        num_products = triu_rows.numel()
        s_product_sum = torch.zeros( (batch_size, num_products), dtype=s_sum.dtype, device=s_sum.device )
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            s_sum += self.s# B x N x 1
            s_product_sum += self.s[:,triu_rows] * self.s[:,triu_cols]
        return s_sum/num_steps, s_product_sum/num_steps
    
    # In this version, we assume target_product_mean is the B x N x N version with 1s on the diagonal, not the upper-triangular version.
    def seek_best_beta(self, num_updates:torch.int=1, sim_length:torch.int=1200):
        target_state_means, target_state_product_means = self.get_target_means()
        combined_target_means = torch.cat( (target_state_means, target_state_product_means), dim=-1 )
        combined_sim_means = torch.zeros_like(combined_target_means)
        unflattened_sizes = (self.num_betas_per_target, self.num_targets)
        beta = self.beta.squeeze(dim=-1).unflatten(dim=0, sizes=unflattened_sizes)# betas_per_target*num_targets x 1 -> betas_per_target x num_targets
        num_nodes = target_state_means.size(dim=-1)
        for _ in range(num_updates):
            combined_sim_means[:,:num_nodes], combined_sim_means[:,num_nodes:] = self.simulate_and_record_means(num_steps=sim_length)# betas_per_target*num_targets x num_nodes or num_pairs
            combined_mean_state_rmse = get_pairwise_rmse(combined_sim_means, combined_target_means)# betas_per_target*num_targets
            best_beta_index = combined_mean_state_rmse.unflatten(dim=0, sizes=unflattened_sizes).argmin(dim=0, keepdim=False)# betas_per_target x num_targets -> num_targets
            for target_index in range(self.num_targets):
                best_beta_index_for_target = best_beta_index[target_index]
                betas_for_target = beta[:,target_index]
                best_beta_for_target = betas_for_target[best_beta_index_for_target]
                range_width = betas_for_target[-1] - betas_for_target[0]
                if best_beta_index_for_target == 0:
                    beta_start = 0.0
                    if best_beta_for_target == 0.0:
                        # If the best value we tried is 0, try values in a smaller range around 0.
                        # There might be better values that are close to but not exactly 0.
                        beta_start = 0.0
                        beta_end = betas_for_target[1]
                    else:
                        # If the lowest value we tried is the best and is not 0, try values less than that value.
                        # An even lower value might be even better.
                        # Preserve the current range width so long as it does not take us below 0.
                        beta_start = max(0.0, best_beta_for_target-range_width)
                        beta_end = best_beta_for_target
                elif best_beta_index_for_target == self.num_betas_per_target-1:
                    # If the highest value we tried is the best, try values greater than that value.
                    # There might be even higher values that are better.
                    # Preserve the current range width.
                    beta_start = best_beta_for_target
                    beta_end = best_beta_for_target + range_width
                else:
                    # If the best value was between two other values, search the interval between those values at higher resolution.
                    beta_start = betas_for_target[best_beta_index_for_target-1]
                    beta_end = betas_for_target[best_beta_index_for_target+1]
                beta[:,target_index] = torch.linspace(start=beta_start, end=beta_end, steps=self.num_betas_per_target, dtype=beta.dtype, device=beta.device)
            self.num_beta_updates += 1
            # print( f'{sim+1}th beta[0]', beta[:,0].tolist() )
            self.beta = beta.flatten(start_dim=0, end_dim=1).unsqueeze(dim=-1)
        return combined_sim_means[:,:num_nodes], combined_sim_means[:,num_nodes:]
    
    # data_ts is a time series with dimensions B x N x T where T can be any integer >= 1.
    # In this version, we assume target_product_mean is the B x N x N version with 1s on the diagonal, not the upper-triangular version.
    def fit(self, target_mean:torch.Tensor, target_product_mean:torch.Tensor, num_updates:torch.int=1, steps_per_update:torch.int=50, learning_rate:torch.float=0.001):
        for _ in range(num_updates):
            s_mean, s_product_mean = self.simulate_and_record_means_square(num_steps=steps_per_update)
            self.h += learning_rate * (target_mean - s_mean)
            # Since the product of any state with itself is +1, the diagonals of both the target and sim product means are 1.
            # Consequently, the difference between the diagonals is 0, leading to no change in J, which is initialized to 0.
            # This is intentional, since no node should have a coupling to itsef.
            self.J += learning_rate * (target_product_mean - s_product_mean)
            self.num_param_updates += 1
        return s_mean, square_to_triu_pairs(s_product_mean)
    
    # data_ts is a time series with dimensions B x N x T where T can be any integer >= 1.
    def fit_to_stored_means(self, num_updates:torch.int=1, steps_per_update:torch.int=50, learning_rate:torch.float=0.001):
        return self.fit(target_mean=self.target_state_means, target_product_mean=self.target_state_product_means, num_updates=num_updates, steps_per_update=steps_per_update, learning_rate=learning_rate)
    
    # Use this if target_product_mean is in upper triangular form.
    def fit_triu(self, target_mean:torch.Tensor, target_product_mean:torch.Tensor, num_updates:torch.int=1, steps_per_update:torch.int=50, learning_rate:torch.float=0.001):
        return self.fit( target_mean=target_mean, target_product_mean=triu_to_square_pairs(triu_pairs=target_product_mean, diag_fill=1.0), num_updates=num_updates, steps_per_update=steps_per_update, learning_rate=learning_rate )
    
    def fit_pseudolikelihood(self, target_ts:torch.Tensor, target_state_means:torch.Tensor, target_state_product_means:torch.Tensor, num_updates:int, learning_rate:torch.float=0.001):
        J_diff = torch.zeros_like(self.J)
        num_time_points = target_ts.size(dim=-1)
        triu_row, triu_col = self.get_triu_indices_for_products()
        for _ in range(num_updates):
            mean_field = torch.tanh( self.h.unsqueeze(dim=-1) + torch.matmul(self.J, target_ts) )# S x N x 1 + (S x N x N @ S x N x T) -> S x N x T
            mean_field_mean = torch.mean(input=mean_field, dim=-1, keepdim=False)# mean(S x N x T, dim=-1, keepdim=True) -> S x N x 1
            h_diff = target_state_means - mean_field_mean
            mean_field_outer_product = torch.matmul( target_ts, mean_field.transpose(dim0=-2, dim1=-1) )/num_time_points# S x N x T @ transpose(S x N x T) -> S x N x T @ S x T x N -> S x N x N
            mean_field_outer_product_triu = 0.5 * (mean_field_outer_product[:,triu_row,triu_col] + mean_field_outer_product[:,triu_col,triu_row])
            J_diff_triu = target_state_product_means - mean_field_outer_product_triu
            J_diff[:,triu_row,triu_col] = J_diff_triu
            J_diff[:,triu_col,triu_row] = J_diff_triu
            self.h += learning_rate * h_diff
            self.J += learning_rate * J_diff
            self.num_param_updates += 1
        return mean_field_mean, mean_field_outer_product_triu
    
    def fit_pseudolikelihood_to_stored_means(self, target_ts:torch.Tensor, num_updates:int, learning_rate:torch.float=0.001):
        target_state_means, target_state_product_means = self.get_target_means()
        return self.fit_pseudolikelihood(target_ts=target_ts, target_state_means=target_state_means, target_state_product_means=target_state_product_means, num_updates=num_updates, learning_rate=learning_rate)

    def fit_pseudolikelihood_optimizer(self, target_ts:torch.Tensor, target_state_means:torch.Tensor, target_state_product_means:torch.Tensor, num_updates:int):
        combined_target = torch.cat( ( target_state_means, target_state_product_means ), dim=-1 )
        for _ in range(num_updates):
            self.optimizer.zero_grad()
            combined_prediction = self(target_ts)
            loss = self.loss_fn(combined_prediction, combined_target)
            loss.backward()
            self.optimizer.step()
            self.num_param_updates += 1
        num_nodes = self.s.size(dim=-1)
        return combined_prediction[:,:num_nodes], combined_prediction[:,num_nodes:]
    
    def fit_pseudolikelihood_to_stored_means_optimizer(self, target_ts:torch.Tensor, num_updates:int):
        target_state_means, target_state_product_means = self.get_target_means()
        return self.fit_pseudolikelihood_optimizer(target_ts=target_ts, target_state_means=target_state_means, target_state_product_means=target_state_product_means, num_updates=num_updates)
    
    def get_diag_mask(self):
        return torch.ones_like(self.J) - torch.eye( n=self.J.size(dim=-1), dtype=self.J.dtype, device=self.J.device ).unsqueeze(dim=0)

    # Run a simulation, and record and return the functional connectivity matrix (FC).
    # The FC has dimensions B x N x N where fc[b,i,j] is the Pearson correlation between nodes i and j in model b.
    # We replace any 0s in the denominator of the division at the end with the small number epsilon to avoid division by 0.
    # Since the FC is symmetric with 1 along the diagonal, we only return the upper triangular part above the diagonal.
    def simulate_and_record_fc(self, num_steps:int, epsilon:torch.float=10e-10):
        s_mean, s_product_mean = self.simulate_and_record_means(num_steps=num_steps)
        return get_fc_binary(s_mean=s_mean, s_product_mean=s_product_mean, epsilon=epsilon)
    
    # Run a simulation, and record and return the Fisher Information Matrix (FIM) estimated from the observed states.
    # Since the FIM is symmetric, we only retain the upper triangular part, including the diagonal.
    def simulate_and_record_fim(self, num_steps:int):
        batch_size, num_nodes = self.s.size()
        state_triu_rows, state_triu_cols = self.get_triu_indices_for_products()
        # Unlike with the products matrix, the diagonal of the FIM is meaningful.
        num_observables = num_nodes + state_triu_rows.numel()
        fim_triu_indices = torch.triu_indices(row=num_observables, col=num_observables, offset=0, dtype=int_type, device=self.s.device)
        fim_triu_rows = fim_triu_indices[0]
        fim_triu_cols = fim_triu_indices[1]
        num_fim_triu_elements = fim_triu_indices.size(dim=-1)
        observables = torch.zeros( (batch_size, num_observables), dtype=self.s.dtype, device=self.s.device )
        observables_mean = torch.zeros_like(observables)
        observable_product_mean = torch.zeros( (batch_size, num_fim_triu_elements), dtype=self.s.dtype, device=self.s.device )
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            observables[:,:num_nodes] = self.s
            observables[:,num_nodes:] = self.s[:,state_triu_rows] * self.s[:,state_triu_cols]
            observables_mean += observables
            observable_product_mean += observables[:,fim_triu_rows] * observables[:,fim_triu_cols]
        observables_mean /= num_steps
        observable_product_mean /= num_steps
        fim = observable_product_mean - observables_mean[:,fim_triu_rows] * observables_mean[:,fim_triu_cols]
        return fim
    
    # Run a simulation, and record and return the Fisher Information Matrix (FIM) estimated from the observed states.
    # Record the FIM as a square matrix.
    def simulate_and_record_fim_square(self, num_steps:int):
        batch_size, num_nodes = self.s.size()
        state_triu_rows, state_triu_cols = self.get_triu_indices_for_products()
        # Unlike with the products matrix, the diagonal of the FIM is meaningful.
        num_observables = num_nodes + state_triu_rows.numel()
        observables = torch.zeros( (batch_size, num_observables), dtype=self.s.dtype, device=self.s.device )
        observables_mean = torch.zeros_like(observables)
        fim = torch.zeros( (batch_size, num_observables, num_observables), dtype=self.s.dtype, device=self.s.device )
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            observables[:,:num_nodes] = self.s
            observables[:,num_nodes:] = self.s[:,state_triu_rows] * self.s[:,state_triu_cols]
            observables_mean += observables
            fim += observables[:,:,None] * observables[:,None,:]
        fim /= num_steps
        observables_mean /= num_steps
        fim -= observables_mean[:,:,None] * observables_mean[:,None,:]
        return fim
    
    # Run a simulation, and record and return the Fisher Information Matrix (FIM) estimated from the observed states.
    # Since the FIM is symmetric, we only retain the upper triangular part, including the diagonal.
    def simulate_and_record_one_fim(self, num_steps:int):
        num_nodes = self.s.size(dim=-1)
        state_triu_rows, state_triu_cols = self.get_triu_indices_for_products()
        # Unlike with the products matrix, the diagonal of the FIM is meaningful.
        num_observables = num_nodes + state_triu_rows.numel()
        fim_triu_indices = torch.triu_indices(row=num_observables, col=num_observables, offset=0, dtype=int_type, device=self.s.device)
        fim_triu_rows = fim_triu_indices[0]
        fim_triu_cols = fim_triu_indices[1]
        num_fim_triu_elements = fim_triu_indices.size(dim=-1)
        observables = torch.zeros( (num_observables), dtype=self.s.dtype, device=self.s.device )
        observables_mean = torch.zeros_like(observables)
        observable_product_mean = torch.zeros( (num_fim_triu_elements), dtype=self.s.dtype, device=self.s.device )
        for _ in range(num_steps):
            self.do_balanced_metropolis_step()
            observables[:,:num_nodes] = self.s
            observables[:,num_nodes:] = self.s[:,state_triu_rows] * self.s[:,state_triu_cols]
            observables_mean += observables
            observable_product_mean += observables[:,fim_triu_rows] * observables[:,fim_triu_cols]
        observables_mean /= num_steps
        observable_product_mean /= num_steps
        fim = observable_product_mean - observables_mean[:,fim_triu_rows] * observables_mean[:,fim_triu_cols]
        return fim

    # Simulate the Ising model.
    # Record the full time series of states.
    def simulate_and_record_time_series(self, num_steps:int):
        batch_size, num_nodes = self.s.size()
        time_series = torch.zeros( size=(batch_size, num_nodes, num_steps), dtype=self.s.dtype, device=self.s.device )
        for step in range(num_steps):
            self.do_balanced_metropolis_step()
            time_series[:,:,step] = self.s
        return time_series