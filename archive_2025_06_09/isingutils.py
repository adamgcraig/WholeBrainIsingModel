# functions we use in multiple Ising-model-related scripts
# based on Sida Chen's code, in turn based on
# Panas, D., Amin, H., Maccione, A., Muthmann, O., van Rossum, M., Berdondini, L., & Hennig, M. H. (2015).
# Sloppiness in spontaneously active neuronal networks. Journal of Neuroscience, 35(22), 8480-8492.
# https://www.jneurosci.org/content/35/22/8480
 
import torch

def get_random_state(batch_size:int, num_nodes:int, dtype=torch.float, device='cpu'):
    s = 2.0*torch.randint( 2, (batch_size, num_nodes), dtype=dtype, device=device ) - 1.0
    return s

def get_random_state_like(h:torch.Tensor):
    return 2.0 * torch.randint_like(input=h, high=2) - 1.0

def get_batched_ising_models(batch_size:int, num_nodes:int, dtype=torch.float, device='cpu'):
    J = torch.randn( (batch_size, num_nodes, num_nodes), dtype=dtype, device=device )
    J = 0.5*( J + J.transpose(dim0=-2, dim1=-1) )
    J = J - torch.diag_embed( torch.diagonal(J,dim1=-2,dim2=-1) )
    h = torch.zeros( (batch_size, num_nodes), dtype=dtype, device=device )
    s = 2.0*torch.randint( 2, (batch_size, num_nodes), dtype=dtype, device=device ) - 1.0
    return J, h, s

def standardize_and_binarize_ts_data(ts:torch.Tensor, threshold:float=0.1, time_dim:int=-2):
    ts_std, ts_mean = torch.std_mean(ts, dim=time_dim, keepdim=True)
    return 2.0 * (  ( (ts - ts_mean)/ts_std ) > threshold  ) - 1.0

def median_binarize_ts_data(ts:torch.Tensor, time_dim:int=-2):
    return 2.0 * ( ts > ts.median(dim=time_dim, keepdim=True).values ) - 1.0

def get_batched_delta_h(J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, node_index:int):
    return 2.0 * (  torch.sum( J[:,:,node_index] * s, dim=-1 ) + h[:,node_index]  ) * s[:,node_index]

def run_batched_balanced_metropolis_sim_step(J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, beta:float=0.5):
    num_nodes = s.size(dim=-1)
    # batch_size, num_nodes = s.size()
    # if not torch.is_tensor(sim_ts):
    #     sim_ts = torch.zeros( (batch_size, num_steps, num_nodes), dtype=s.dtype, device=s.device )
    node_order = torch.randperm(n=num_nodes, device=s.device)
    for i in range(num_nodes):
        node_index = node_order[i].item()
        deltaH = get_batched_delta_h(J=J, h=h, s=s, node_index=node_index)
        prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
        flip = torch.bernoulli(prob_accept)
        s[:,node_index] *= ( 1.0 - 2.0*flip )
    return s

def run_batched_balanced_metropolis_sim(sim_ts:torch.Tensor, J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, num_steps:int, beta:float=0.5):
    # batch_size, num_nodes = s.size()
    # if not torch.is_tensor(sim_ts):
    #     sim_ts = torch.zeros( (batch_size, num_steps, num_nodes), dtype=s.dtype, device=s.device )
    for t in range(num_steps):
        s = run_batched_balanced_metropolis_sim_step(J=J, h=h, s=s, beta=beta)
        sim_ts[:,t,:] = s
    return sim_ts, s

# We frequently just want to work with the functional connectivity, not the full time series.
# For long simulation times, just keeping running totals we need in order to calculate the FC has a lower memory footprint than storing the full time series.
def run_batched_balanced_metropolis_sim_for_fc(J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, num_steps:int, beta:float=0.5, epsilon=0.0000000001):
    # batch_size, num_nodes = s.size()
    # if not torch.is_tensor(sim_ts):
    #     sim_ts = torch.zeros( (batch_size, num_steps, num_nodes), dtype=s.dtype, device=s.device )
    original_s_sizes = s.size()
    has_multibatch = original_s_sizes.numel() > 2
    if has_multibatch:
        s = s.flatten(start_dim=0, end_dim=-2)
        h = h.flatten(start_dim=0, end_dim=-2)
        J = J.flatten(start_dim=0, end_dim=-3)
    batch_size, num_nodes = s.size()
    dtype = s.dtype
    device = s.device
    s_sum = torch.zeros_like(s)
    s_product_sum = s_sum[:,:,None] * s_sum[:,None,:]
    sim_fc = torch.zeros( (batch_size, num_nodes, num_nodes), dtype=dtype, device=device )
    for _ in range(num_steps):
        s = run_batched_balanced_metropolis_sim_step(J=J, h=h, s=s, beta=beta)
        s_sum += s
        s_product_sum += (s[:,:,None] * s[:,None,:])
    s_mean = s_sum/num_steps
    s_product_mean = s_product_sum/num_steps
    s_squared_mean = torch.diagonal(s_product_mean, dim1=-2, dim2=-1)
    s_std = torch.sqrt( s_squared_mean - s_mean * s_mean )
    s_cov = s_product_mean - s_mean[:,:,None] * s_mean[:,None,:]
    s_std_prod = s_std[:,:,None] * s_std[:,None,:]
    if torch.any( s_std_prod == 0.0 ):
        s_cov += epsilon
        s_std_prod += epsilon
    sim_fc = s_cov/s_std_prod
    if has_multibatch:
        multibatch_sizes = original_s_sizes[:-1]
        s = s.unflatten( dim=0, sizes=multibatch_sizes )
        # We do not return new h and J, so we do not need to unflatten the views we have.
        sim_fc = sim_fc.unflatten( dim=0, sizes=multibatch_sizes ) 
    return sim_fc, s

def run_batched_parallel_metropolis_sim(sim_ts:torch.Tensor, J:torch.Tensor, h:torch.Tensor, s:torch.Tensor, num_steps:int, beta:float=0.5):
    # batch_size, num_nodes = s.size()
    # if not torch.is_tensor(sim_ts):
    #     sim_ts = torch.zeros( (batch_size, num_steps, num_nodes), dtype=s.dtype, device=s.device )
    for t in range(num_steps):
        deltaH = 2.0 * (   torch.matmul( s[:,None,:], J ).squeeze() + h   ) * s
        prob_accept = torch.clamp( torch.exp(-beta*deltaH), min=0.0, max=1.0 )
        flip = torch.bernoulli(prob_accept)
        s *= ( 1.0 - 2.0*flip )
        sim_ts[:,t,:] = s
    return sim_ts, s