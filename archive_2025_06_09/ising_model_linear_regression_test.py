# Test whether we can use linear regression to recover an Ising model from time series data.
# 1. Randomly generate an Ising model A_orig.
# 2. Generate a matrix R of all 2^N possible states.
# 3. Run the model, and count the number of times each state occurs.
# 4. Continue running until each state occurs at least once.
# 5. Compare these counts to the analytical probabilities calculated from the Ising model parameters.
# 6. Use least-squares regression on the empirical probabilities to compute a second Ising model, A_lstsq.
# 7. Compute its analytical and observed state probabilities, and compare them to those of A_orig.
import torch
import time

with torch.no_grad():
    code_start_time = time.time()
    device = torch.device('cuda')
    float_type = torch.float
    int_type = torch.int

    def get_probs(augmented_states:torch.Tensor, params:torch.Tensor):
        probs_unnormed = torch.matmul( params.transpose(dim0=0, dim1=1), augmented_states.float() ).exp().squeeze()
        return probs_unnormed/torch.sum(probs_unnormed)

    def augment_state(state:torch.Tensor):
        num_nodes = state.numel()
        triu_indices = torch.triu_indices(row=num_nodes, col=num_nodes, dtype=int_type, device=device)
        products = state * state.transpose(dim0=0, dim1=1)
        triu_products = products[ triu_indices[0,:], triu_indices[1,:] ].unsqueeze(dim=1)
        return torch.cat( (state, triu_products), dim=0 )

    def simulate_for_state_probabilities(states:torch.Tensor, probabilities:torch.Tensor, max_steps:int=1000000, stop_when_all_sampled:bool=True):
        counts = torch.zeros( (num_states,), dtype=int_type, device=device )
        node_order = torch.randint( low=0, high=num_nodes, size=(max_steps,), dtype=int_type, device=device )
        choice_thresholds = torch.rand( size=(max_steps,), dtype=float_type, device=device )
        old_state_index = torch.argmax(probabilities).item()# Start from the most probable state.
        old_state = states[:,old_state_index].clone().unsqueeze(dim=1)
        old_probability = probabilities[old_state_index].item()
        for step in range(max_steps):
            new_state = old_state.clone()
            node = node_order[step].item()
            new_state[node] *= -1
            new_state_index = torch.nonzero( torch.sum(new_state * states, dim=0) == num_nodes ).item()
            new_probability = probabilities[new_state_index].item()
            do_flip = choice_thresholds[step].item() < new_probability/old_probability
            # print( 'step', step, ': s[', old_state_index, ']=', old_state.flatten(), 'P=', old_probability, '-', node, '-> s[', new_state_index, ']=', new_state.flatten(), 'P=', new_probability, '?', do_flip )
            if do_flip:
                old_state_index = new_state_index
                old_state = new_state
                old_probability = new_probability
                counts[new_state_index] += 1
            else:
                counts[old_state_index] += 1
            if stop_when_all_sampled and torch.all(counts > 0):
                break
        return counts/counts.sum()

    num_nodes = 4
    print(f'num nodes {num_nodes}')
    num_params = num_nodes + ( num_nodes*(num_nodes-1) )//2
    print(f'num params {num_params}')
    num_states = 2**num_nodes
    print(f'num states {num_states}')

    node_indices = torch.arange(start=0, end=num_nodes, dtype=int_type, device=device)
    two_powers = torch.special.exp2(node_indices)
    state_indices = torch.arange(start=0, end=num_states, dtype=int_type, device=device)
    flip_here = 1 - 2*(   torch.remainder(  state_indices.unsqueeze(dim=0).repeat( (num_nodes,1) ), two_powers.unsqueeze(dim=1).repeat( (1,num_states) )  ) == 0   ).int()
    states = flip_here.cumprod(dim=1)
    print( 'states', states )

    augmented_states = torch.zeros( (num_params, num_states), dtype=int_type, device=device )
    augmented_states[:num_nodes,:] = states
    m = num_nodes
    for i in range(num_nodes-1):
        for j in range(i+1,num_nodes):
            augmented_states[m,:] = states[i,:] * states[j,:]
            m += 1
    print( 'states augmented with products', augmented_states )

    params_orig = torch.randn( (num_params,1), dtype=float_type, device=device )
    print( 'original parameters', params_orig.flatten().tolist() )

    probs_orig = get_probs(augmented_states=augmented_states, params=params_orig)
    print( 'original probabilities', probs_orig.tolist() )

    num_steps = 1000000# It takes about a million steps for the sampled probabilities to converge to something close to the true probabilities for this 4-node toy model. 
    probs_sim = simulate_for_state_probabilities(states=states, probabilities=probs_orig, max_steps=num_steps, stop_when_all_sampled=False)
    print( f'probabilities observed in {num_steps}-step simulation', probs_sim.tolist() )
    kl_div = torch.sum(  probs_orig * torch.log2( probs_orig/(probs_sim + 10e-10) )  )
    print(f'KL divergence {kl_div:.3g}')
    num_observed = probs_sim.count_nonzero()
    print(f'{num_observed} out of {num_states} states observed')

    augmented_states_transpose_float = augmented_states.float().transpose(dim0=0, dim1=1)
    probs_orig_log_column = probs_orig.log().unsqueeze(dim=1)
    params_lstsq = torch.linalg.lstsq(augmented_states_transpose_float, probs_orig_log_column)
    print( 'parameters from least squares over augmented states and original probabilities', params_lstsq )
    # probs_lstsq = torch.matmul( augmented_states_transpose_float, params_lstsq.solution )
    # print( 'probabilities form multiplying the least squares solution back onto the states' )

    probs_sim_log_column = (probs_sim+10e-10).log().unsqueeze(dim=1)

    params_lstsq_sim = torch.linalg.lstsq(augmented_states_transpose_float, probs_sim_log_column)
    print( 'parameters from least squares over augmented states and probabilities estimated from simulation', params_lstsq_sim )

    # In the limit, we get a formula where params[m] = sum( state[m,k]*log(probability[k]), over k in 1...2^k ).
    # Try this out.
    params_lstsq_limit = torch.mean(augmented_states_transpose_float * probs_orig_log_column, dim=0)
    print( 'parameters from limit of least squares', params_lstsq_limit )

    # Also try it with the sampled probabilities.
    params_lstsq_limit_sim = torch.mean(augmented_states_transpose_float * probs_sim_log_column, dim=0)
    print( 'parameters from limit of least squares with sampled probabilities', params_lstsq_limit_sim )

    # Try a more stable way of estimating from sampled probabilities.
    log_counts_sim = torch.log(num_steps * probs_sim + 10e-10).unsqueeze(dim=1)
    params_lstsq_limit_sim_counts = torch.mean(augmented_states_transpose_float * log_counts_sim, dim=0)
    print( 'parameters from limit of least squares with sampled counts', params_lstsq_limit_sim_counts )
    print(f'done, time {time.time() - code_start_time:.3f}')
