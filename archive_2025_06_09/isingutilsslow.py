# functions we use in multiple Ising-model-related scripts
# based on Sida Chen's code, in turn based on
# Panas, D., Amin, H., Maccione, A., Muthmann, O., van Rossum, M., Berdondini, L., & Hennig, M. H. (2015).
# Sloppiness in spontaneously active neuronal networks. Journal of Neuroscience, 35(22), 8480-8492.
# https://www.jneurosci.org/content/35/22/8480
 
import torch

float_type = torch.float
int_type = torch.int

def get_random_state(batch_size:int, num_nodes:int, dtype=float_type, device='cpu'):
    s = 2.0*torch.randint( 2, (batch_size, num_nodes), dtype=dtype, device=device ) - 1.0
    return s

def get_random_state_like(h:torch.Tensor):
    return 2.0 * torch.randint_like(input=h, high=2) - 1.0

# Let N = h.numel() = s1.numel() = s2.numel().
# This is not quite correct syntax, since we cannot call sum() on lists like this, but it makes the order of operations clearer.
# For an unbatched Ising model, energy = sum([h[i]*s[i] for i in range(N)]) + beta*sum( [ [J[i,j] for i in range(N)] for j in range(N)] )
def get_energy(s:torch.Tensor, h:torch.Tensor, J:torch.Tensor, beta:float=0.5):
    return torch.sum(h*s, dim=-1) + beta * torch.sum( J * s.unsqueeze(dim=-2) * s.unsqueeze(dim=-1), dim=(-1,-2) )

# E = torch.sum( h*s + beta * J * s.unsqueeze(dim=-2) * s.unsqueeze(dim=-1) )
# For simplicity, assume we are just doing one model so that there are no batch dimensions.
# Batched runs are independent, so this does not lead to a loss of generality.
# If we flip the sign of s[i], multiplying it by -1, changing the state from s1 to s2,
# deltaE = torch.sum(h*s2) + beta * torch.sum( J * s2.unsqueeze(dim=-2) * s2.unsqueeze(dim=-1) )
#        - torch.sum(h*s1) - beta * torch.sum( J * s1.unsqueeze(dim=-2) * s1.unsqueeze(dim=-1) )
# deltaE[i] = sum(  [ h[j]*s2[j] for j in range(N) ]  ) + sum( [  [ beta * J[j,k] * s2[j] * s2[k] for j in range(N) ] for k in range(N)  ] )
#           - sum(  [ h[j]*s1[j] for j in range(N) ]  ) - sum( [  [ beta * J[j,k] * s1[j] * s1[k] for j in range(N) ] for k in range(N)  ] )
# We can now change the order of the additions and subtractions, combining the sum() calls and list comprehensions.
# deltaE[i] = sum(  [ h[j]*s2[j] - h[j]*s1[j] for j in range(N) ]  )
#           + sum( [  [ beta * J[j,k] * s2[j] * s2[k] - beta * J[j,k] * s1[j] * s1[k] for j in range(N) ] for k in range(N)  ] )
# Factor out common factors.
# deltaE[i] = sum(  [ h[j]*(s2[j] - s1[j]) for j in range(N) ]  )
#            + sum( [  [ beta * J[j,k] * (s2[j] * s2[k] - s1[j] * s1[k]) for j in range(N) ] for k in range(N)  ] )
# Split up the sums into parts that behave differently, again abusing syntax a little.
# deltaE[i] = sum(  [ h[j]*(s2[j] - s1[j]) for j in range(N) - [i] ]  )
#        + sum(  [ h[j]*(s2[j] - s1[j]) for j in [i] ]  )
#        + sum( [  [ beta * J[j,k] * (s2[j] * s2[k] - s1[j] * s1[k]) for j in range(N) - [i] ] for k in range(N) - [i]  ] )
#        + sum( [  [ beta * J[j,k] * (s2[j] * s2[k] - s1[j] * s1[k]) for j in range(N) - [i] ] for k in [i]  ] )
#        + sum( [  [ beta * J[j,k] * (s2[j] * s2[k] - s1[j] * s1[k]) for j in [i] ] for k in range(N) - [i]  ] )
#        + sum( [  [ beta * J[j,k] * (s2[j] * s2[k] - s1[j] * s1[k]) for j in [i] ] for k in [i]  ] )
# If j != i, then h[j]*(s2[j] - s1[j]) = h[j]*(s1[j] - s1[j]) = 0.
# If j == i, then h[j]*(s2[j] - s1[j]) = h[j]*(-1*s1[j] - s1[j]) = -2*h[j]*s1[j].
# If j != i and k != i, then J[j,k] * (s2[j] * s2[k] - s1[j] * s1[k]) = J[j,k] * (s1[j] * s1[k] - s1[j] * s1[k]) = 0.
# If j != i and k == i, then J[j,k] * (s2[j] * s2[k] - s1[j] * s1[k]) = J[j,k] * ( s1[j] * (-1) * s1[k] - s1[j] * s1[k] ) = -2*J[j,k]*s1[j]*s1[k].
# If j == i and k != i, then J[j,k] * (s2[j] * s2[k] - s1[j] * s1[k]) = J[j,k] * ( (-1) * s1[j] * s1[k] - s1[j] * s1[k] ) = -2*J[j,k]*s1[j]*s1[k].
# If j == i and k == i, then J[j,k] * (s2[j] * s2[k] - s1[j] * s1[k]) = J[j,k] * ( (-1) * s1[j] * (-1) * s1[k] - s1[j] * s1[k] )
#                                                                     = J[j,k] * ( s1[j] * s1[k] - s1[j] * s1[k] ) = 0.
# Put these all together.
# deltaE[i] = sum(  [ 0 for j in range(N) - [i] ]  )
#        + sum(  [ -2*h[j]*s1[j] for j in [i] ]  )
#        + sum( [  [ 0 for j in range(N) - [i] ] for k in range(N) - [i]  ] )
#        + sum( [  [ -2*beta*J[j,k]*s1[j]*s1[k] for j in range(N) - [i] ] for k in [i]  ] )
#        + sum( [  [ -2*beta*J[j,k]*s1[j]*s1[k] for j in [i] ] for k in range(N) - [i]  ] )
#        + sum( [  [ 0 for j in [i] ] for k in [i]  ] )
# Remove all-0 sums.
# deltaE[i] = sum(  [ -2*h[j]*s1[j] for j in [i] ]  )
#        + sum( [  [ -2*beta*J[j,k]*s1[j]*s1[k] for j in range(N) - [i] ] for k in [i]  ] )
#        + sum( [  [ -2*beta*J[j,k]*s1[j]*s1[k] for j in [i] ] for k in range(N) - [i]  ] )
# Simplify one-term sums.
# deltaE[i] = -2*h[i]*s1[i]
#        + sum( [ -2*beta*J[j,i]*s1[j]*s1[i] for j in range(N) - [i] ] )
#        + sum( [ -2*beta*J[i,k]*s1[i]*s1[k] for k in range(N) - [i] ] )
# We can swap the variable name k for j in the second sum without loss of generality.
# deltaE[i] = -2*h[i]*s1[i]
#        + sum( [ -2*beta*J[j,i]*s1[j]*s1[i] for j in range(N) - [i] ] )
#        + sum( [ -2*beta*J[i,j]*s1[i]*s1[j] for j in range(N) - [i] ] )
# Factor out common factors.
# deltaE[i] = -2*s1[i]*(   h[i] + beta*(  sum( [ J[j,i]*s1[j] for j in range(N) - [i] ] ) + sum( [ J[i,j]*s1[j] for j in range(N) - [i] ] )  )   )
# Combine the sums.
# deltaE[i] = -2*s1[i]*(   h[i] + beta*sum( [ (J[j,i]+J[i,j])*s1[j] for j in range(N) - [i] ] )   )
# Rewrite without the list comprehensions using just vector operations.
# deltaE[i] = -2*s1[i]*(  h[i] + beta*torch.sum( (J[:,i] + J[i,:])*s1 ) - 2*beta*J[i,i]*s1[i]  )
# deltaE[i] = -2*s1[i]*(  h[i] + beta*torch.sum( (J[:,i] + J[i,:])*s1 )  ) - -2*beta*s1[i]*2*J[i,i]*s1[i]
# deltaE[i] = -2*s1[i]*(  h[i] + beta*torch.sum( (J[:,i] + J[i,:])*s1 )  ) + 4*beta*J[i,i]
# We can eliminate the last term if we assume J[i,i] = 0.
# If we also assume that J is symmetric, then we can further simplify the expression.
# deltaE[i] = -2*s1[i]*( h[i] + beta*torch.sum(2*J[i,:]*s1) )
# deltaE[i] = -2*s1[i]*( h[i] + 2*beta*toch.sum(J[i,:]*s1) )
def get_energy_change_if_flipped(i:torch.int, s:torch.Tensor, h:torch.Tensor, J:torch.Tensor, beta:float=0.5):
    i_tensor = torch.tensor([i], dtype=torch.int, device=J.device)
    J_row = torch.index_select(input=J, dim=-1, index=i_tensor).squeeze(dim=-1)
    J_col = torch.index_select(input=J, dim=-2, index=i_tensor).squeeze(dim=-2)
    J_ii = torch.index_select( input=J_row, dim=-1, index=i_tensor ).squeeze(dim=-1)
    h_i = torch.index_select(input=h, dim=-1, index=i_tensor).squeeze(dim=-1)
    s_i = torch.index_select(input=s, dim=-1, index=i_tensor).squeeze(dim=-1)
    return -2*s_i*(  h_i + beta*torch.sum( (J_row + J_col)*s, dim=-1 ).squeeze(dim=-1) - 2*J_ii*s_i  )

# We can also compute what the energy change would be given each possible choice of i.
# deltaE[i] = -2*s1[i]*(  h[i] + beta*torch.sum( (J[:,i] + J[i,:])*s1 )  ) + 4*J[i,i]
# deltaE = -2*s1*(  h + beta*torch.matmul( J + J.transpose(dim0=-2, dim1=-1), s1 )  ) + 4*J[i,i]
def get_energy_change_if_flipped_for_each(s:torch.Tensor, h:torch.Tensor, J:torch.Tensor, beta:float=0.5):
    return -2*s*(  h + beta*torch.matmul( J + J.transpose(dim0=-2, dim1=-1), s.unsqueeze(dim=-1) ).squeeze(dim=-1)  ) + 4*torch.diagonal(input=J, offset=0, dim1=-2, dim2=-1)

# How does the change in energy that would occur if we flipped node i change when we flip node j?
# deltaE[i] = -2*s1[i]*(   h[i] + beta*torch.sum( [ (J[j,i]+J[i,j])*s1[j] for j in range(N) - [i] ] )   )
# Use k in sums so that we do not get the loop variable confused with j.
# Let deltaE1[i] = -2*s1[i]*(   h[i] + beta*sum( [ (J[k,i]+J[i,k])*s1[j] for k in range(N) - [i] ] )   ),
#     deltaE2[i] = -2*s2[i]*(   h[i] + beta*sum( [ (J[k,i]+J[i,k])*s2[j] for k in range(N) - [i] ] )   ),
#     s2[j] = s1[j] where i!= j,
#     s2[i] = -1*s1[i].
# deltaDeltaE[i,j] = deltaE2[i] - deltaE1[i]
# deltaDeltaE[i,j] = -2*s2[i]*(   h[i] + beta*sum( [ (J[k,i]+J[i,k])*s2[j] for k in range(N) - [i] ] )   )
#                  - -2*s1[i]*(   h[i] + beta*sum( [ (J[k,i]+J[i,k])*s1[j] for k in range(N) - [i] ] )   )
# Distribute some factors so that we can regroup.
# deltaDeltaE[i,j] = -2*s2[i]*h[i] + -2*s2[i]*beta*sum( [ (J[k,i]+J[i,k])*s2[j] for k in range(N) - [i] ] )
#                   + 2*s1[i]*h[i] + 2*s1[i]*beta*sum( [ (J[k,i]+J[i,k])*s1[j] for k in range(N) - [i] ] )
# deltaDeltaE[i,j] = -2*s2[i]*h[i] + sum( [ -2*beta*s2[i]*s2[j]*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
#                   + 2*s1[i]*h[i] + sum( [ 2*beta*s1[i]*s1[j]*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# Combine the things we can combine.
# deltaDeltaE[i,j] = -2*s2[i]*h[i] + 2*s1[i]*h[i] + sum( [ -2*beta*s2[i]*s2[k]*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
#                                                 + sum( [ 2*beta*s1[i]*s1[k]*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# deltaDeltaE[i,j] = -2*s2[i]*h[i] + 2*s1[i]*h[i] + sum( [ -2*beta*s2[i]*s2[j]*(J[k,i]+J[i,k]) + 2*beta*s1[i]*s1[j]*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# deltaDeltaE[i,j] = 2*s1[i]*h[i] - 2*s2[i]*h[i] + sum( [ 2*beta*s1[i]*s1[k]*(J[k,i]+J[i,k]) - 2*beta*s2[i]*s2[k]*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# deltaDeltaE[i,j] = 2*(s1[i] - s2[i])*h[i] + sum( [ 2*beta*(s1[i]*s1[k] - s2[i]*s2[k])*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# Case where i != j:
# deltaDeltaE[i,j] = 2*(s1[i] - s1[i])*h[i] + sum( [ 2*beta*(s1[i]*s1[k] - s1[i]*s2[k])*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# deltaDeltaE[i,j] = sum( [ 2*beta*(s1[i]*s1[k] - s1[i]*s2[k])*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# deltaDeltaE[i,j] = 2*beta*s1[i]*sum( [ (s1[k] - s2[k])*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# Separate out term where k == j.
# deltaDeltaE[i,j] = 2*beta*s1[i]*sum( [ (s1[k] - s2[k])*(J[k,i]+J[i,k]) for k in range(N) - [i] - [j] ] ) + 2*beta*s1[i]*(s1[j] - s2[j])*(J[j,i]+J[i,j])
# deltaDeltaE[i,j] = 2*beta*s1[i]*sum( [ (s1[k] - s1[k])*(J[k,i]+J[i,k]) for k in range(N) - [i] - [j] ] ) + 2*beta*s1[i]*(s1[j] - -1*s1[j])*(J[j,i]+J[i,j])
# deltaDeltaE[i,j] = 2*beta*s1[i]*sum( [ 0*(J[k,i]+J[i,k]) for k in range(N) - [i] - [j] ] ) + 2*2*beta*s1[i]*s1[j]*(J[j,i]+J[i,j])
# deltaDeltaE[i,j] = 4*beta*s1[i]*s1[j]*(J[j,i]+J[i,j])
# Case where i == j:
# deltaDeltaE[i,j] = 2*(s1[i] - -1*s1[i])*h[i] + sum( [ 2*beta*(s1[i]*s1[k] - -1*s1[i]*s2[k])*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# deltaDeltaE[i,j] = 2*2*s1[i]*h[i] + sum( [ 2*beta*s1[i]*(s1[k] + s2[k])*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# We do not have a term in the sum where k == i, so s2[k] == s1[k] for all terms.
# deltaDeltaE[i,j] = 2*2*s1[i]*h[i] + sum( [ 2*2*beta*s1[i]*s1[k]*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )
# deltaDeltaE[i,j] = 4*s1[i]*(  h[i] + beta*sum( [ s1[k]*(J[k,i]+J[i,k]) for k in range(N) - [i] ] )  )
# Rewrite in terms of vector operations.
# deltaDeltaE[i,j] = 4*s1[i]*(  h[i] + beta*torch.sum( s1*(J[:,i]+J[i,:]) ) - 2*beta*s1[i]*J[i,i]  )
# deltaDeltaE[i,j] = 4*s1[i]*(  h[i] + beta*torch.sum( s1*(J[:,i]+J[i,:]) )  ) - 4*beta*s1[i]*2*s1[i]*J[i,i]
# deltaDeltaE[i,j] = 4*s1[i]*(  h[i] + beta*torch.sum( s1*(J[:,i]+J[i,:]) )  ) - 8*beta*J[i,i] = -2*deltaE[i]
def get_change_in_change_in_energy_if_flipped(i:torch.int, j:torch.int, s:torch.Tensor, h:torch.Tensor, J:torch.Tensor, beta:float=0.5):
    if i == j:
        return -2*get_energy_change_if_flipped(i=i, s=s, h=h, J=J, beta=beta)
    else:
        i_tensor = torch.tensor([i], dtype=torch.int, device=J.device)
        j_tensor = torch.tensor([j], dtype=torch.int, device=J.device)
        s_i = torch.index_select(input=s, dim=-1, index=i_tensor).flatten(start_dim=-2, end_dim=-1)
        s_j = torch.index_select(input=s, dim=-1, index=j_tensor).flatten(start_dim=-2, end_dim=-1)
        J_ij = torch.index_select( input=torch.index_select(input=J, dim=-2, index=i_tensor), dim=-1, index=j_tensor ).flatten(start_dim=-3, end_dim=-1)
        J_ji = torch.index_select( input=torch.index_select(input=J, dim=-2, index=i_tensor), dim=-1, index=j_tensor ).flatten(start_dim=-3, end_dim=-1)
        return 4*beta*s_i*s_j*(J_ij+J_ji)

# We can also get the changes for all choices of i together.
# If i != j, deltaDeltaE[i,j] = 4*beta*s1[i]*s1[j]*(J[j,i]+J[i,j]), so we can get all of them with deltaDeltaE[:,j] = 4*beta*s1*s1[j]*(J[j,:]+J[:,j])
def get_change_in_all_changes_in_energy_if_flipped(j:torch.int, s:torch.Tensor, h:torch.Tensor, J:torch.Tensor, beta:float=0.5):
    j_tensor = torch.tensor([j], dtype=torch.int, device=J.device)
    J_row = torch.index_select(input=J, dim=-1, index=j_tensor).squeeze(dim=-1)
    J_col = torch.index_select(input=J, dim=-2, index=j_tensor).squeeze(dim=-2)
    s_j = torch.index_select(input=s, dim=-1, index=j_tensor)
    change_in_changes = 4*beta*s*s_j*(J_row+J_col)
    change_in_changes_self = -2*get_energy_change_if_flipped(i=j, s=s, h=h, J=J, beta=beta).unsqueeze(dim=-1)
    change_in_changes.index_copy_(dim=-1, index=j_tensor, source=change_in_changes_self)
    return change_in_changes

# We can even get the changes for all [i,j] pairs.
# For each off-diagonal element, deltaDeltaE[i,j] = 4*beta*s1[i]*s1[j]*(J[j,i]+J[i,j]), so deltaDeltaE = 4*beta*s1[:,None]*s[None,:]*( J + J.transpose(dim0=-2, dim1=-1) )
def get_all_changes_in_all_changes_in_energy_if_flipped(s:torch.Tensor, h:torch.Tensor, J:torch.Tensor, beta:float=0.5):
    changes_off_diagonal = 4*beta*s.unsqueeze(dim=-1)*s.unsqueeze(dim=-2)*( J + torch.transpose(J, dim0=-2, dim1=-1) )
    changes_on_diagonal = -2*get_energy_change_if_flipped_for_each(s=s, h=h, J=J, beta=beta)
    diag_changes = torch.diag_embed( input=torch.diagonal(input=changes_on_diagonal, dim1=-2, dim2=-1), dim1=-2, dim2=-1 )
    return changes_off_diagonal - diag_changes + torch.diag_embed(input=changes_on_diagonal, dim1=-2, dim2=-1)

# Consider what we actually need when simulating the model.
# The probability of a given flip being considered does not depend on the absolute energy of the system, only the change in energy that the flip would produce.
# We could calculate the change for every flip we consider when we consider it, but this may not be optimal.
# The calculation of the energy change for a single flip takes O(N) operations.
# In most cases, only a minority of prospective flips actually occur.
# When a flip does not occur, the state does not change, so what the energy changes for different possible flips are does not change.
# Suppose we calculate all the changes in energy for all potential flips at the start and only update them when a flip actually occurs.
# When no flip occurs, we only have the unit-time check that takes in the energy change of the flip and outputs the decision.
# When a flip does occur, we need to update what the energy changes would be for all the nodes.
# Doing this calculation from scratch takes O(N^2) operations.
# However, getting the changes in changes in energies only takes O(N) time.
# Each element other than the one for the flipped node itself takes unit time, since we are just calculating 4*beta*s1[i]*s1[j]*(J[j,i]+J[i,j]).
# Calculating the changes of the changes for all of them thus takes O(N) time.
# The change for the flipped element is equal to -2 times the change in energy from flipping it.
# This would take O(N) time to compute from scratch, but we already have the change in energy, so we can just multiply it by -2, which takes unit time.
# Note that deltaE[i] + -2*deltaE[i] = -deltaE[i], meaning that flipping the node and flipping it back have inverse effects on the energy of the system.
# Take a closer look:
# deltaE[i] = -2*s1[i]*(  h[i] + beta*torch.sum( (J[:,i] + J[i,:])*s1 )  ) + 4*beta*J[i,i] takes N + 5 multiplications and 2*N + 2 additions.
# deltaDeltaE[:,j] = (4*beta*s1[j])*s1*(J[j,:]+J[:,j]) takes 2*N + 2 multiplications and N additions.
# Multiplications may be more costly than addition, making the full set of deltaDeltaE calculations more costly than the single deltaE calculation,
# but the difference is at most a factor of 2.

# In either case, we can eliminate one set of N additions by pre-symmetrizing J: J_sym = J + J.transpose(dim0=-2, dim1=-1).
# Now, deltaE[i] = -2*s1[i]*( h[i] + beta*torch.sum(J_sym[:,i]*s1) ) + 2*beta*J_sym[i,i]
# and deltaDeltaE[:,j] = (4*beta*s1[j])*s1*J_sym[:,j].
# We can also save some steps by pre-zeroing the diagonal of J_sym so that we adding
# deltaDeltaE[:,j] = (4*beta*s1[j])*s1*J_sym[:,j] to deltaE does not affect the value of deltaE[j].
def get_J_sym(J:torch.Tensor):
    J_sym = J + J.transpose(dim0=-2, dim1=-1)
    J_sym -= torch.diag_embed( input=torch.diagonal(input=J_sym, dim1=-2, dim2=-1), dim1=-2, dim2=-1 )
    return J_sym

# Note that all our calculations above are in terms of the unchanged state.
# As such, we need to update the energy before updating the state.
# Package the two actions together in one function so that we do not forget and put them in the wrong order.

# How do we do this for batched Ising models?
# We use the balanced Metropolis algorithm where we iterate over the nodes in order and either do or do not flip each one.
# When we have a batch of models, the decisions of the models at a given node are independent.
# If one model does not do the flip, we do not update it.
# As such, we need to select out the models that do perform the flip and only operate on them.

# Trying to allow for multiple batch dimensions is getting too complicated.
# For the following code, we assume one batch dimension.
# Let N be the number of nodes and B the number of independent models.
# In fact, for simplicity, all Tensors will have the same number of dimensions.
# This avoids ambiguity when broadcasting vector operations.
# J is B x N x N.
# h, s, and delta_energy are B x N x 1.

# To make it easier to keep track of things, package everything together in one PyTorch Module.

class IsingModel(torch.nn.Module):

    def __init__(self, reps_per_subject:int, num_subjects:int, num_nodes:int, beta:torch.float=0.5, dtype=torch.float, device='cpu'):
        super(IsingModel, self).__init__()
        # Keep track of both how large the model is along each dimension and which dimension is which.
        self.num_nodes = num_nodes
        self.rep_dim = 0
        self.subject_dim = 1
        self.node_dim0 = 2# rows of J
        self.node_dim1 = 3# columns of J
        self.beta = beta
        self.betaneg2 = -2.0 * beta
        self.inv_betaneg2 = 1.0/self.betaneg2
        column_vector_size = (reps_per_subject, num_subjects, num_nodes)
        matrix_size = (reps_per_subject, num_subjects, num_nodes, num_nodes)
        # Set the model parameters.
        # Let h start out as 0.
        # Since we binarize the data to which we are fitting to have about as many - as + states,
        # we expect that, in most fitted models, it will be close to 0 anyway.
        self.h = torch.zeros(size=column_vector_size, dtype=dtype, device=device)
        # Randomize J so that it starts out with about as many positive couplings as negarive ones.
        J = torch.randn(size=matrix_size, dtype=dtype, device=device)
        # We only ever use the symmetric version with the diagonal zeroed.
        self.J_sym = J + J.transpose(dim0=self.node_dim0, dim1=self.node_dim1)
        self.J_sym -= torch.diag_embed( input=torch.diagonal(self.J_sym, dim1=self.node_dim0, dim2=self.node_dim1), dim1=self.node_dim0, dim2=self.node_dim1 )
        # Each state is either -1 or +1.
        # Even though it only takes on integer values, we give it the same floating point data type as h, J, and delta_energy,
        # because we frequently multiply it by these other Tensors.
        # If we set it to an integer, we would have to cast it to the float type for every operation.
        self.s = 2*torch.randint(low=0, high=1, size=column_vector_size, dtype=dtype, device=device) - 1
        # Precalculate what the energy change would be for each possible choice of flip.
        # If we adjust the model parameters h and J, we need to recalculate all of the energies from scratch, which takes O(N^2) time.
        # However, when just performing an ordinary Metropolis simulation step, we only need to perform any updates if any nodes flip.
        # Furthermore, we can update the existing delta_energy values to reflect the effect of a single flip in O(N) time.
        # self.calculate_delta_energy()
        # Pre-allocate some space for storing the means and covariances of the node states.
        # We use these when fitting the model.
        self.s_mean = self.s.clone()# B x N x 1
        self.s_cov = self.s[:,:,:,None] * self.s[:,:,None,:]# * torch.transpose(self.s, dim0=node_dim0, dim1=node_dim1)# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
    
    def forward(self):
        self.do_ising_model_step()
        return self.s

    def do_ising_model_step(self):
        # deltaE[i] = -2*s1[i]*( h[i] + 2*beta*toch.sum(J[i,:]*s1) )
        # P_flip[i] = torch.exp( -beta*deltaE[i] )
        # If deltaE < 0, then P_flip > 1, so it will definitely flip.
        # If deltaE > 0, the 0 < P_flip < 1, smaller for larger deltaE.
        # It is faster to just pre-generate all our randomly selected floats between 0 and 1.
        # rand_choice = torch.rand_like(input=self.s)# B x N x 1
        # for j in torch.arange(start=0, end=self.num_nodes, step=1, dtype=torch.int, device=self.s.device):
        #     self.s[:,:,j] *= 1.0 - 2.0*(   rand_choice[:,:,j] < (  self.betaneg2 * self.s[:,:,j] * ( self.h[:,:,j] + torch.sum(self.J_sym[:,:,:,j]*self.s, dim=-1) )  ).exp()   ).float()
        rand_choice = self.inv_betaneg2 * torch.rand_like(input=self.s).log()# B x N x 1
        for j in torch.arange(start=0, end=self.num_nodes, step=1, dtype=torch.int, device=self.s.device):
            self.s[:,:,j] *= 1.0 - 2.0*(   rand_choice[:,:,j] > (  self.s[:,:,j] * ( self.h[:,:,j] + torch.sum(self.J_sym[:,:,:,j]*self.s, dim=-1) )  )   ).float()
        # log_rand_choice_over_betaneg2 = self.inv_betaneg2*torch.rand_like(input=self.s).log()# B x N x 1
        # for j in torch.arange(start=0, end=self.num_nodes, step=1, dtype=torch.int, device=self.s.device):
        #     self.s[:,:,j] *= 1.0 - 2.0*(   log_rand_choice_over_betaneg2[:,:,j] < (  self.s[:,:,j] * ( self.h[:,:,j] + torch.sum(self.J_sym[:,:,:,j]*self.s, dim=-1) )  )   ).float()
    
    def calculate_delta_energy(self):
        return -2.0 * self.s * ( self.h + torch.sum(self.J_sym * self.s[:,:,None,:], dim=-1, keepdim=False) )
    
    # Simulate the Ising model, and record the mean and covariance of the state.
    def simulate_and_record_mean_and_cov(self, num_steps:int):
        self.s_mean.zero_()
        self.s_cov.zero_()
        # flips_per_region = torch.zeros_like(self.s)
        for _ in range(num_steps):
            # s_pre = self.s.clone()
            self.do_ising_model_step()
            # flips_per_region += (self.s - s_pre).abs()/2.0
            self.s_mean += self.s# B x N x 1
            self.s_cov += self.s[:,:,:,None] * self.s[:,:,None,:]# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
        # print('flips per region')
        # print( flips_per_region.squeeze().mean(dim=0) )
        self.s_mean /= num_steps
        self.s_cov /= num_steps
    
    # Simulate the Ising model, and record the mean and covariance of the state.
    def simulate_and_record_mean_and_cov_faster(self, num_steps:int):
        self.s_mean.zero_()
        self.s_cov.zero_()
        # flips_per_region = torch.zeros_like(self.s)
        for _ in range(num_steps):
            log_rand_choice_over_betaneg2 = self.inv_betaneg2*torch.rand_like(input=self.s).log()# B x N x 1
            for n in torch.arange(start=0, end=self.num_nodes, step=1, dtype=torch.int, device=self.s.device):
                self.s[:,:,n] *= 1.0 - 2.0*(   log_rand_choice_over_betaneg2[:,:,n] < (  self.s[:,:,n] * ( self.h[:,:,n] + torch.sum(self.J_sym[:,:,:,n]*self.s, dim=-1) )  )   ).float()
            self.s_mean += self.s# B x N x 1
            self.s_cov += self.s[:,:,:,None] * self.s[:,:,None,:]# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
        # print('flips per region')
        # print( flips_per_region.squeeze().mean(dim=0) )
        self.s_mean /= num_steps
        self.s_cov /= num_steps
    
    # Simulate the Ising model.
    # Compare the means and covariances of the nodes to target values, and adjust as appropriate.
    # target_mean should have the same dimensions as self.s_mean, B x N x 1.
    # target_cov should have the same dimensions as self.s_cov, B x N x N.
    def update_params(self, target_mean:torch.Tensor, target_cov:torch.Tensor, learning_rate:torch.float=0.001):
        # print( 'delta h', learning_rate * (target_mean - self.s_mean) )
        self.h += learning_rate * (target_mean - self.s_mean)
        # print( 'delta J', learning_rate * (target_cov - self.s_cov) )
        self.J_sym += learning_rate * (target_cov - self.s_cov)
    
    # Simulate the Ising model.
    # Compare the means and covariances of the nodes to target values, and adjust as appropriate.
    # Use an exponent to better-condition the learning.
    # To preserve the sign of the change, it should be an odd integer or reciprocal of an odd integer.
    def update_params_pow(self, target_mean:torch.Tensor, target_cov:torch.Tensor, learning_rate:torch.float=0.001, power:torch.float=1/3):
        # print( 'delta h', learning_rate * (target_mean - self.s_mean) )
        self.h += learning_rate * torch.pow( input=(target_mean - self.s_mean), exponent=power )
        # print( 'delta J', learning_rate * (target_cov - self.s_cov) )
        self.J_sym += learning_rate * torch.pow( input=(target_cov - self.s_cov), exponent=power )
    
    # Simulate the Ising model.
    # Compare the means and covariances of the nodes to target values, and adjust as appropriate.
    # Use the logarithm to rescale the rate of change.
    def update_params_log(self, target_mean:torch.Tensor, target_cov:torch.Tensor, learning_rate:torch.float=0.001):
        # print( 'delta h', learning_rate * torch.sign(target_mean) * torch.log( torch.abs(target_mean / self.s_mean) ) )
        self.h += learning_rate * torch.sign(target_mean - self.s_mean) * torch.abs(  torch.log( torch.abs(target_mean / self.s_mean) )  )
        # print( 'delta J', learning_rate * torch.sign(target_cov) * torch.log( torch.abs(target_cov / self.s_cov) ) )
        self.J_sym += learning_rate * torch.sign(target_cov - self.s_cov) * torch.abs(  torch.log( torch.abs(target_cov / self.s_cov) )  )
    
    # data_ts is a time series with dimensions B x N x T where T can be any integer >= 1.
    def fit(self, data_means:torch.Tensor, data_covs:torch.Tensor, num_epochs:int=1, window_length:int=50, learning_rate:torch.float=0.001):
        num_windows = data_means.size(dim=-1)
        for _ in range(num_epochs):
            for window in range(num_windows):
                self.simulate_and_record_mean_and_cov(num_steps=window_length)
                self.update_params(target_mean=data_means[:,:,:,window], target_cov=data_covs[:,:,:,:,window], learning_rate=learning_rate)
    
    # data_ts is a time series with dimensions B x N x T where T can be any integer >= 1.
    def fit_faster(self, data_means:torch.Tensor, data_covs:torch.Tensor, num_epochs:int=1, window_length:int=50, learning_rate:torch.float=0.001):
        num_windows = data_means.size(dim=-1)
        for _ in range(num_epochs):
            for window_index in range(num_windows):
                self.s_mean.zero_()
                self.s_cov.zero_()
                # flips_per_region = torch.zeros_like(self.s)
                for _ in range(window_length):
                    log_rand_choice_over_betaneg2 = self.inv_betaneg2*torch.rand_like(input=self.s).log()# B x N x 1
                    for n in torch.arange(start=0, end=self.num_nodes, step=1, dtype=torch.int, device=self.s.device):
                        self.s[:,:,n] *= 1.0 - 2.0*(   log_rand_choice_over_betaneg2[:,:,n] > (  self.s[:,:,n] * ( self.h[:,:,n] + torch.sum(self.J_sym[:,:,:,n]*self.s, dim=-1) )  )   ).float()
                    self.s_mean += self.s# B x N x 1
                    self.s_cov += self.s[:,:,:,None] * self.s[:,:,None,:]# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
                # print('flips per region')
                # print( flips_per_region.squeeze().mean(dim=0) )
                self.s_mean /= window_length
                self.s_cov /= window_length
                # print( 'delta h', learning_rate * (target_mean - self.s_mean) )
                self.h += learning_rate * (data_means[:,:,:,window_index] - self.s_mean)
                # print( 'delta J', learning_rate * (target_cov - self.s_cov) )
                self.J_sym += learning_rate * (data_covs[:,:,:,:,window_index] - self.s_cov)
    
    # data_ts is a time series with dimensions B x N x T where T can be any integer >= 1.
    def fit_cubed_faster(self, data_means:torch.Tensor, data_covs:torch.Tensor, num_epochs:int=1, window_length:int=50, learning_rate:torch.float=0.001):
        num_windows = data_means.size(dim=-1)
        for _ in range(num_epochs):
            for window_index in range(num_windows):
                self.s_mean.zero_()
                self.s_cov.zero_()
                # flips_per_region = torch.zeros_like(self.s)
                for _ in range(window_length):
                    log_rand_choice_over_betaneg2 = self.inv_betaneg2*torch.rand_like(input=self.s).log()# B x N x 1
                    for n in torch.arange(start=0, end=self.num_nodes, step=1, dtype=torch.int, device=self.s.device):
                        self.s[:,:,n] *= 1.0 - 2.0*(   log_rand_choice_over_betaneg2[:,:,n] > (  self.s[:,:,n] * ( self.h[:,:,n] + torch.sum(self.J_sym[:,:,:,n]*self.s, dim=-1) )  )   ).float()
                    self.s_mean += self.s# B x N x 1
                    self.s_cov += self.s[:,:,:,None] * self.s[:,:,None,:]# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
                # print('flips per region')
                # print( flips_per_region.squeeze().mean(dim=0) )
                self.s_mean /= window_length
                self.s_cov /= window_length
                # print( 'delta h', learning_rate * (target_mean - self.s_mean) )
                delta_mean = data_means[:,:,:,window_index] - self.s_mean
                self.h += learning_rate * delta_mean * delta_mean * delta_mean
                # print( 'delta J', learning_rate * (target_cov - self.s_cov) )
                delta_cov = data_covs[:,:,:,:,window_index] - self.s_cov
                self.J_sym += learning_rate * delta_cov * delta_cov * delta_cov
    
    # Run a simulation, and record and return the functional connectivity matrix (FC).
    # The FC has dimensions B x N x N where fc[b,i,j] is the Pearson correlation between nodes i and j in model b.
    # We replace any 0s in the denominator of the division at the end with the small number epsilon to avoid division by 0.
    def simulate_and_record_fc(self, num_steps:int, epsilon:torch.float=10e-10):
        self.simulate_and_record_mean_and_cov(num_steps=num_steps)
        return get_fc(s_mean=self.s_mean, s_cov=self.s_cov, epsilon=epsilon)
    
    # Run a simulation, and record and return the functional connectivity matrix (FC).
    # The FC has dimensions B x N x N where fc[b,i,j] is the Pearson correlation between nodes i and j in model b.
    # We replace any 0s in the denominator of the division at the end with the small number epsilon to avoid division by 0.
    def simulate_and_record_fc_faster(self, num_steps:int, epsilon:torch.float=10e-10):
        self.simulate_and_record_mean_and_cov_faster(num_steps=num_steps)
        return get_fc(s_mean=self.s_mean, s_cov=self.s_cov, epsilon=epsilon)
    
    # Run a simulation, and record and return the functional connectivity matrix (FC).
    # The FC has dimensions B x N x N where fc[b,i,j] is the Pearson correlation between nodes i and j in model b.
    # We replace any 0s in the denominator of the division at the end with the small number epsilon to avoid division by 0.
    def simulate_and_record_fc_and_fim(self, num_steps:int, epsilon:torch.float=10e-10):
        self.s_mean.zero_()
        self.s_cov.zero_()
        reps_per_subject, num_subjects, num_nodes = self.s.size()
        num_params = num_nodes + ( num_nodes*(num_nodes-1) )//2
        params = torch.zeros( (reps_per_subject, num_subjects, num_params), dtype=self.s.dtype, device=self.s.device )
        param_cov = torch.zeros( (reps_per_subject, num_subjects, num_params, num_params), dtype=self.s.dtype, device=self.s.device )
        # flips_per_region = torch.zeros_like(self.s)
        for _ in range(num_steps):
            # s_pre = self.s.clone()
            self.do_ising_model_step()
            # flips_per_region += (self.s - s_pre).abs()/2.0
            self.s_mean += self.s# B x N x 1
            s_outer = self.s[:,:,:,None] * self.s[:,:,None,:]# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            self.s_cov += s_outer
            params[:,:,:num_nodes] = self.s
            params[:,:,num_nodes:] = get_triu_flattened(s_outer)
            param_cov += params[:,:,:,None] * params[:,:,None,:]
        # print('flips per region')
        # print( flips_per_region.squeeze().mean(dim=0) )
        self.s_mean /= num_steps
        self.s_cov /= num_steps
        param_cov /= num_steps
        param_mean = torch.cat(  ( self.s_mean, get_triu_flattened(self.s_cov) ), dim=-1  )
        fim = param_cov - param_mean[:,:,:,None] * param_mean[:,:,None,:]
        return get_fc(s_mean=self.s_mean, s_cov=self.s_cov, epsilon=epsilon), fim
    
    # Run a simulation, and record and return the functional connectivity matrix (FC).
    # The FC has dimensions B x N x N where fc[b,i,j] is the Pearson correlation between nodes i and j in model b.
    # We replace any 0s in the denominator of the division at the end with the small number epsilon to avoid division by 0.
    def simulate_and_record_fc_and_fim_faster(self, num_steps:int, epsilon:torch.float=10e-10):
        self.s_mean.zero_()
        self.s_cov.zero_()
        reps_per_subject, num_subjects, num_nodes = self.s.size()
        num_params = num_nodes + ( num_nodes*(num_nodes-1) )//2
        params = torch.zeros( (reps_per_subject, num_subjects, num_params), dtype=self.s.dtype, device=self.s.device )
        param_cov = torch.zeros( (reps_per_subject, num_subjects, num_params, num_params), dtype=self.s.dtype, device=self.s.device )
        # flips_per_region = torch.zeros_like(self.s)
        for _ in range(num_steps):
            # s_pre = self.s.clone()
            log_rand_choice_over_betaneg2 = self.inv_betaneg2*torch.rand_like(input=self.s).log()# B x N x 1
            for n in torch.arange(start=0, end=self.num_nodes, step=1, dtype=torch.int, device=self.s.device):
                self.s[:,:,n] *= 1.0 - 2.0*(   log_rand_choice_over_betaneg2[:,:,n] > (  self.s[:,:,n] * ( self.h[:,:,n] + torch.sum(self.J_sym[:,:,:,n]*self.s, dim=-1) )  )   ).float()
            # flips_per_region += (self.s - s_pre).abs()/2.0
            self.s_mean += self.s# B x N x 1
            s_outer = self.s[:,:,:,None] * self.s[:,:,None,:]# B x N x 1 * B x 1 x N gets broadcast to B x N x N.
            self.s_cov += s_outer
            params[:,:,:num_nodes] = self.s
            params[:,:,num_nodes:] = get_triu_flattened(s_outer)
            param_cov += params[:,:,:,None] * params[:,:,None,:]
        # print('flips per region')
        # print( flips_per_region.squeeze().mean(dim=0) )
        self.s_mean /= num_steps
        self.s_cov /= num_steps
        param_cov /= num_steps
        param_mean = torch.cat(  ( self.s_mean, get_triu_flattened(self.s_cov) ), dim=-1  )
        fim = param_cov - param_mean[:,:,:,None] * param_mean[:,:,None,:]
        return get_fc(s_mean=self.s_mean, s_cov=self.s_cov, epsilon=epsilon), fim
    
    def get_params_vec(self):
        return torch.cat(  ( self.h, get_triu_flattened(self.J_sym) ), dim=-1  )

def binarize_data_ts(data_ts:torch.Tensor, step_dim:int, threshold='median', zero_fill:torch.float=-1):
    # Choose a threshold for each 1D time series.
    # If we get 'median', we use the median value so that the numbers of -1 and +1 values are as nearly equal as possible.
    # If we get a floating point value, we use a threshold that many standard deviations above the mean.
    # Otherwise, we print a warning and use the mean.
    if threshold == 'none':
        return data_ts
    elif threshold == 'median':
        sign_threshold = torch.median(input=data_ts, dim=step_dim, keepdim=True).values
    else:
        std_ts, mean_ts = torch.std_mean(input=data_ts, dim=step_dim, keepdim=True)
        sign_threshold = mean_ts + float(threshold)*std_ts
    # elif type(threshold) == torch.float:
    # else:
    #     print(f'WARNING: threshold should be either the word "median" or a scalar floating point value, but we received {threshold}. We will use the mean.')
    #     sign_threshold = torch.mean(input=data_ts, dim=step_dim, keepdim=True)
    # Values below the threshold map to -1.
    # Values above it map to +1.
    # Values exactly at the threshold map to zero_fill.
    if (zero_fill != -1) and (zero_fill != 1):
        print(f'WARNING: zero_fill should be either -1 or +1, but we received {zero_fill}. We will use it as-is give-or-take casting to a floating-point type.')
    binary_ts = torch.sign(data_ts - sign_threshold)
    binary_ts += float(zero_fill) * (binary_ts == 0).float()
    return binary_ts

# input data_ts should be num subjects x subject ts (4) x num time points / 4 x brain regions
# output data_ts will then be num reps x num subjects x brain regions x num time points
# with values binarized to -1 or +1
def prep_individual_data_ts(data_ts:torch.Tensor, num_reps:int=1, threshold='median', zero_fill:torch.float=-1):
    return torch.transpose(  input=torch.flatten( input=binarize_data_ts(data_ts=data_ts, step_dim=-2, threshold=threshold, zero_fill=zero_fill), start_dim=1, end_dim=2 ), dim0=-2, dim1=-1  ).unsqueeze(dim=0).repeat( (num_reps,1,1,1) )

# data_ts should be reps_per_subject x num_subjects x num_nodes x T where T is some positive integer.
# If T is not divisible by window, we will truncate it to window*(T//window).
# data_means will be reps_per_subject x num_subjects x num_nodes x num_windows
# data_covs will be reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows
def get_data_means_and_covs(data_ts:torch.Tensor, window:int):
    num_steps = data_ts.size(dim=-1)
    num_windows = num_steps//window
    num_steps_in_windows = num_windows * window
    # reps_per_subject x num_subjects x num_nodes x num_windows*window -> reps_per_subject x num_subjects x num_nodes x num_windows x window
    data_ts_windows = data_ts[:,:,:,:num_steps_in_windows].unflatten( dim=-1, sizes=(num_windows, window) )
    data_means = torch.mean(data_ts_windows, dim=-1)# reps_per_subject x num_subjects x num_nodes x num_windows
    # reps_per_subject x num_subjects x num_nodes x 1 x num_windows x window * reps_per_subject x num_subjects x num_nodes x 1 x num_windows x window
    # -> reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows x window
    # mean within each window -> reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows
    data_covs = torch.mean(data_ts_windows[:,:,:,None,:,:] * data_ts_windows[:,:,None,:,:,:], dim=-1)
    return data_means, data_covs

# data_ts should be reps_per_subject x num_subjects x num_nodes x T where T is some positive integer.
# If T is not divisible by window, we will truncate it to window*(T//window).
# data_means will be reps_per_subject x num_subjects x num_nodes x num_windows
# data_covs will be reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows
def get_data_means_and_covs_slow(data_ts:torch.Tensor, window:int):
    reps_per_subject, num_subjects, num_nodes, num_steps = data_ts.size()
    num_windows = num_steps//window
    num_steps_in_windows = num_windows * window
    # reps_per_subject x num_subjects x num_nodes x num_windows*window -> reps_per_subject x num_subjects x num_nodes x num_windows x window
    data_ts_windows = data_ts[:,:,:,:num_steps_in_windows].unflatten( dim=-1, sizes=(num_windows, window) )
    data_means = torch.mean(data_ts_windows, dim=-1)# reps_per_subject x num_subjects x num_nodes x num_windows
    # reps_per_subject x num_subjects x num_nodes x 1 x num_windows x window * reps_per_subject x num_subjects x num_nodes x 1 x num_windows x window
    # -> reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows x window
    # mean within each window -> reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows
    data_covs = torch.zeros( (reps_per_subject, num_subjects, num_nodes, num_nodes, num_windows), dtype=data_ts.dtype, device=data_ts.device )
    for w in range(num_windows):
        data_covs[:,:,:,:,w] = torch.mean(data_ts_windows[:,:,:,None,w,:] * data_ts_windows[:,:,None,:,w,:], dim=-1)
    return data_means, data_covs

# data_ts should be reps_per_subject x num_subjects x num_nodes x T where T is some positive integer.
# If T is not divisible by window, we will truncate it to window*(T//window).
# data_means will be reps_per_subject x num_subjects x num_nodes x num_windows
# data_covs will be reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows
def get_data_means_and_covs_slower(data_ts:torch.Tensor, window:int):
    reps_per_subject, num_subjects, num_nodes, num_steps = data_ts.size()
    num_windows = num_steps//window
    num_steps_in_windows = num_windows * window
    # reps_per_subject x num_subjects x num_nodes x num_windows*window -> reps_per_subject x num_subjects x num_nodes x num_windows x window
    data_ts_windows = data_ts[:,:,:,:num_steps_in_windows].unflatten( dim=-1, sizes=(num_windows, window) )
    data_means = torch.mean(data_ts_windows, dim=-1)# reps_per_subject x num_subjects x num_nodes x num_windows
    # reps_per_subject x num_subjects x num_nodes x 1 x num_windows x window * reps_per_subject x num_subjects x num_nodes x 1 x num_windows x window
    # -> reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows x window
    # mean within each window -> reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows
    data_covs = torch.zeros( (reps_per_subject, num_subjects, num_nodes, num_nodes, num_windows), dtype=data_ts.dtype, device=data_ts.device )
    for w in range(num_windows):
        for t in range(window):
            data_covs[:,:,:,:,w] += data_ts_windows[:,:,:,None,w,t] * data_ts_windows[:,:,None,:,w,t]
    data_covs /= window
    return data_means, data_covs

# data_ts should be reps_per_subject x num_subjects x num_nodes x T where T is some positive integer.
# If T is not divisible by window, we will truncate it to window*(T//window).
# data_means will be reps_per_subject x num_subjects x num_nodes x num_windows
# data_covs will be reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows
# data_covs_covs will be reps_per_subject x num_subjects x num_params x num_params x num_windows
# where num_params = num_nodes + num_nodes*(num_nodes-1)//2
def get_param_means_and_covs_slower(data_ts:torch.Tensor, window:int):
    reps_per_subject, num_subjects, num_nodes, num_steps = data_ts.size()
    num_windows = num_steps//window
    num_steps_in_windows = num_windows * window
    # reps_per_subject x num_subjects x num_nodes x num_windows*window -> reps_per_subject x num_subjects x num_nodes x num_windows x window
    data_ts_windows = data_ts[:,:,:,:num_steps_in_windows].unflatten( dim=-1, sizes=(num_windows, window) )
    # reps_per_subject x num_subjects x num_nodes x 1 x num_windows x window * reps_per_subject x num_subjects x num_nodes x 1 x num_windows x window
    # -> reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows x window
    # mean within each window -> reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows
    num_params = num_nodes + ( num_nodes*(num_nodes-1) )//2
    params = torch.zeros( (reps_per_subject, num_subjects, num_params), dtype=data_ts.dtype, device=data_ts.device )
    param_means = torch.zeros( (reps_per_subject, num_subjects, num_params, num_windows), dtype=data_ts.dtype, device=data_ts.device )
    param_covs = param_means[:,:,:,None,:] * param_means[:,:,None,:,:]
    for w in range(num_windows):
        for t in range(window):
            data_state = data_ts_windows[:,:,:,w,t]
            data_cross = data_state[:,:,:,None] * data_state[:,:,None,:]
            params[:,:,:num_nodes] = data_state
            params[:,:,num_nodes:] = get_triu_flattened(data_cross)
            param_means[:,:,:,w] += params
            param_covs[:,:,:,:,w] += params[:,:,:,None] * params[:,:,None,:]
    param_means /= window
    param_covs /= window
    param_covs -= param_means[:,:,:,None,:] * param_means[:,:,None,:,:]
    return param_means, param_covs

def get_fc(s_mean:torch.Tensor, s_cov:torch.Tensor, epsilon:torch.float=10e-10):
    # s_mean should be reps_per_subject x num_subjects x num_nodes x num_windows
    # s_cov should be reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows
    # data_mean and data_cov are averaged over the time dimension with models and nodes or node pairs remaining separate.
    # We do not assume the state is binarized to -1, +1, so we need to retrieve the actual mean of the square from the diagonal of the product means.
    s_squared_mean = torch.diagonal(input=s_cov, dim1=-2, dim2=-1)# .unsqueeze(dim=-1)
    # The standard deviation of s is then sqrt( mean(s^2) - mean(s)*mean(s) ).
    s_std = torch.sqrt( s_squared_mean - s_mean.square() )
    # Center the covariance by taking mean(s * s^T) - mean(s) * mean(s)^T.
    s_cov_centered = s_cov - s_mean[:,:,:,None] * s_mean[:,:,None,:]# torch.transpose(self.s_mean, dim0=self.node_dim0, dim1=self.node_dim1)
    # The Pearson correlation is then the centered covariance rescaled by the outer product of the standard deviation with itself.
    std_product = s_std[:,:,:,None] * s_std[:,:,None,:]#s_std * s_std.transpose(dim0=self.node_dim0, dim1=self.node_dim1)
    # std_product += epsilon*(std_product == 0).float()
    return s_cov_centered/(std_product + epsilon)

def get_fc_binarized(s_mean:torch.Tensor, s_cov:torch.Tensor, epsilon:torch.float=10e-10):
    # s_mean should be reps_per_subject x num_subjects x num_nodes x num_windows
    # s_cov should be reps_per_subject x num_subjects x num_nodes x num_nodes x num_windows
    # data_mean and data_cov are averaged over the time dimension with models and nodes or node pairs remaining separate.
    # A state is either -1 or +1. The square of either of these is 1, so mean(s^2) is.
    # Alternately, we can take
    # s_squared_mean = torch.diagonal(input=s_cov, dim1=-2, dim2=-1)# .unsqueeze(dim=-1)
    # The standard deviation of s is then sqrt( mean(s^2) - mean(s)*mean(s) ).
    s_std = torch.sqrt( 1 - s_mean.square() )
    # Center the covariance by taking mean(s * s^T) - mean(s) * mean(s)^T.
    s_cov_centered = s_cov - s_mean[:,:,:,None] * s_mean[:,:,None,:]# torch.transpose(self.s_mean, dim0=self.node_dim0, dim1=self.node_dim1)
    # The Pearson correlation is then the centered covariance rescaled by the outer product of the standard deviation with itself.
    std_product = s_std[:,:,:,None] * s_std[:,:,None,:]#s_std * s_std.transpose(dim0=self.node_dim0, dim1=self.node_dim1)
    # std_product += epsilon*(std_product == 0).float()
    return s_cov_centered/(std_product + epsilon)

# Get the elements of a 2D matrix above the diagonal.
# We assume the second-to-last dimension is the rows and the last dimension the columns.
# Any dimensions prior to the last two are considered batch dimensions. 
def get_triu_flattened(mat:torch.Tensor):
    num_rows = mat.size(dim=-2)
    num_cols = mat.size(dim=-1)
    row_indices = torch.arange(end=num_rows, dtype=mat.dtype, device=mat.device).unsqueeze(dim=-1)
    col_indices = torch.arange(end=num_cols, dtype=mat.dtype, device=mat.device).unsqueeze(dim=-2)
    triu_indices = (row_indices < col_indices).flatten().nonzero().flatten()
    return torch.index_select( input=mat.flatten(start_dim=-2, end_dim=-1), dim=-1, index=triu_indices )

def get_triu_corr(mat1:torch.Tensor, mat2:torch.Tensor, epsilon:torch.float=10e-10):
    mat1_ut = get_triu_flattened(mat1)
    mat2_ut = get_triu_flattened(mat2)
    mean_of_product = torch.mean(mat1_ut * mat2_ut, dim=-1, keepdim=True)
    product_of_means = torch.mean(mat1_ut, dim=-1, keepdim=True) * torch.mean(mat2_ut, dim=-1, keepdim=True)
    mat1_ut_std = torch.sqrt(  torch.mean( mat1_ut.square(), dim=-1, keepdim=True ) - torch.mean(mat1_ut, dim=-1, keepdim=True).square()  )
    mat2_ut_std = torch.sqrt(  torch.mean( mat2_ut.square(), dim=-1, keepdim=True ) - torch.mean(mat2_ut, dim=-1, keepdim=True).square()  )
    std_product = mat1_ut_std * mat2_ut_std
    std_product += (std_product == 0.0).float() * epsilon
    corr = ( (mean_of_product - product_of_means)/std_product ).squeeze()#.unsqueeze(dim=-1)
    return corr

def get_rmse(mat1:torch.Tensor, mat2:torch.Tensor):
    # We only take the mean across the last dimension so that we have one value per batch, however many batch dimensions we have.
    return torch.sqrt(  torch.mean(  (mat1 - mat2).square(), dim=-1, keepdim=True  )  ).squeeze()#.unsqueeze(dim=-1)

def get_triu_rmse(mat1:torch.Tensor, mat2:torch.Tensor):
    # We do the subtraction first so that we only need to call get_triu_flattened() once.
    # Subtraction is a relatively cheap operation, especially with GPU Tensors.
    return torch.sqrt(  torch.mean(  ( get_triu_flattened(mat1 - mat2) ).square(), dim=-1, keepdim=True  )  ).squeeze()#.unsqueeze(dim=-1)
