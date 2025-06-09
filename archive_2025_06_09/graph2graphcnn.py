# Graph2GraphCNN is a stacked graph convolutional network that maps from one set of node and edge features to another.
# inputs:
# node input features Tensor of size num_graphs x num_nodes x num_input_node_features
# edge input features Tensor of size num_graphs x num_nodes x num_nodes x num_input_edge_features
# outputs:
# node output features Tensor of size num_graphs x num_nodes x num_output_node_features
# edge output features Tensor of size num_graphs x num_nodes x num_nodes x num_output_edge_features
# The model consists of alternating layers of MultiLayerPerceptron (MLP) models that act on individual nodes and edges and graph message passing.
# Each MLP consists of two MLPs.
# The node transformation MLP transforms node features from one representation to another.
# The edge transformation MLP takes in the representations of the edge and its endpoints and outputs a new edge representation.
# In the message passing layer, we use the representations of the in-edges of a node as the weights with which to take a weighted average of the node representations of its neighbors.
# We then concatenate this to its node representation to be passed in to the next node transformation MLP.

import torch
from collections import OrderedDict

float_type = torch.float
int_type = torch.int

def get_diag_mask(num_nodes:int, dtype=float_type, device='cpu'):
    return torch.ones( size=(1, num_nodes, num_nodes, 1), dtype=dtype, device=device ) - torch.eye(n=num_nodes, dtype=dtype, device=device).unsqueeze(dim=0).unsqueeze(dim=-1)

class GraphMSELoss(torch.nn.Module):
    def __init__(self, num_nodes:int, dtype=float_type, device='cpu'):
        super(GraphMSELoss, self).__init__()
        self.simple_mse_loss = torch.nn.MSELoss()
        self.num_nodes = num_nodes
        self.num_nodes_squared = num_nodes * num_nodes
        self.num_nodes_squared_minus_num_nodes = self.num_nodes_squared - self.num_nodes
        self.diag_mask = get_diag_mask(num_nodes=num_nodes, dtype=dtype, device=device)
    def forward(self, node_features_pred:torch.Tensor, edge_features_pred:torch.Tensor, node_features_target:torch.Tensor, edge_features_target:torch.Tensor):
        return ( self.num_nodes * self.simple_mse_loss(node_features_pred, node_features_target) + self.num_nodes_squared * self.simple_mse_loss(edge_features_pred*self.diag_mask, edge_features_target*self.diag_mask) )/(self.num_nodes_squared)

# Create a Graph2GraphCNN with any depth containing MLPs that themselves have any desired widths and depths.
# layer_widths is a list of pairs of lists.
# For example, [ ([2,4,5],[3,6,5]), ([10,8,7],[15,11,9]) ] gives us a GCNN with the following steps:
# MLPPair 0
#   The node transformation MLP takes in 2-vectors of node features, has 1 hidden layer of width 4 and one output layer of width 5, outputs 5-vectors of node features.
#   The edge transformation MLP takes in 3-vectors of edge features, has 1 hidden layer of width 6 and one output layer of width 5, outputs 5-vectors of edge features.
# GraphMessagePass 0
#   The node aggregation aggregates the neighbors of each node and concatenates the result to the node representation, producing 10-vectors of node features.
#   The edge aggregation concatenates the endpoint node features of each edge to the edge features, producing 15-vectors of edge features.
# MLPPair 1
#   The node transformation MLP takes in 10-vectors of node features, has 1 hidden layer of width 8 and one output layer of width 7, outputs 7-vectors of node features.
#   The edge transformation MLP takes in 15-vectors of edge features, has 1 hidden layer of width 11 and one output layer of width 9, outputs 9-vectors of edge features.
#   At the end, we multiply the edge features Tensor by diag_mask to 0 out the diagonal of each edge feature matrix, preventing the graph from having self-loops.
class Graph2GraphCNN(torch.nn.Module):

    def __init__(self, layer_widths:list, dtype=float_type, device='cpu'):
        super(Graph2GraphCNN, self).__init__()
        # print('making Graph2GraphCNN with layer widths')
        # print(layer_widths)
        self.dtype = dtype
        self.device = device
        self.layers = torch.nn.Sequential(   OrderedDict(  [ self.get_label_layer_pair(index=i, layer_type=layer_type, node_widths=layer_widths[i][0], edge_widths=layer_widths[i][1]) for i in range( len(layer_widths)-1 ) for layer_type in ('mlp_pair', 'graph_conv') ] + [ self.get_label_layer_pair( index=len(layer_widths)-1, layer_type='mlp_pair', node_widths=layer_widths[-1][0], edge_widths=layer_widths[-1][1] ) ]  )   )
        self.diag_mask = get_diag_mask( num_nodes=layer_widths[-1][1][-1], dtype=self.dtype, device=self.device )
    
    # Have the forward function output the entropy of the given state.
    def forward(self, node_in_features:torch.Tensor, edge_in_features:torch.Tensor):
        node_out_features, edge_out_features = self.layers( (node_in_features, edge_in_features) )
        return node_out_features, (edge_out_features * self.diag_mask)
    
    def get_label_layer_pair(self, index:int, layer_type:str, node_widths:list, edge_widths:list):
        if layer_type == 'mlp_pair':
            return ( f'mlp_pair_{index}_nodes_{node_widths[0]}_to_{node_widths[-1]}_edges_{edge_widths[0]}_to_{edge_widths[-1]}', MLPPair(layer_widths_left=node_widths, layer_widths_right=edge_widths, dtype=self.dtype, device=self.device) )
        else:
            return ( f'graph_conv_{index}_nodes_{node_widths[-1]}_to_{2*node_widths[-1]}_edges_{edge_widths[-1]}_to_{3*edge_widths[-1]}', GraphMessagePass() )


# This subclass of Graph2GraphCNN has the same architecture but enforces a more uniform structure, making initialization easier.
class UniformGraph2GraphCNN(Graph2GraphCNN):

    def __init__(self, num_node_in_features:int=7, num_edge_in_features:int=1, num_node_out_features:int=1, num_edge_out_features:int=1, mlp_hidden_width:int=7, num_node_mlp_hidden_layers:int=1, num_edge_mlp_hidden_layers:int=1, num_graph_message_passes:int=1, dtype=float_type, device='cpu'):
        node_width_after_graph = 2*mlp_hidden_width
        edge_width_after_graph = 3*mlp_hidden_width
        layer_widths = [ ([num_node_in_features] + [mlp_hidden_width]*(num_node_mlp_hidden_layers),[num_edge_in_features] + [mlp_hidden_width]*num_edge_mlp_hidden_layers) ] + [ ([node_width_after_graph] + [mlp_hidden_width]*num_node_mlp_hidden_layers, [edge_width_after_graph] + [mlp_hidden_width]*num_edge_mlp_hidden_layers) ]*(num_graph_message_passes-1) + [ ([node_width_after_graph] + [mlp_hidden_width]*(num_node_mlp_hidden_layers-1) + [num_node_out_features], [edge_width_after_graph] + [mlp_hidden_width]*(num_edge_mlp_hidden_layers-1) + [num_edge_out_features]) ]
        super(UniformGraph2GraphCNN, self).__init__(layer_widths=layer_widths, dtype=dtype, device=device)

# Take in a pair containing node representations and edge representations.
# Node representations are in a Tensor of size num_graphs x num_nodes x num_features.
# Edge representations are in a Tensor of size num_graphs x num_nodes x num_nodes x num_features.
# First, we use the edge features as weights with which to take a weighted average of node features:
# This produces a new Tensor of aggregated node features of size num_graphs x num_nodes x num_features.
# We append this to the original Tensor of node features to produce a new node representation Tensor of size num_graphs x num_nodes x 2num_features.
# We also append the original node feature Tensors of the endpoints of each edge to the edge features to get a new edge representation Tensor of size num_graphs x num_nodes x num_nodes x 3num_features.
class GraphMessagePass(torch.nn.Module):

    def __init__(self):
        super(GraphMessagePass, self).__init__()
    
    # Have the forward function output the entropy of the given state.
    def forward(self, x_pair:tuple):
        x_node = x_pair[0]
        x_edge = x_pair[1]
        num_nodes = x_node.size(dim=1)
        x_node_repeat_in_rows = x_node[:,None,:,:].repeat( (1,num_nodes,1,1) )
        x_node_repeat_in_cols = x_node[:,:,None,:].repeat( (1,1,num_nodes,1) )
        x_aggregate = torch.sum(x_edge * x_node_repeat_in_rows, dim=-2)/torch.sum(x_edge, dim=-2)
        return torch.cat( (x_node, x_aggregate), dim=-1 ), torch.cat( (x_node_repeat_in_rows, x_node_repeat_in_cols, x_edge), dim=-1 )

# Create a class with a pair of MLPs that act in parallel.
class MLPPair(torch.nn.Module):

    def __init__(self, layer_widths_left:list, layer_widths_right:list, dtype=float_type, device='cpu'):
        super(MLPPair, self).__init__()
        self.mlp_left = MultiLayerPerceptron(layer_widths=layer_widths_left, dtype=dtype, device=device)
        self.mlp_right = MultiLayerPerceptron(layer_widths=layer_widths_right, dtype=dtype, device=device)
    
    # Have the forward function output the entropy of the given state.
    def forward(self, x_pair:tuple):
        return self.mlp_left(x_pair[0]), self.mlp_right(x_pair[1])

# This class lets us define a multi-layer perceptron network just by passing in a list of layer widths.
# For example, consider the minimal case where we have just one hidden layer.
# We pass in a list of 3 positive integers, [a, b, c].
# The MLP will then have a Sequential Module with 3 layers:
# 'linear_0_a_b', torch.nn.Linear(in_features=a, out_features=b)
# 'relu_0_b', torch.nn.ReLU()
# 'output', torch.nn.Linear(in_features=b, out_features=c)
class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, layer_widths:list, dtype=float_type, device='cpu'):
        super(MultiLayerPerceptron, self).__init__()
        self.dtype = dtype
        self.device = device
        self.layers = torch.nn.Sequential(   OrderedDict(  [ self.get_label_layer_pair(index=i, layer_type=layer_type, num_in=layer_widths[i], num_out=layer_widths[i+1]) for i in range( len(layer_widths)-2 ) for layer_type in ('linear', 'relu') ] + [ self.get_label_layer_pair( index=len(layer_widths)-1, layer_type='linear', num_in=layer_widths[-2], num_out=layer_widths[-1] ) ]  )   )
    
    # Have the forward function output the entropy of the given state.
    def forward(self, x:torch.Tensor):
        return self.layers(x)
    
    def get_label_layer_pair(self, index:int, layer_type:str,  num_in:int, num_out:int):
        if layer_type == 'linear':
            return ( f'linear_{index}_{num_in}_to_{num_out}', torch.nn.Linear(in_features=num_in, out_features=num_out, dtype=self.dtype, device=self.device) )
        else:
            return ( f'relu_{index}_{num_out}_to_{num_out}', torch.nn.ReLU() )

class UniformMultiLayerPerceptron(MultiLayerPerceptron):

    def __init__(self, num_in_features:int=7, num_out_features:int=1, hidden_layer_width:int=7, num_hidden_layers:int=1, dtype=float_type, device='cpu'):
        super(UniformMultiLayerPerceptron, self).__init__(layer_widths=[num_in_features]+[hidden_layer_width]*num_hidden_layers+[num_out_features], dtype=dtype, device=device)

class MLPWithSkips(torch.nn.Module):

    def __init__(self, num_in_features:int=7, num_out_features:int=1, input_layer_width:int=14, num_input_layers:int=1, middle_layer_width:int=7, num_middle_layers:int=1, output_layer_width:int=7, num_output_layers:int=1, dtype=float_type, device='cpu'):
        super(MLPWithSkips, self).__init__()
        self.dtype = dtype
        self.device = device
        self.in_layers = UniformMultiLayerPerceptron(num_in_features=num_in_features, num_out_features=input_layer_width, hidden_layer_width=input_layer_width, num_hidden_layers=num_input_layers, dtype=dtype, device=device)
        self.middle_layers = UniformMultiLayerPerceptron(num_in_features=input_layer_width, num_out_features=middle_layer_width, hidden_layer_width=middle_layer_width, num_hidden_layers=num_middle_layers, dtype=dtype, device=device)
        self.out_layers = UniformMultiLayerPerceptron(num_in_features=(input_layer_width+middle_layer_width), num_out_features=num_out_features, hidden_layer_width=output_layer_width, num_hidden_layers=num_output_layers, dtype=dtype, device=device)
    
    # Have the forward function output the entropy of the given state.
    def forward(self, x:torch.Tensor):
        short_rep = self.in_layers(x)
        long_rep = self.middle_layers(short_rep)
        return self.out_layers(  torch.cat( (short_rep, long_rep), dim=-1 )  )
    
    def get_label_layer_pair(self, index:int, layer_type:str,  num_in:int, num_out:int):
        if layer_type == 'linear':
            return ( f'linear_{index}_{num_in}_to_{num_out}', torch.nn.Linear(in_features=num_in, out_features=num_out, dtype=self.dtype, device=self.device) )
        else:
            return ( f'relu_{index}_{num_out}_to_{num_out}', torch.nn.ReLU() )
