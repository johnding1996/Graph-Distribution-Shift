import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    3WLGNN / ThreeWLGNN
    Provably Powerful Graph Networks (Maron et al., 2019)
    https://papers.nips.cc/paper/8488-provably-powerful-graph-networks.pdf

    CODE adapted from https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch/
"""


class ThreeWLGNNNet(nn.Module):
    def __init__(self, gnn_type, num_tasks, feature_dim, n_layers=3, depth_of_mlp=2, hidden_dim=128,
                 residual=False, **model_kwargs):
        assert gnn_type == '3wlgnn'
        super(ThreeWLGNNNet, self).__init__()
        self.in_dim_node = feature_dim
        self.residual = residual

        block_features = [hidden_dim] * n_layers  # L here is the block number
        original_features_num = self.in_dim_node + 1  # Number of features of the input

        # sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = RegularBlock(depth_of_mlp, last_layer_features, next_layer_features, self.residual)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        self.fc_layers = nn.ModuleList()
        for output_features in block_features:
            # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
            fc = FullyConnected(2 * output_features, num_tasks, activation_fn=None)
            self.fc_layers.append(fc)

    def forward(self, x):
        scores = torch.tensor(0, device=x.device, dtype=x.dtype)
        for i, block in enumerate(self.reg_blocks):
            x = block(x)
            scores = self.fc_layers[i](diag_offdiag_maxpool(x)) + scores
        return scores


class ThreeWLGNNEdgeNet(nn.Module):
    def __init__(self, gnn_type, num_tasks, feature_dim, n_layers=3, depth_of_mlp=2, hidden_dim=128,
                 residual=False, **model_kwargs):

        assert gnn_type == '3wlgnn-edge'
        super(ThreeWLGNNEdgeNet, self).__init__()

        self.in_dim_node = feature_dim
        self.residual = residual

        block_features = [hidden_dim] * n_layers  # L here is the block number
        original_features_num = self.in_dim_node + self.num_bond_type + 1  # Number of features of the input

        # sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = RegularBlock(depth_of_mlp, last_layer_features, next_layer_features, self.residual)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        self.fc_layers = nn.ModuleList()
        for output_features in block_features:
            # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
            fc = FullyConnected(2 * output_features, num_tasks, activation_fn=None)
            self.fc_layers.append(fc)

    def forward(self, x_no_edge_feat, x_with_edge_feat):
        x = x_with_edge_feat

        scores = torch.tensor(0, device=x.device, dtype=x.dtype)
        for i, block in enumerate(self.reg_blocks):
            x = block(x)
            scores = self.fc_layers[i](diag_offdiag_maxpool(x)) + scores
        return scores


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class RegularBlock(nn.Module):
    """
    Imputs: N x input_depth x m x m
    Take the input through 2 parallel MLP routes, multiply the result, and add a skip-connection at the end.
    At the skip-connection, reduce the dimension back to output_depth
    """

    def __init__(self, depth_of_mlp, in_features, out_features, residual=False):
        super().__init__()

        self.residual = residual

        self.mlp1 = MlpBlock(in_features, out_features, depth_of_mlp)
        self.mlp2 = MlpBlock(in_features, out_features, depth_of_mlp)

        self.skip = SkipConnection(in_features + out_features, out_features)

        if self.residual:
            self.res_x = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        mlp1 = self.mlp1(inputs)
        mlp2 = self.mlp2(inputs)

        mult = torch.matmul(mlp1, mlp2)

        out = self.skip(in1=inputs, in2=mult)

        if self.residual:
            # Now, changing shapes from [1xdxnxn] to [nxnxd] for Linear() layer
            inputs, out = inputs.permute(3, 2, 1, 0).squeeze(), out.permute(3, 2, 1, 0).squeeze()

            residual_ = self.res_x(inputs)
            out = residual_ + out  # residual connection

            # Returning output back to original shape
            out = out.permute(2, 1, 0).unsqueeze(0)

        return out

class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    """
    def __init__(self, in_features, out_features, depth_of_mlp, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.convs = nn.ModuleList()
        for i in range(depth_of_mlp):
            self.convs.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True))
            _init_weights(self.convs[-1])
            in_features = out_features

    def forward(self, inputs):
        out = inputs
        for conv_layer in self.convs:
            out = self.activation(conv_layer(out))

        return out

class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_features: d1+d2
    :param out_features: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=True)
        _init_weights(self.conv)

    def forward(self, in1, in2):
        # in1: N x d1 x m x m
        # in2: N x d2 x m x m
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out


class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, activation_fn=nn.functional.relu):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)
        _init_weights(self.fc)

        self.activation = activation_fn

    def forward(self, input):
        out = self.fc(input)
        if self.activation is not None:
            out = self.activation(out)

        return out


def diag_offdiag_maxpool(input):
    N = input.shape[-1]

    max_diag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]  # BxS

    # with torch.no_grad():
    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * input)
    val = torch.abs(torch.add(max_val, min_val))

    min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N, N)

    max_offdiag = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]  # BxS

    return torch.cat((max_diag, max_offdiag), dim=1)  # output Bx2S


def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    # nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class LayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d).unsqueeze(0).unsqueeze(0))  # shape is 1 x 1 x d
        self.b = nn.Parameter(torch.zeros(d).unsqueeze(0).unsqueeze(0))  # shape is 1 x 1 x d

    def forward(self, x):
        # x tensor of the shape n x n x d
        mean = x.mean(dim=(0, 1), keepdim=True)
        var = x.var(dim=(0, 1), keepdim=True, unbiased=False)
        x = self.a * (x - mean) / torch.sqrt(var + 1e-6) + self.b  # shape is n x n x d
        return x