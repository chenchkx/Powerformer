from __future__ import division
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GINConv
from dgl.nn.pytorch import GINEConv
from collections import OrderedDict
import dgl
import networks.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ModuleType = Type[nn.Module]


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)  # b*118
        dim = 1

        number_of_logits = input.size(dim) # 118

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


def miniblock(
    input_size: int,
    output_size: int = 0,
    norm_layer: Optional[ModuleType] = None,
    activation: Optional[ModuleType] = None,
    linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers


class MLP(nn.Module):
    """Simple MLP backbone.

    Create a MLP of size input_dim * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * output_dim

    :param int input_dim: dimension of the input vector.
    :param int output_dim: dimension of the output vector. If set to 0, there
        is no final linear layer.
    :param hidden_sizes: shape of MLP passed in as a list, not including
        input_dim and output_dim.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: which device to create this model on. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, norm, activ in zip(
            hidden_sizes[:-1], hidden_sizes[1:], norm_layer_list, activation_list
        ):
            model += miniblock(in_dim, out_dim, norm, activ, linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)

    def forward(self, s: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.device is not None:
            s = torch.as_tensor(
                s,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
        return self.model(s.flatten(1))  # type: ignore



class ZeroNet(nn.Module):
    def forward(self, x):
        return torch.zeros(1)


class Net(nn.Module):
    def __init__(
            self, output_shape,
            base_type,
            append_hidden_shapes=[],
            append_hidden_init_func=init.basic_init,
            net_last_init_func=init.uniform_init,
            activation_func=F.relu,
            **kwargs):

        super().__init__()

        self.base = base_type(activation_func=activation_func, **kwargs)
        self.activation_func = activation_func
        append_input_shape = self.base.output_shape
        self.append_fcs = []
        for i, next_shape in enumerate(append_hidden_shapes):
            fc = nn.Linear(append_input_shape, next_shape)
            append_hidden_init_func(fc)
            self.append_fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("append_fc{}".format(i), fc)
            append_input_shape = next_shape

        self.last = nn.Linear(append_input_shape, output_shape)
        net_last_init_func(self.last)

    def forward(self, x):
        out = self.base(x)

        for append_fc in self.append_fcs:
            out = append_fc(out)
            out = self.activation_func(out)

        out = self.last(out)
        return out


class FlattenNet(Net):
    def forward(self, input):
        out = torch.cat(input, dim = -1)
        return super().forward(out)

def null_activation(x):
    return x

class ModularGatedCascadeCondNet(nn.Module):
    def __init__(self, output_shape,
            base_type, em_input_shape, input_shape,
            em_hidden_shapes,
            hidden_shapes,

            num_layers, num_modules,

            module_hidden,

            gating_hidden, num_gating_layers,

            # gated_hidden
            add_bn = True,
            pre_softmax = False,
            cond_ob = True,
            module_hidden_init_func = init.basic_init,
            last_init_func = init.uniform_init,
            activation_func = F.relu,
             **kwargs ):

        super().__init__()

        self.base = base_type( 
                        last_activation_func = null_activation,
                        input_shape = input_shape,
                        activation_func = activation_func,
                        hidden_shapes = hidden_shapes,
                        **kwargs )
        self.em_base = base_type(
                        last_activation_func = null_activation,
                        input_shape = em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes,
                        **kwargs )

        self.activation_func = activation_func

        module_input_shape = self.base.output_shape
        self.layer_modules = []

        self.num_layers = num_layers
        self.num_modules = num_modules

        for i in range(num_layers):
            layer_module = []
            for j in range( num_modules ):
                fc = nn.Linear(module_input_shape, module_hidden)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    )
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i,j), module)

            module_input_shape = module_hidden
            self.layer_modules.append(layer_module)

        self.last = nn.Linear(module_input_shape, output_shape)
        last_init_func( self.last )

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated" 
        gating_input_shape = self.em_base.output_shape
        self.gating_fcs = []
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden)
            module_hidden_init_func(gating_fc)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape,
                    num_modules * num_modules )
        last_init_func( self.gating_weight_fc_0)
        # self.gating_weight_fcs.append(self.gating_weight_fc_0)

        for layer_idx in range(num_layers-2):
            gating_weight_cond_fc = nn.Linear((layer_idx+1) * \
                                               num_modules * num_modules,
                                              gating_input_shape)
            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)

            gating_weight_fc = nn.Linear(gating_input_shape,
                                         num_modules * num_modules)
            last_init_func(gating_weight_fc)
            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)

        self.gating_weight_cond_last = nn.Linear((num_layers-1) * \
                                                 num_modules * num_modules,
                                                 gating_input_shape)
        module_hidden_init_func(self.gating_weight_cond_last)

        self.gating_weight_last = nn.Linear(gating_input_shape, num_modules)
        last_init_func( self.gating_weight_last )

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

    def forward(self, x, embedding_input, return_weights = False):
        # Return weights for visualization
        out = self.base(x)
        embedding = self.em_base(embedding_input)

        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)

        if len(self.gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []

        raw_weight = self.gating_weight_fc_0(self.activation_func(embedding))

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)
        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim = -1)

        module_outputs = [(layer_module(out)).unsqueeze(-2) \
                for layer_module in self.layer_modules[0]]

        module_outputs = torch.cat(module_outputs, dim = -2 )

        # [TODO] Optimize using 1 * 1 convolution.

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.layer_modules[i + 1]):
                module_input = (module_outputs * \
                    weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                        layer_module(module_input)
                ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim = -2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        out = self.last(out)

        if return_weights:
            return out, weights, last_weight
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout_attn=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout_attn = nn.Dropout(dropout_attn)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask.bool(), float('-inf'))

        attn = F.softmax(attn, dim=2)

        # some weights might be nan (if agent is inactive and all entities were masked)
        attn = attn.masked_fill(attn != attn, 0)

        attn = self.dropout_attn(attn)

        out = torch.bmm(attn, v)

        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, dropout_attn=0.0, dropout_attn_out=0.0):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_heads = n_heads

        self.w_k = nn.Linear(emb_dim, n_heads * emb_dim, bias=False)
        self.w_q = nn.Linear(emb_dim, n_heads * emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, n_heads * emb_dim, bias=False)

        self.attention = ScaledDotProductAttention(temperature=emb_dim ** 0.5, dropout_attn=dropout_attn)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)

        self.fc = nn.Linear(n_heads * emb_dim, emb_dim)

    def forward(self, x, mask=None):
        b, t, e = x.size()
        n_heads = self.n_heads

        k = self.w_k(x).view(b, t, n_heads, e)
        q = self.w_q(x).view(b, t, n_heads, e)
        v = self.w_v(x).view(b, t, n_heads, e)

        k = k.permute(2, 0, 1, 3).contiguous().view(n_heads * b, t, e)
        q = q.permute(2, 0, 1, 3).contiguous().view(n_heads * b, t, e)
        v = v.permute(2, 0, 1, 3).contiguous().view(n_heads * b, t, e)

        if mask is not None:
            mask = mask.repeat(n_heads, 1, 1)

        out, _ = self.attention(q, k, v, mask=mask)

        out = out.view(n_heads, b, t, e)
        out = out.permute(1, 2, 0, 3).contiguous().view(b, t, n_heads * e)

        out = self.fc(out)
        out = self.dropout_attn_out(out)

        return out

class MLPlayer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 device='cpu'):
        super(MLPlayer, self).__init__()
        self.device = device
        self.mlp = nn.Sequential(nn.Linear(in_feats, n_hidden),
                                    nn.BatchNorm1d(n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden, n_classes))

    def forward(self, nfeat):
        h = self.mlp(nfeat)
        return h


class GCNlayer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 device='cpu'):
        super(GCNlayer, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        # self.in_feats = in_feats
        self.n_layers = n_layers
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != self.n_layers:
                h = F.relu(h)
        return h

# GIN Convolution Layer
class GINlayer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 device='cpu'):
        super(GINlayer, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        # self.in_feats = in_feats
        self.n_layers = n_layers
        # input layer
        self.layers.append(GINConv(apply_func = MLPlayer(in_feats,n_hidden,n_hidden)))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GINConv(apply_func = MLPlayer(n_hidden,n_hidden,n_hidden)))
        # output layer
        self.layers.append(GINConv(apply_func = MLPlayer(n_hidden,n_hidden,n_classes)))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != self.n_layers:
                h = F.relu(h)
        return h


# GIN Convolution Layer with Edge Features
class GINElayer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 device='cpu'):
        super(GINElayer, self).__init__()
        self.device = device
        self.edge_layers = nn.ModuleList()
        self.gine_layers = nn.ModuleList()
        # self.in_feats = in_feats
        self.n_layers = n_layers
        # input layer
        self.edge_layers.append(nn.Linear(in_feats, in_feats))
        self.gine_layers.append(GINEConv(apply_func = nn.Sequential(nn.Linear(in_feats, n_hidden),
                                    nn.BatchNorm1d(n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden, n_hidden))))
        # hidden layers
        for i in range(n_layers - 1):
            self.edge_layers.append(nn.Linear(in_feats, n_hidden))
            self.gine_layers.append(GINEConv(apply_func = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                    nn.BatchNorm1d(n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden, n_hidden))))
        # output layer
        self.edge_layers.append(nn.Linear(in_feats, n_hidden))
        self.gine_layers.append(GINEConv(apply_func = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_classes))))

    def forward(self, g, nfeat, efeat):
        h = nfeat
        for i, layer in enumerate(self.gine_layers): 
            h = layer(g, h, self.edge_layers[i](efeat))
            if i != self.n_layers:
                h = F.relu(h)
        return h


# Powerformer-Factor
class SelfAttentionNetWeighted_Factor(nn.Module):
    def __init__(self, output_shape,
                 em_input_shape, state_input_shape,
                 task_num: int,
                 hidden_type='MLP', 
                 graph_u = None,
                 graph_d = None,
                 dueling_param=None, device='cuda',
                 ):

        super().__init__()

        self.em_input_shape = em_input_shape
        self.state_input_shape = state_input_shape
        self.output_shape = output_shape
        self.device = device
        self.graph_u = graph_u
        self.graph_d = graph_d
        self.use_dueling = dueling_param is not None
        self.task_num = task_num
        self.f = nn.Softmax(dim=2)
        self.f0 = nn.Softmax(dim=0)
        self.hidden_type = hidden_type
        if hidden_type in ['GIN','GCN','GIN_Factor']:
            self.is_gnn = True
        else:
            self.is_gnn = False

        if hidden_type == 'MLP_Factor':
            self.w_k_p = MLPlayer(in_feats=1, n_hidden=128, n_classes=64, device=self.device)
            self.w_v_p = MLPlayer(in_feats=1, n_hidden=128, n_classes=64, device=self.device)
            # q value convolution: Reactive power 
            self.w_k_q = MLPlayer(in_feats=1, n_hidden=128, n_classes=64, device=self.device)
            self.w_v_q = MLPlayer(in_feats=1, n_hidden=128, n_classes=64, device=self.device)  
            # v value convolution: Voltage
            self.w_k_v = MLPlayer(in_feats=1, n_hidden=128, n_classes=64, device=self.device)
            self.w_v_v = MLPlayer(in_feats=1, n_hidden=128, n_classes=64, device=self.device)  
            # t value convolution: Phase angle
            self.w_k_t = MLPlayer(in_feats=1, n_hidden=128, n_classes=64, device=self.device)
            self.w_v_t = MLPlayer(in_feats=1, n_hidden=128, n_classes=64, device=self.device)  
        elif hidden_type == 'GCN_Factor':
            self.w_k_p = GCNlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v_p = GCNlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            # q value convolution: Reactive power 
            self.w_k_q = GCNlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v_q = GCNlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)  
            # v value convolution: Voltage
            self.w_k_v = GCNlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v_v = GCNlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)  
            # t value convolution: Phase angle
            self.w_k_t = GCNlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v_t = GCNlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)      
        elif hidden_type == 'GIN_Factor':
            # p value convolution: Active power
            self.w_k_p = GINlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v_p = GINlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            # q value convolution: Reactive power 
            self.w_k_q = GINlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v_q = GINlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)  
            # v value convolution: Voltage
            self.w_k_v = GINlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v_v = GINlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)  
            # t value convolution: Phase angle
            self.w_k_t = GINlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v_t = GINlayer(in_feats=1, n_hidden=64, n_classes=64, n_layers=2, device=self.device)  
        else:
            raise Exception('Invalid layer type, must be either "GCN" or "MLP"')

        self.dim_represent = 64     
        self.w_q = nn.Sequential(OrderedDict([
            ('q1', nn.Linear(em_input_shape, 128)),
            ('qRelu', nn.ReLU()),
            ('q2', nn.Linear(128, self.dim_represent)) 
        ]))

        self.alpha = nn.Parameter(torch.Tensor([1/self.task_num*4 for _ in range(task_num*4)]), requires_grad=True)

        # dueling param: {'hidden_sizes': [128]}
        if self.use_dueling:  # dueling DQN
            Q_kwargs, V_kwargs = dueling_param

            Q_hidden_size = Q_kwargs['hidden_sizes'][0]
            V_hidden_size = V_kwargs['hidden_sizes'][0]
            print(Q_hidden_size)
            self.Q = nn.Sequential(OrderedDict([
                ('Q1', nn.Linear(self.dim_represent, Q_hidden_size)), 
                ('QRelu', nn.ReLU()),
                ('Q2', nn.Linear(Q_hidden_size, output_shape))
            ]))
            self.V = nn.Sequential(OrderedDict([
                ('v1', nn.Linear(self.dim_represent, V_hidden_size)),
                ('VRelu', nn.ReLU()),
                ('V2', nn.Linear(V_hidden_size, 1))
            ]))
        else:
            assert False, 'Powerformer must use dueling DQN architecture.'

    def forward(self, x, state=None, show_grid=False, info=None):  # x: b*(state_input_shape+task_num*em_input_shape)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if self.em_input_shape > 50:  # for case118 and case300
            state_in, embedding_in, edges_in = torch.split(x, [self.state_input_shape, self.em_input_shape * self.task_num,
                                                               self.graph_d.num_edges() * 4],
                                                           dim=1)
            embedding_in = torch.reshape(embedding_in, [-1, self.task_num, self.em_input_shape])
            edges_in_p, edges_in_q, edges_in_v, edges_in_t = torch.split(edges_in, [self.graph_d.num_edges(),
                                                                                    self.graph_d.num_edges(),
                                                                                    self.graph_d.num_edges(),
                                                                                    self.graph_d.num_edges()],
                                                                         dim=1)
        else:
            state_in, embedding_in, edges_in = torch.split(x, [self.state_input_shape, self.em_input_shape*self.task_num,
                                                                self.em_input_shape*4], dim=1)
            embedding_in = torch.reshape(embedding_in, [-1, self.task_num, self.em_input_shape])
            edges_in_p, edges_in_q, edges_in_v, edges_in_t = torch.split(edges_in, [self.em_input_shape,self.em_input_shape,
                                                                self.em_input_shape,self.em_input_shape], dim=1)
        embedding_task = []
        for i in range(self.task_num):
            embedding_mask = embedding_in[:,i,:]
            embedding_task.append((embedding_mask*edges_in_p[:,0:self.em_input_shape]).unsqueeze(1))
            embedding_task.append((embedding_mask*edges_in_q[:,0:self.em_input_shape]).unsqueeze(1))
            embedding_task.append((embedding_mask*edges_in_v[:,0:self.em_input_shape]).unsqueeze(1))
            embedding_task.append((embedding_mask*edges_in_t[:,0:self.em_input_shape]).unsqueeze(1))

        ba = state_in.shape[0]  # batch size
        bus_num = int(self.state_input_shape/4)

        q = self.w_q(torch.cat((embedding_task),dim=1))  # b*task_num*(64*4)
        self.alpha.data = self.f0(self.alpha)
        q = torch.matmul(self.alpha, q).reshape(ba, 1, -1)  # b*1*64, weighted pattern
        # q = torch.mean(q, dim=1).reshape(-1, 1, 64)  # b*1*64, mean pattern


        if self.hidden_type in ['GIN_Factor','GCN_Factor']:
            bg = dgl.batch([self.graph_u for _ in range(ba)]).to(self.device)
            state_in = state_in.reshape(-1, 4)
            # p value convolution: Active power
            k_p = self.w_k_p(bg, state_in[:,0].reshape(-1,1)).reshape(-1, bus_num, self.dim_represent)
            v_p = self.w_v_p(bg, state_in[:,0].reshape(-1,1)).reshape(-1, bus_num, self.dim_represent)
            # q value convolution: Reactive power 
            k_q = self.w_k_q(bg, state_in[:,1].reshape(-1,1)).reshape(-1, bus_num, self.dim_represent)
            v_q = self.w_v_q(bg, state_in[:,1].reshape(-1,1)).reshape(-1, bus_num, self.dim_represent)            
            # v value convolution: Voltage
            k_v = self.w_k_v(bg, state_in[:,2].reshape(-1,1)).reshape(-1, bus_num, self.dim_represent)
            v_v = self.w_v_v(bg, state_in[:,2].reshape(-1,1)).reshape(-1, bus_num, self.dim_represent)   
            # t value convolution: Phase angle
            k_t = self.w_k_t(bg, state_in[:,3].reshape(-1,1)).reshape(-1, bus_num, self.dim_represent)
            v_t = self.w_v_t(bg, state_in[:,3].reshape(-1,1)).reshape(-1, bus_num, self.dim_represent)

        else:
            k_p = self.w_k_p(state_in[:,0].reshape(-1,1)).reshape(ba, -1, self.dim_represent)
            v_p = self.w_v_p(state_in[:,0].reshape(-1,1)).reshape(ba, -1, self.dim_represent)
            # q value convolution: Reactive power 
            k_q = self.w_k_q(state_in[:,1].reshape(-1,1)).reshape(ba, -1, self.dim_represent)
            v_q = self.w_v_q(state_in[:,1].reshape(-1,1)).reshape(ba, -1, self.dim_represent)            
            # v value convolution: Voltage
            k_v = self.w_k_v(state_in[:,2].reshape(-1,1)).reshape(ba, -1, self.dim_represent)
            v_v = self.w_v_v(state_in[:,2].reshape(-1,1)).reshape(ba, -1, self.dim_represent)   
            # t value convolution: Phase angle
            k_t = self.w_k_t(state_in[:,3].reshape(-1,1)).reshape(ba, -1, self.dim_represent)
            v_t = self.w_v_t(state_in[:,3].reshape(-1,1)).reshape(ba, -1, self.dim_represent)

        temperature = k_p.shape[0] ** 0.5

        attn_p = torch.bmm(q, (k_p/temperature).transpose(1, 2))
        attn_q = torch.bmm(q, (k_q/temperature).transpose(1, 2))
        attn_v = torch.bmm(q, (k_v/temperature).transpose(1, 2))
        attn_t = torch.bmm(q, (k_t/temperature).transpose(1, 2))

        attn = torch.cat((attn_p, attn_q,attn_v,attn_t),dim=1)
        attn = torch.softmax(attn, dim=1)

        out = v_p*attn[:,0,:].unsqueeze(2)+v_q*attn[:,1,:].unsqueeze(2)+v_v*attn[:,2,:].unsqueeze(2)+v_t*attn[:,3,:].unsqueeze(2)

        out = torch.mean(out,dim=1)

        # attn = torch.bmm(q, (k/temperature).transpose(1, 2))  # mlp: b*1*1, gcn: b*1*bus_num
        # attn = self.f(attn)
        # out = torch.bmm(attn, v).view(ba, -1)  # b*64

        if self.use_dueling:  # Dueling DQN
            Q, V = self.Q(out), self.V(out)
            out = Q - Q.mean(dim=1, keepdim=True) + V
        else:
            assert False, 'Powerformer must use dueling DQN architecture.'
        if show_grid:
            return out, state, attn.detach()
        else:
            return out, state


# Powerformer-W
class SelfAttentionNetWeighted(nn.Module):
    def __init__(self, output_shape,
                 em_input_shape, state_input_shape,
                 task_num: int,
                 hidden_type='MLP', 
                 graph_u = None,
                 graph_d = None,
                 dueling_param=None, device='cuda',
                 ):

        super().__init__()

        self.em_input_shape = em_input_shape
        self.state_input_shape = state_input_shape
        self.output_shape = output_shape
        self.device = device
        self.graph_u = graph_u
        self.graph_d = graph_d
        self.use_dueling = dueling_param is not None
        self.task_num = task_num
        self.f = nn.Softmax(dim=2)
        self.f0 = nn.Softmax(dim=0)
        self.hidden_type = hidden_type
        if hidden_type in ['GIN','GCN']:
            self.is_gnn = True
        else:
            self.is_gnn = False

        if hidden_type == 'MLP':
            self.w_k = nn.Sequential(OrderedDict([
                ('k1', nn.Linear(int(state_input_shape), 128, bias=False)),
                ('kRelu', nn.ReLU()),
                ('k2', nn.Linear(128, 64, bias=False))
            ]))

            self.w_v = nn.Sequential(OrderedDict([
                ('v1', nn.Linear(int(state_input_shape), 128, bias=False)),
                ('vRelu', nn.ReLU()),
                ('v2', nn.Linear(128, 64, bias=False))
            ]))
        elif hidden_type == 'GCN':
            self.w_k = GCNlayer(in_feats=4, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v = GCNlayer(in_feats=4, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
        elif hidden_type == 'GIN':
            self.w_k = GINlayer(in_feats=4, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v = GINlayer(in_feats=4, n_hidden=64, n_classes=64, n_layers=2, device=self.device)            

        else:
            raise Exception('Invalid layer type, must be either "GCN" or "MLP"')

        self.dim_represent = 64     
        self.w_q = nn.Sequential(OrderedDict([
            ('q1', nn.Linear(em_input_shape, 128)),
            ('qRelu', nn.ReLU()),
            ('q2', nn.Linear(128, self.dim_represent)) 
        ]))

        self.alpha = nn.Parameter(torch.Tensor([1/self.task_num for _ in range(task_num)]), requires_grad=True)

        # dueling param: {'hidden_sizes': [128]}
        if self.use_dueling:  # dueling DQN
            Q_kwargs, V_kwargs = dueling_param

            Q_hidden_size = Q_kwargs['hidden_sizes'][0]
            V_hidden_size = V_kwargs['hidden_sizes'][0]
            print(Q_hidden_size)
            self.Q = nn.Sequential(OrderedDict([
                ('Q1', nn.Linear(self.dim_represent, Q_hidden_size)), 
                ('QRelu', nn.ReLU()),
                ('Q2', nn.Linear(Q_hidden_size, output_shape))
            ]))
            self.V = nn.Sequential(OrderedDict([
                ('v1', nn.Linear(self.dim_represent, V_hidden_size)),
                ('VRelu', nn.ReLU()),
                ('V2', nn.Linear(V_hidden_size, 1))
            ]))
        else:
            assert False, 'Powerformer must use dueling DQN architecture.'

    def forward(self, x, state=None, show_grid=False, info=None):  # x: b*(state_input_shape+task_num*em_input_shape)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        state_in, embedding_in, edges_in = torch.split(x, [self.state_input_shape, self.em_input_shape*self.task_num, 
                                                            self.graph_d.num_edges()*4], dim=1) # 472+task_num*172+edges_num*4
        embedding_in = torch.reshape(embedding_in, [-1, self.task_num, self.em_input_shape])   # b*task_num*line_num
        edges_in_p, edges_in_q, edges_in_v, edges_in_t = torch.split(edges_in, [self.graph_d.num_edges(),self.graph_d.num_edges(),
                                                            self.graph_d.num_edges(),self.graph_d.num_edges()], dim=1) # edges_num,edges_num,edges_num,edges_num
        
        ba = state_in.shape[0]  # batch size
        bus_num = int(self.state_input_shape/4)

        q = self.w_q(embedding_in)  # b*task_num*64
        self.alpha.data = self.f0(self.alpha)
        q = torch.matmul(self.alpha, q).reshape(ba, 1, -1)  # b*1*64, weighted pattern
        # q = torch.mean(q, dim=1).reshape(-1, 1, 64)  # b*1*64, mean pattern

        if self.hidden_type in ['GCN','GIN']:
            bg = dgl.batch([self.graph_u for _ in range(ba)]).to(self.device)
            state_in = state_in.reshape(-1, 4)
            k = self.w_k(bg, state_in).reshape(-1, bus_num, self.dim_represent)  # b*bus_num*64
            v = self.w_v(bg, state_in).reshape(-1, bus_num, self.dim_represent)  # b*bus_num*64
        else:
            k = self.w_k(state_in).reshape(ba, -1, self.dim_represent)  # b*1*64
            v = self.w_v(state_in).reshape(ba, -1, self.dim_represent)  # b*1*64

        temperature = k.shape[0] ** 0.5

        attn = torch.bmm(q, (k/temperature).transpose(1, 2))  # mlp: b*1*1, gcn: b*1*bus_num
        attn = self.f(attn)
        out = torch.bmm(attn, v).view(ba, -1)  # b*64

        if self.use_dueling:  # Dueling DQN
            Q, V = self.Q(out), self.V(out)
            out = Q - Q.mean(dim=1, keepdim=True) + V
        else:
            assert False, 'Powerformer must use dueling DQN architecture.'
        if show_grid:
            return out, state, attn.detach()
        else:
            return out, state


class SoftNet(nn.Module):
    def __init__(self, output_shape,
                 base_type, em_input_shape, input_shape,
                 em_hidden_shapes,
                 hidden_shapes,
                 num_layers, num_modules,
                 module_hidden,
                 gating_hidden, num_gating_layers,
                 # gated_hidden
                 add_bn=False,
                 pre_softmax=False,
                 dueling_param=None,
                 device='cuda',
                 cond_ob=True,
                 module_hidden_init_func=init.basic_init,
                 last_init_func=init.uniform_init,
                 activation_func=F.relu,
                 is_last=True,
                 softmax=False,
                 **kwargs):

        super().__init__()

        self.base = base_type(
                        last_activation_func = null_activation,
                        input_shape = input_shape,
                        activation_func = activation_func,
                        hidden_shapes = hidden_shapes,
                        **kwargs )
        self.em_base = base_type(
                        last_activation_func = null_activation,
                        input_shape = em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes,
                        **kwargs )

        self.em_input_shape = em_input_shape
        self.input_shape = input_shape

        self.activation_func = activation_func

        module_input_shape = self.base.output_shape
        self.layer_modules = []

        self.num_layers = num_layers
        self.num_modules = num_modules

        self.device = device
        self.use_dueling = dueling_param is not None
        self.is_last = is_last
        self.softmax = softmax

        for i in range(num_layers):
            layer_module = []
            for j in range( num_modules ):
                fc = nn.Linear(module_input_shape, module_hidden)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    )
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i,j), module)

            module_input_shape = module_hidden
            self.layer_modules.append(layer_module)

        self.last = nn.Linear(module_input_shape, output_shape)
        last_init_func( self.last )

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated"
        gating_input_shape = self.em_base.output_shape
        self.gating_fcs = []
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden)
            module_hidden_init_func(gating_fc)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape,
                                            num_modules * num_modules )
        last_init_func( self.gating_weight_fc_0)
        # self.gating_weight_fcs.append(self.gating_weight_fc_0)

        for layer_idx in range(num_layers-2):
            gating_weight_cond_fc = nn.Linear((layer_idx+1) *
                                              num_modules * num_modules,
                                              gating_input_shape)
            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)

            gating_weight_fc = nn.Linear(gating_input_shape,
                                         num_modules * num_modules)
            last_init_func(gating_weight_fc)
            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)

        self.gating_weight_cond_last = nn.Linear((num_layers-1) * \
                                                 num_modules * num_modules,
                                                 gating_input_shape)
        module_hidden_init_func(self.gating_weight_cond_last)

        self.gating_weight_last = nn.Linear(gating_input_shape, num_modules)
        last_init_func( self.gating_weight_last )

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = output_shape, 1
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": output_shape,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": output_shape,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(self, x, state=None, info=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        # print(x.shape)
        # x = torch.tensor(x).to(self.device)
        # state_in, embedding_in = torch.split(x, [self.input_shape, self.em_input_shape], dim=-1)
        state_in, embedding_in = torch.split(x, [self.input_shape, self.em_input_shape], dim=1)
        # Return weights for visualization
        # state0 = torch.tensor(state_in).to(self.device)
        out = self.base(state_in)
        # embedding_input = info['task_embedding']
        embedding = self.em_base(embedding_in)

        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)

        if len(self.gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []

        raw_weight = self.gating_weight_fc_0(self.activation_func(embedding))

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)
        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim = -1)

        module_outputs = [(layer_module(out)).unsqueeze(-2) \
                for layer_module in self.layer_modules[0]]

        module_outputs = torch.cat(module_outputs, dim = -2 )

        # [TODO] Optimize using 1 * 1 convolution.

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.layer_modules[i + 1]):
                module_input = (module_outputs * \
                    weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                        layer_module(module_input)
                ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim = -2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        if self.is_last:
            out = self.last(out)

        # bsz = out.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(out), self.V(out)
            # if self.num_atoms > 1:
            #     q = q.view(bsz, -1, self.num_atoms)
            #     v = v.view(bsz, -1, self.num_atoms)
            out = q - q.mean(dim=1, keepdim=True) + v
        # elif self.num_atoms > 1:
        #     logits = logits.view(bsz, -1, self.num_atoms)
        # if self.softmax:
        #     logits = torch.softmax(logits, dim=-1)
        if self.softmax:
            out = torch.softmax(out, dim=-1)

        # if return_weights:
        #     return out, weights, last_weight
        return out, state


class FlattenModularGatedCascadeCondNet(ModularGatedCascadeCondNet):
    def forward(self, input, embedding_input, return_weights = False):
        out = torch.cat( input, dim = -1 )
        return super().forward(out, embedding_input, return_weights = return_weights)

 
class BootstrappedNet(Net):
    def __init__(self, output_shape, 
                 head_num = 10,
                 **kwargs ):
        self.head_num = head_num
        self.origin_output_shape = output_shape
        output_shape *= self.head_num
        super().__init__(output_shape = output_shape, **kwargs)

    def forward(self, x, idx):
        base_shape = x.shape[:-1]
        out = super().forward(x)
        out_shape = base_shape + torch.Size([self.origin_output_shape, self.head_num])
        view_idx_shape = base_shape + torch.Size([1, 1])
        expand_idx_shape = base_shape + torch.Size([self.origin_output_shape, 1])
        
        out = out.reshape(out_shape)

        idx = idx.view(view_idx_shape)
        idx = idx.expand(expand_idx_shape)

        out = out.gather(-1, idx).squeeze(-1)
        return out


class FlattenBootstrappedNet(BootstrappedNet):
    def forward(self, input, idx ):
        out = torch.cat( input, dim = -1 )
        return super().forward(out, idx)


def sparse(tensor):
    for i in range(len(tensor)):
        for j in range(len(tensor[i])):
            if tensor[i][j] < 1e-03:
                tensor[i][j] = 0
    return tensor


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout=0):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.in_feats = in_feats
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        batch = features.shape[0]
        bg = dgl.batch([self.g for _ in range(batch)]).to('cuda')  # 训练时调用
        # bg = dgl.batch([self.g for _ in range(batch)])  # 评估时调用
        h = torch.reshape(features, [-1, self.in_feats])
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(bg, h)
        return h


if __name__ == '__main__':
    # m = torch.rand((2, 12))
    # print(m)
    # n = torch.reshape(m, [-1, 3, 4])
    # print(n)
    # alpha = torch.ones(3)
    # alpha[0] *= 0.5
    # alpha[2] *= 1.5
    # print(alpha)
    #
    # x = torch.matmul(alpha, n)
    # print(x)
    m = torch.rand((2, 5, 8))
    print(m)
    q = torch.mean(m, dim=1)
    print(q)

    n = torch.reshape(m, [-1, 8, 1])
    print(n)

    class MLPBase(nn.Module):
        def __init__(self, input_shape, hidden_shapes, activation_func=F.relu, init_func=init.basic_init,
                     last_activation_func=None):
            super().__init__()

            self.activation_func = activation_func
            self.fcs = []
            if last_activation_func is not None:
                self.last_activation_func = last_activation_func
            else:
                self.last_activation_func = activation_func
            input_shape = np.prod(input_shape)

            self.output_shape = input_shape
            for i, next_shape in enumerate(hidden_shapes):
                fc = nn.Linear(input_shape, next_shape)
                init_func(fc)
                self.fcs.append(fc)
                # set attr for pytorch to track parameters( device )
                self.__setattr__("fc{}".format(i), fc)

                input_shape = next_shape
                self.output_shape = next_shape

        def forward(self, x):

            out = x
            for fc in self.fcs[:-1]:
                out = fc(out)
                out = self.activation_func(out)
            out = self.fcs[-1](out)
            out = self.last_activation_func(out)
            return out

    net = MLPBase(last_activation_func=null_activation,
            input_shape=173,
            activation_func=F.relu,
            hidden_shapes=[64])
    x = torch.rand((64, 4, 173))
    y = net(x)
    print(type(y))
    print(y.shape)
