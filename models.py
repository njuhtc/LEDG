import torch
import utils as u
from argparse import Namespace
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn
import math

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform, kaiming_uniform, zeros
from torch_geometric.utils import softmax, add_remaining_self_loops, remove_self_loops, add_self_loops
from torch_scatter import scatter_add

from typing import Optional, Tuple, Union
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_, set_diag

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge

class Sp_GCN(torch.nn.Module):
    def __init__(self,args,activation):
        super().__init__()
        self.activation = activation
        self.num_layers = args.num_layers

        self.w_list = nn.ParameterList()
        for i in range(self.num_layers):
            if i==0:
                w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
                u.reset_param(w_i)
            else:
                w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
                u.reset_param(w_i)
            self.w_list.append(w_i)


    def forward(self, A_list, Nodes_list, nodes_mask_list, vars = None):
        # print(len(Nodes_list), len(A_list))

        if vars is None:
            vars = self.w_list

        # node_feats = Nodes_list[-1]
        node_feats = Nodes_list

        #A_list: T, each element sparse tensor
        #take only last adj matrix in time
        
        Ahat = A_list
        # Ahat = A_list[-1]
        #Ahat: NxN ~ 30k
        #sparse multiplication

        # Ahat NxN
        # self.node_embs = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        last_l = self.activation(Ahat.matmul(node_feats.matmul(vars[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(vars[i])))

        return last_l
    
    def test(self, A_list, Nodes_list, nodes_mask_list, vars = None):

        node_feats = Nodes_list[-1]
        #A_list: T, each element sparse tensor
        #take only last adj matrix in time
        Ahat = A_list[-1]
        #Ahat: NxN ~ 30k
        #sparse multiplication

        # Ahat NxN
        # self.node_embs = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        return last_l

class GAT(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super(GAT, self).__init__()
        # self.activation = activation

        self.conv1 = GATConv(args.feats_per_node, args.layer_1_feats, heads=8, dropout=0.3)
        self.conv2 = GATConv(args.layer_1_feats * 8, args.layer_2_feats, heads=1, concat=False,
                             dropout=0.3)

    def forward(self, A_list, Nodes_list, nodes_mask_list, vars=None):
        x = Nodes_list
        edge_index = A_list

        # x = F.dropout(node_feats, p=0.6, training=self.training)
        if vars is None:
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index)
        else:
            x = F.elu(self.conv1(x, edge_index, vars[:4]))
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index, vars[4:])

        return x

class GIN(torch.nn.Module):
    def __init__(self, in_dim, num_classes, num_layers, hidden):
        super(GIN, self).__init__()
        
        self.conv1 = GINConv(Sequential(
            Linear(in_dim, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ),
                             train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                        train_eps=True))
        self.lin1 = Linear(hidden, hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, device, vars = None):
        x, edge_index, batch = data
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def __repr__(self):
        return self.__class__.__name__

class GATConv(MessagePassing):
    # code from pytorch_geometric https://github.com/rusty1s/pytorch_geometric
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0., bias=True, **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self._alpha = None

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.bias)

    @torch.jit._overload_method
    def forward(self, x, edge_index, return_attention_weights=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
        pass

    @torch.jit._overload_method
    def forward(self, x, edge_index, return_attention_weights=None):
        # type: (Tensor, Tensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        pass

    def forward(self, x, edge_index, vars=None, return_attention_weights=None):
        r""""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        x_prop: Tuple[torch.Tensor, torch.Tensor] = (x, x)  # Dummy.

        if vars is None:
            att_i = self.att_i
            att_j = self.att_j
            bias = self.bias
            # lin_weights = self.lin.weight
        else:
            att_i = vars[0]
            att_j = vars[1]
            bias = vars[2]
            lin_weights = vars[3]

        if isinstance(x, torch.Tensor):
            if vars is None:
                x = self.lin(x)
            else:
                x = F.linear(x, lin_weights)
            x_prop = (x, x)
        else:
            raise ValueError('x should be tensor')

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x_prop[1].size(self.node_dim))

        out = self.propagate(edge_index, x=x_prop, atti = att_i, attj = att_j)

        alpha = self._alpha
        self._alpha = None
        assert alpha is not None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if bias is not None:
            out += bias

        if isinstance(return_attention_weights, bool):
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_index_i,
                size_i: Optional[int], atti, attj):
        # Compute attention coefficients.
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * atti).sum(-1) + (x_j * attj).sum(-1)
        # alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)
        self._alpha = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class Sp_Skip_GCN(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.W_feat = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))

    def forward(self,A_list, Nodes_list = None):
        node_feats = Nodes_list[-1]
        #A_list: T, each element sparse tensor
        #take only last adj matrix in time
        Ahat = A_list[-1]
        #Ahat: NxN ~ 30k
        #sparse multiplication

        # Ahat NxN
        # self.node_feats = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        l1 = self.activation(Ahat.matmul(node_feats.matmul(self.W1)))
        l2 = self.activation(Ahat.matmul(l1.matmul(self.W2)) + (node_feats.matmul(self.W3)))

        return l2

class Sp_Skip_NodeFeats_GCN(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)

    def forward(self,A_list, Nodes_list = None):
        node_feats = Nodes_list[-1]
        Ahat = A_list[-1]
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        skip_last_l = torch.cat((last_l,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input
        return skip_last_l

class Sp_GCN_LSTM_A(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

    def forward(self,A_list, Nodes_list = None, nodes_mask_list = None):
        last_l_seq=[]
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            #A_list: T, each element sparse tensor
            #note(bwheatman, tfk): change order of matrix multiply
            last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            for i in range(1, self.num_layers):
                last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)

        last_l_seq = torch.stack(last_l_seq)

        out, _ = self.rnn(last_l_seq, None)
        return out[-1]


class Sp_GCN_GRU_A(Sp_GCN_LSTM_A):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn = nn.GRU(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

class Sp_GCN_LSTM_B(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        assert args.num_layers == 2, 'GCN-LSTM and GCN-GRU requires 2 conv layers.'
        self.rnn_l1 = nn.LSTM(
                input_size=args.layer_1_feats,
                hidden_size=args.lstm_l1_feats,
                num_layers=args.lstm_l1_layers
                )

        self.rnn_l2 = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )
        self.W2 = Parameter(torch.Tensor(args.lstm_l1_feats, args.layer_2_feats))
        u.reset_param(self.W2)

    def forward(self,A_list, Nodes_list = None, nodes_mask_list = None):
        l1_seq=[]
        l2_seq=[]
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            l1 = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            l1_seq.append(l1)

        l1_seq = torch.stack(l1_seq)

        out_l1, _ = self.rnn_l1(l1_seq, None)

        for i in range(len(A_list)):
            Ahat = A_list[i]
            out_t_l1 = out_l1[i]
            #A_list: T, each element sparse tensor
            l2 = self.activation(Ahat.matmul(out_t_l1).matmul(self.w_list[1]))
            l2_seq.append(l2)

        l2_seq = torch.stack(l2_seq)

        out, _ = self.rnn_l2(l2_seq, None)
        return out[-1]


class Sp_GCN_GRU_B(Sp_GCN_LSTM_B):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn_l1 = nn.GRU(
                input_size=args.layer_1_feats,
                hidden_size=args.lstm_l1_feats,
                num_layers=args.lstm_l1_layers
               )

        self.rnn_l2 = nn.GRU(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

class Classifier(torch.nn.Module):
    def __init__(self,args,out_features=2, in_features = None):
        super(Classifier,self).__init__()
        self.activation = torch.nn.ReLU(inplace=True)

        if in_features is not None:
            num_feats = in_features
        elif args.experiment_type in ['sp_lstm_A_trainer', 'sp_lstm_B_trainer',
                                    'sp_weighted_lstm_A', 'sp_weighted_lstm_B'] :
            num_feats = args.gcn_parameters['lstm_l2_feats'] * 2
        else:
            num_feats = args.gcn_parameters['layer_2_feats'] * 2
        print ('CLS num_feats',num_feats)

        self.vars = nn.ParameterList()

        w = Parameter(torch.ones((args.gcn_parameters['cls_feats'], num_feats)))
        nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.gcn_parameters['cls_feats'])))

        w2 = Parameter(torch.ones((out_features, args.gcn_parameters['cls_feats'])))
        nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(out_features)))

        # self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = num_feats,
        #                                                out_features =args.gcn_parameters['cls_feats']),
        #                                activation,
        #                                torch.nn.Linear(in_features = args.gcn_parameters['cls_feats'],
        #                                                out_features = out_features))

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        # print(F.linear(x, vars[0]).size())
        # print(vars[0].size(), vars[1].size(), x.size())
        x = F.linear(x, vars[0], vars[1])
        x = self.activation(x)
        x = F.linear(x, vars[2], vars[3])
        return x

class Adapter(torch.nn.Module):
    def __init__(self, args, in_features = None):
        super(Adapter, self).__init__()
        self.activation = torch.nn.ReLU(inplace=True)
        if in_features is not None:
            num_feats = in_features
        print ('ADAPT num_feats',num_feats)
        out_features = num_feats

        self.vars = nn.ParameterList()

        w = Parameter(torch.ones((num_feats, num_feats)))
        nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(num_feats)))

        w2 = Parameter(torch.ones((out_features, num_feats)))
        nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(num_feats)))
    
    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        x = self.activation(x)
        x = F.linear(x, vars[2], vars[3])
        return torch.sigmoid(x)

class TimePredicter(torch.nn.Module):
    def __init__(self, args, in_features = None):
        super(TimePredicter, self).__init__()
        self.activation = torch.nn.ReLU(inplace=True)
        if in_features is not None:
            num_feats = in_features
        print ('ADAPT num_feats',num_feats)
        out_features = 1

        self.vars = nn.ParameterList()

        w = Parameter(torch.ones((num_feats, num_feats)))
        nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(num_feats)))

        w2 = Parameter(torch.ones((out_features, num_feats)))
        nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(out_features)))
    
    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        # print(F.linear(x, vars[0]).size())
        # print(vars[0].size(), vars[1].size(), x.size())
        x = F.linear(x, vars[0], vars[1])
        x = self.activation(x)
        x = F.linear(x, vars[2], vars[3])
        return x