import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Linear = nn.Linear(in_features, out_features)
        self.bias = bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, out_features))
            # NOTE: error with the initialization
            nn.init.xavier_uniform_(self.bias.data, gain=1.414)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        support = self.Linear(input)  # torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if hasattr(self, 'bias'):
            return output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


AGGREGATIONS = {
    'sum': torch.sum,
    'mean': torch.mean,
    'max': torch.max,
}


class GraphConvLayer(nn.Module):
    """Graph convolution layer.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """

    def __init__(self, in_features, out_features,
                 aggregation='sum',
                 bias=False):
        super(GraphConvLayer, self).__init__()

        if aggregation not in AGGREGATIONS.keys():
            raise ValueError("'aggregation' argument has to be one of "
                             "'sum', 'mean' or 'max'.")
        self.aggregate = lambda nodes: AGGREGATIONS[aggregation](nodes, dim=1)

        self.linear = nn.Linear(in_features, out_features)
        self.self_loop_w = nn.Linear(in_features, out_features)
        if self.bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            nn.init.xavier_uniform_(self.bias.data, gain=1.414)

    def forward(self, adj, x):
        support = self.linear(x)
        x = torch.spmm(adj, support)
        if hasattr(self, 'bias'):
            return x + self.bias
        return x


class _GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, adj, h):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.mm(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) \
            + ' -> ' + str(self.out_features) + ') '\
            + f'dropout: {self.dropout} '\
            + f'alpha: {self.alpha} '\
            + f'concat: {self.concat} '


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True,
                 HeadNum=1):
        super().__init__()
        self.Heads = [_GraphAttentionLayer(in_features, out_features,
                                           dropout, alpha, concat)
                      for _ in range(HeadNum)]
        for i, Head in enumerate(self.Heads):
            self.add_module('Head_{}'.format(i), Head)

    def forward(self, adj, h):
        HeadRes = [Head(adj, h) for Head in self.Heads]
        h = torch.cat(HeadRes, dim=1)
        return h
