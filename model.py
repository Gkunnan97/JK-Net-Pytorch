import torch
import torch.nn as nn
from layer import GraphAttentionLayer, GraphConvolution, GraphConvLayer


class GNN(nn.Module):
    def __init__(self, LayerList, dropout):
        super().__init__()
        for i, Layer in enumerate(LayerList):
            self.add_module('Layer_{}'.format(i), Layer)

        self.LayerList = LayerList
        self.dropout = dropout

    def _GetHiddenEmbedding(self, adj, x):
        HiddenLayerOutput = []
        for layer in self.LayerList:
            x = torch.dropout(layer(adj, x), self.dropout,
                              train=self.training)
            HiddenLayerOutput.append(x)
        return HiddenLayerOutput

    @classmethod
    def GetLayers(cls, DimList, Type, **kwargs):
        """
        Args:
            Type(:str)
        """
        LayerFun = {'GCN': GraphConvolution,
                    'GAT': GraphAttentionLayer,
                    'GraphSage': GraphConvLayer}[Type]
        res = []
        for i in range(len(DimList)-1):
            res.append(LayerFun(DimList[i], DimList[i+1], **kwargs))
        return res

    def forward(self, adj, x):
        for layer in self.LayerList:
            x = torch.dropout(layer(adj, x), self.dropout,
                              train=self.training)
        return torch.log_softmax(x, 1)


class JKNetConcat(GNN):
    def __init__(self, LayerList, dropout, LinerDim):
        super().__init__(LayerList, dropout)
        self.FinalLinear = nn.Linear(*LinerDim)

    def forward(self, adj, x):
        HiddenEmbeddings = self._GetHiddenEmbedding(adj, x)
        Embedding = torch.cat(HiddenEmbeddings, dim=1)
        Embedding = self.FinalLinear(Embedding)
        return torch.log_softmax(Embedding, 1)


class JKNetMaxpool(GNN):
    def __init__(self, LayerList, dropout, LinearDim):
        super().__init__(LayerList, dropout)
        self.Linear = nn.Linear(*LinearDim)

    def forward(self, adj, x):
        HiddenEmbeddings = self._GetHiddenEmbedding(adj, x)
        Embedding = torch.stack(HiddenEmbeddings, dim=0)
        # NOTE: mind the max
        Embedding = torch.max(Embedding, dim=0)[0]
        Embedding = self.Linear(Embedding)
        return torch.log_softmax(Embedding, 1)


class JKNetLSTM(GNN):
    def __init__(self, LayerList, dropout, LinearDim, **LstmParas):
        super().__init__(LayerList, dropout)
        self.Lstm = nn.LSTM(**LstmParas)
        LstmOutDim = LstmParas.get("hidden_size")
        self.AttMap = nn.Linear(2 * LstmOutDim, 1)
        self.Linear = nn.Linear(*LinearDim)

    def forward(self, adj, x):
        HiddenEmbeddings = self._GetHiddenEmbedding(adj, x)
        # pdb.set_trace()
        Embedding = torch.stack(HiddenEmbeddings, dim=0)
        LstmRes = self.Lstm(Embedding)[0]
        AttentionWeight = self.AttMap(LstmRes)
        AttentionWeight = torch.softmax(AttentionWeight, 0)
        AttentionWeight = AttentionWeight.repeat(1, 1, Embedding.shape[-1])
        Embedding = torch.sum(Embedding * AttentionWeight, 0)
        Embedding = self.Linear(Embedding)
        return torch.log_softmax(Embedding, 1)
