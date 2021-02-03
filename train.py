import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import GNN, JKNetConcat, JKNetLSTM, JKNetMaxpool
from utils import _load_data, accuracy


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='dataset name.')
    # model can be "Fast" or "AS"
    parser.add_argument('--model', type=str, default='Concat',
                        help='model name.')
    parser.add_argument('--layer', type=str, default='GCN',
                        help='model name.')
    parser.add_argument('--test_gap', type=int, default=1,
                        help='the train epochs between two test')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def train(adj, features, labels, train_ind, train_times):
    t = time.time()
    model.train()
    train_labels = labels[train_ind]
    for epoch in range(train_times):
        AvgLoss, AvgAcc, Nbatch = 0, 0, 0
        # for batch_inds, batch_labels in get_batches(train_ind,
        #                                             train_labels,
        #                                             batch_size):

        output = model(adj, features)
        output = output[train_ind]
        loss_train = loss_fn(output, train_labels)
        acc_train = accuracy(output, train_labels)
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()

        AvgLoss += loss_train
        AvgAcc += acc_train
        Nbatch += 1

        AvgLoss /= Nbatch
        AvgAcc /= Nbatch
    # only return the result of the last epoch
    return AvgLoss.item(), AvgAcc.item(), time.time() - t


def test(test_adj, test_feats, test_labels, test_ind):
    t = time.time()
    model.eval()
    outputs = model(test_adj, test_feats)
    outputs, test_labels = outputs[test_ind], test_labels[test_ind]
    loss_test = loss_fn(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)

    return loss_test.item(), acc_test.item(), time.time() - t


if __name__ == '__main__':
    # load data, set superpara and constant
    args = get_args()
    test_gap = args.test_gap

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    (adj, features, labels, train_mask, val_mask,
     test_mask) = _load_data(args.dataset)  # 'pubmed'
    adj = adj.to_dense()
    Nfeats = features.shape[1]
    Nclass = labels.shape[1]

    # set device
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.max(1)[1].to(device)

    # init model, optimizer and loss function
    LayerType, ModelType = args.layer, args.model
    Hidden = 16
    DimList = [Nfeats, Hidden, Hidden]
    if LayerType == "GCN":
        LayerList = GNN.GetLayers(DimList, "GCN")
    elif LayerType == "GAT":
        alpha = 0.2
        DimList = [Nfeats, Hidden, Hidden, Hidden]
        LayerList = GNN.GetLayers(DimList, "GAT",
                                  dropout=args.dropout,
                                  alpha=alpha)
    elif LayerType == "GraphSage":
        pass
    else:
        raise ValueError(f"ModelType has wrong value: {ModelType}")

    if ModelType == "Concat":
        model = JKNetConcat(LayerList, args.dropout,
                            (sum(DimList[1:]), Nclass))
    elif ModelType == "MaxPooling":
        model = JKNetMaxpool(LayerList, args.dropout, (Hidden, Nclass))
    elif ModelType == "LSTM":
        LstmArgs = {'input_size': Hidden, 'hidden_size': 3,
                    'num_layers': 1, 'bidirectional': True}
        model = JKNetLSTM(LayerList, args.dropout,
                          (Hidden, Nclass), **LstmArgs)
    elif ModelType == 'GAT' or ModelType == 'GCN':
        model = GNN(LayerList, args.dropout)
    else:
        raise ValueError(f"ModelType has wrong value: {ModelType}")

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = F.nll_loss

    # train and test
    for epochs in range(0, args.epochs // test_gap):
        train_loss, train_acc, train_time = train(adj,
                                                  features,
                                                  labels,
                                                  train_mask,
                                                  test_gap)
        test_loss, test_acc, test_time = test(adj,
                                              features,
                                              labels,
                                              test_mask)
        print(f"epchs: {epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              f"train_acc: {train_acc:.3f}, "
              f"train_times: {train_time:.3f}s "
              f"test_loss: {test_loss:.3f}, "
              f"test_acc: {test_acc:.3f}, "
              f"test_times: {test_time:.3f}s")
