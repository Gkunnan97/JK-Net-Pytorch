# JK-Net-Pytorch
Pytorch implementation of JK-Net, paper: Representation Learning on Graphs with Jumping Knowledge Networks. Supported datasets are cora and citeseer.


Referenced Repositories:
  1) [Code of tensorflow implementation](https://github.com/ShinKyuY/Representation_Learning_on_Graphs_with_Jumping_Knowledge_Networks)
  2) [Pytorch implementation with dgl](https://github.com/mori97/JKNet-dgl)

## Requirements
    * PyTorch 1.14
    * Python 3.7

## Usage
```
python train.py --dataset dataset_name --model model_name --layer layer_name
```

    --dataset_name: core, citseer
    --model: Concat, Maxpooling, LSTM, GCN, GAT
    --layer: GCN, GAT

## Reference
    Paper: Representation Learning on Graphs with Jumping Knowledge Networks