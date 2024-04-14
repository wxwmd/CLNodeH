import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class MLP(torch.nn.Module):
    def __init__(self, input_num, output_num):
        super(MLP, self).__init__()
        self.layer = nn.Linear(input_num, output_num)

    def forward(self, x):
        x = self.layer(x)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)


class GCNNet(torch.nn.Module):
    def __init__(self, dataset, hidden_num, out_num):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_num)
        self.conv2 = GCNConv(hidden_num, out_num)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)


class GCNClassifier(torch.nn.Module):
    def __init__(self, dataset, hidden_num, embedding_num):
        super().__init__()
        self.gnn = GCNNet(dataset, hidden_num, embedding_num)
        self.mlp = MLP(embedding_num, dataset.num_classes)

    def forward(self, data):
        embedding = self.gnn(data)
        out = self.mlp(embedding)
        return embedding, out
