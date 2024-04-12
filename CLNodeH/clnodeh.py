# 峰值显存占有：20G

import copy

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from hgb import HGBDataset
from early_stop import EarlyStop
from HAN import HAN
from util import sort_training_nodes, training_scheduler, setup_seed, preprocessingHGBdata, test, add_noise

setup_seed(5)

NUM_EPOCHS = 500
PATIENCE = 50
scheduler = 'geom'

metapaths = [[('author', 'paper'), ('paper', 'author')],
             [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')],
             [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')]]

dataset = HGBDataset('./data', 'dblp')
data = dataset[0]
node_index = 0
preprocessingHGBdata(data, node_index, 800, 400, 2000)
data = add_noise(data, node_index, 0.3)

data_m = copy.deepcopy(data)
data = T.AddMetaPaths(metapaths, weighted=True, drop_orig_edge_types=True, drop_unconnected_node_types=True)(data)



num_class = data.node_stores[node_index]['y'].unique().shape[0]
target_node_type = 'author'

model_f1 = HAN(data, target_node_type, in_channels=-1, hidden_channels=64, num_heads=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model_f1 = data.to(device), model_f1.to(device)

optimizer = torch.optim.Adam(model_f1.parameters(), lr=0.01, weight_decay=5e-4)
early_stop_f1 = EarlyStop(PATIENCE, './checkpoints/best_f1.pth')

for epoch in range(NUM_EPOCHS):
    train_ids = data[target_node_type].train_mask
    model_f1.train()
    optimizer.zero_grad()
    _, out = model_f1(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(out[train_ids], data[target_node_type].y[train_ids])
    loss.backward()
    optimizer.step()

    _, val_acc, _ = test(model_f1, data, target_node_type, 'micro-f1')
    # early stop
    if not early_stop_f1.step(val_acc, model_f1):
        break

model_f1 = torch.load('./checkpoints/best_f1.pth')
model_f1.eval()
train_acc, val_acc, test_acc = test(model_f1, data, target_node_type, 'micro-f1')
print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

embedding, out = model_f1(data.x_dict, data.edge_index_dict)
label = out.argmax(dim=-1)

# -------------------测算difficulty----------------------
semantics = [[('author', 'paper'), ('paper', 'author')],
             [('author', 'paper'), ('paper', 'conference'), ('conference', 'paper'), ('paper', 'author')],
             [('author', 'paper'), ('paper', 'term'), ('term', 'paper'), ('paper', 'author')]]
data_m = T.AddMetaPaths(semantics, weighted=True, drop_orig_edge_types=True, drop_unconnected_node_types=True)(data_m)
semantic_attn = [0.5, 0.3, 0.2]

sorted_trainset = sort_training_nodes(data_m, 0, label, semantic_attn, embedding, 0.5)

# ------------------- clnode中的f2----------------------------
# 网格搜索最优的lambda和T
best_lambda = 0
best_T = 0
best_val_acc = 0
for lam in [0.25, 0.5, 0.75]:
    for T in [50, 100, 200]:
        model_gs = HAN(data, target_node_type, in_channels=-1, hidden_channels=64, num_heads=2)
        model_gs = model_gs.to(device)
        optimizer_gs = torch.optim.Adam(model_gs.parameters(), lr=0.01, weight_decay=5e-4)
        early_stop_gs = EarlyStop(PATIENCE, './checkpoints/best_model-' + str(lam) + '-' + str(T) + '.pth')
        for epoch in range(NUM_EPOCHS):
            size = training_scheduler(lam, epoch, T, scheduler)
            batch_id = sorted_trainset[:int(size * sorted_trainset.shape[0])]
            optimizer_gs.zero_grad()
            model_gs.train()
            _, out = model_gs(data.x_dict, data.edge_index_dict)
            loss = F.cross_entropy(out[batch_id], data[target_node_type].y[batch_id])
            loss.backward()
            optimizer_gs.step()

            # 在验证集上计算准确率
            model_gs.eval()
            pred = model_gs(data.x_dict, data.edge_index_dict)[1].argmax(dim=-1)
            mask = data[target_node_type]['val_mask']
            acc = (pred[mask] == data[target_node_type].y[mask]).sum() / mask.sum()

            # early stop
            if not early_stop_gs.step(acc, model_gs):
                break

        model_gs = torch.load('./checkpoints/best_model-' + str(lam) + '-' + str(T) + '.pth')
        model_gs.eval()
        pred = model_gs(data.x_dict, data.edge_index_dict)[1].argmax(dim=-1)
        mask = data[target_node_type]['val_mask']
        val_acc = (pred[mask] == data[target_node_type].y[mask]).sum() / mask.sum()
        print('the lambda is {:.2f}, the T is {}, the val_acc is {:.4f}'.format(lam, T, val_acc), end='\n')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lambda = lam
            best_T = T

model_cl = torch.load('./checkpoints/best_model-' + str(best_lambda) + '-' + str(best_T) + '.pth')
model_cl.eval()
train_acc, val_acc, test_acc = test(model_cl, data, target_node_type, 'micro-f1')
print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
