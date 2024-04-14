import copy
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.datasets import HGBDataset
from torch_geometric.data.hetero_data import HeteroData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# neighborhood-based difficulty measurer
def weighted_neighborhood_difficulty_measurer(data, node_index, edge_index, label):
    # 加上自环，将节点本身的标签也计算在内
    neighbor_label = data.edge_stores[edge_index]['edge_index']
    # 得到每个节点的邻居标签
    neighbor_label[1] = label[neighbor_label[1]]

    edge_weight = torch.unsqueeze(data.edge_stores[edge_index]['edge_weight'], 0)

    weighted_neighbor_label = torch.cat((neighbor_label, edge_weight), dim=0)

    weighted_neighbor_label = torch.transpose(weighted_neighbor_label, 0, 1)
    # 1. 将 (col1, col2) 转换为字符串，以便用作字典的键
    key_values = [f'{col1.item()},{col2.item()}' for col1, col2, _ in weighted_neighbor_label]
    # 2. 构建一个字典，用于累加相同 (col1, col2) 的 col3 值
    sum_dict = {}
    for i, key in enumerate(key_values):
        if key in sum_dict:
            # 如果已经存在，累加该 (col1, col2) 对应的 col3
            sum_dict[key] += weighted_neighbor_label[i, 2].item()
        else:
            # 否则，记录该 (col1, col2) 对应的 col3
            sum_dict[key] = weighted_neighbor_label[i, 2].item()
    # 3. 根据字典构建新的 tensor
    weighted_class_distribution = torch.zeros(len(sum_dict), 3)
    for i, (key, sum_value) in enumerate(sum_dict.items()):
        col1, col2 = map(float, key.split(','))
        weighted_class_distribution[i, 0] = col1
        weighted_class_distribution[i, 1] = col2
        weighted_class_distribution[i, 2] = sum_value

    index = weighted_class_distribution[:, :2]
    weights = weighted_class_distribution[:, 2]

    neighbor_class = torch.sparse_coo_tensor(index.T, weights)
    neighbor_class = neighbor_class.to_dense().float().to(device)
    # 开始计算节点的邻居信息熵
    train_ids = data.node_stores[node_index]['train_mask'].nonzero().squeeze(dim=1)
    data.node_stores[node_index]['train_id'] = train_ids
    neighbor_class = neighbor_class[train_ids]
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)
    return local_difficulty.to(device)


# feature-based difficulty measurer
def feature_difficulty_measurer(data, node_index, label, embedding):
    train_ids = data.node_stores[node_index]['train_mask'].nonzero().squeeze(dim=1)
    normalized_embedding = F.normalize(torch.exp(embedding))
    classes = label.unique()
    class_features = {}
    for c in classes:
        class_nodes = torch.nonzero(label == c).squeeze(1)
        node_features = normalized_embedding.index_select(0, class_nodes)
        class_feature = node_features.sum(dim=0)
        # 这里注意归一化
        class_feature = class_feature / torch.sqrt((class_feature * class_feature).sum())
        class_features[c.item()] = class_feature

    similarity = {}
    for u in train_ids:
        # 做了实验，认为让节点乘错误的类别feature，看看效果
        feature = normalized_embedding[u]
        class_feature = class_features[label[u].item()]
        sim = torch.dot(feature, class_feature)
        sum = torch.tensor(0.).to(device)
        for cf in class_features.values():
            sum += torch.dot(feature, cf)
        sim = sim * len(classes) / sum
        similarity[u.item()] = sim

    class_avg = {}
    for c in classes:
        count = 0.
        sum = 0.
        for u in train_ids:
            if label[u] == c:
                count += 1
                sum += similarity[u.item()]
        if count != 0:
            class_avg[c.item()] = sum / count

    global_difficulty = []

    for u in train_ids:
        sim = similarity[u.item()] / class_avg[label[u].item()]
        # print(u,sim)
        sim = torch.tensor(1) if sim > 0.95 else sim
        node_difficulty = 1 / sim
        global_difficulty.append(node_difficulty)

    return torch.tensor(global_difficulty).to(device)


# multi-perspective difficulty measurer
def difficulty_measurer(data, node_index, label, semantic_attn, embedding, alpha):
    global_difficulty = feature_difficulty_measurer(data, node_index, label, embedding)

    local_difficulty = torch.zeros_like(global_difficulty)
    for i in range(len(data.edge_types)):
        attn = semantic_attn[i]
        local_difficulty += weighted_neighborhood_difficulty_measurer(data, node_index, i,  label) * attn

    node_difficulty = alpha * local_difficulty + (1 - alpha) * global_difficulty
    return node_difficulty


# sort training nodes by difficulty
def sort_training_nodes(data, node_index: int, label, semantic_attn, embedding, alpha=0.5):
    '''
    :param data: 异构图数据
    :param node_index: 节点类型下标
    :param edge_indexes: 边类型下标，支持多个metapath
    :param label: f1预测得到的节点伪标签
    :param semantic_attn: 每个元路径的权重
    :param embedding: f1预测得到的节点嵌入
    :param alpha: feature-based difficulty measurer的权重
    :return: 一个排序好的训练节点数组，按照节点的难度升序排序
    '''
    node_difficulty = difficulty_measurer(data, node_index, label, semantic_attn, embedding, alpha)
    _, indices = torch.sort(node_difficulty)
    train_ids = data.node_stores[node_index]['train_id'].to(device)
    sorted_trainset = train_ids[indices]
    return sorted_trainset


def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))


def random_split(data, node_index, training_nodes_num):
    node_data = data.node_stores[node_index]
    num_classes = torch.unique(node_data['y']).shape[0]

    node_id = torch.nonzero(node_data['train_mask']).squeeze().numpy()
    np.random.shuffle(node_id)

    # pick at least one node for each class
    class_to_node = {}
    for i in range(len(node_id) - num_classes):
        node = node_id[i]
        node_class = node_data['y'][node].item()
        if node_class not in class_to_node:
            class_to_node[node_class] = node
            node_id = np.delete(node_id, i)

    for i in range(num_classes):
        node = class_to_node[i]
        node_id = np.insert(node_id, 0, node)

    train_ids = torch.tensor(node_id[:training_nodes_num], dtype=torch.long)

    node_data['train_mask'] = torch.full(node_data['y'].shape, False)
    node_data['train_mask'][train_ids] = True
    node_data['train_id'] = train_ids
    return node_data

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def preprocessingHGBdata(data: HGBDataset, node_index, num_train, num_val, num_test, processLabel = False):
    '''
    对HGB数据集进行预处理
    :param data: hgb数据集，有dblp，imdb，freebase，acm四类
    :param node_index: 进行node classification的节点类型下标
    :return:
    '''
    train_mask = data.node_stores[node_index]['y']
    n = train_mask.shape[0]
    node_ids = torch.randperm(n)
    train_mask = torch.BoolTensor([False] * n)
    val_mask = torch.BoolTensor([False] * n)
    test_mask = torch.BoolTensor([False] * n)
    train_mask[node_ids[:num_train]] = True
    val_mask[node_ids[num_train: num_train+num_val]] = True
    test_mask[node_ids[num_train+num_val: num_train+num_val+num_test]] = True
    data.node_stores[node_index]['train_mask'] = train_mask
    data.node_stores[node_index]['val_mask'] = val_mask
    data.node_stores[node_index]['test_mask'] = test_mask

    if processLabel:
        y = data.node_stores[node_index]['y']
        y = torch.argmax(y, dim=1)
        data.node_stores[node_index]['y'] = y


@torch.no_grad()
def test(model_input, data, node_type, evaluation='acc'):
    model_input.eval()
    _, out = model_input(data.x_dict, data.edge_index_dict)
    pred = out.argmax(dim=-1)

    results = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[node_type][split]
        y = data[node_type].y[mask]
        y_pred = pred[mask]

        result = 0
        if evaluation == 'acc':
            result = (y_pred == y).sum() / mask.sum()
        elif evaluation == 'macro-f1':
            result = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        elif evaluation == 'micro-f1':
            result = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='micro')
        results.append(float(result))
    return results

@torch.no_grad()
def test_metapath2vec(model_input, data, node_type, evaluation='acc'):
    model_input.eval()

    pred = model_input(node_type)

    results = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[node_type][split]
        y = data[node_type].y[mask]
        y_pred = pred[mask]

        result = 0
        if evaluation == 'acc':
            result = (y_pred == y).sum() / mask.sum()
        elif evaluation == 'macro-f1':
            result = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
        elif evaluation == 'micro-f1':
            result = f1_score(y.cpu().numpy(), y_pred.cpu().numpy(), average='micro')
        results.append(float(result))
    return results


def get_wrong_label(true_label, classes, attack='uniform'):
    if attack == 'uniform':
        labels = np.arange(classes)
        labels = np.delete(labels, true_label)
        return random.choice(labels)
    else:
        return (true_label + classes - 1) % classes


def add_noise(data, node_index, percent, attack='uniform'):
    """
    给异质图数据加上噪声
    :param data: HerotoData
    :param node_index:
    :param percent:
    :param attack:
    :return: HerotoData
    """
    data_dict = data.to_dict()
    y = copy.deepcopy(data.node_stores[node_index]['y'])
    train_mask = data.node_stores[node_index]['train_mask']
    val_mask = data.node_stores[node_index]['val_mask']
    num_class = torch.unique(y).shape[0]

    wrong_train_num = (int)(train_mask.sum() * percent)
    train_ids = train_mask.nonzero().view(-1)
    wrong_train_ids = train_ids[:wrong_train_num]
    for wrong_id in wrong_train_ids:
        y[wrong_id] = get_wrong_label(y[wrong_id].cpu(), num_class, attack)

    val_ids = torch.nonzero(val_mask).squeeze(dim=1).cpu()  # 要用numpy shuffle
    np.random.shuffle(val_ids.numpy())
    val_wrong_ids = val_ids[:int(percent * val_ids.shape[0])].to(device)
    for wrong_id in val_wrong_ids:
        y[wrong_id] = get_wrong_label(y[wrong_id].cpu(), num_class)

    target_node_type = data.node_types[node_index]
    data_dict[target_node_type]['y'] = y
    return HeteroData.from_dict(data_dict)
