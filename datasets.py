import os.path as osp
import re
import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from feature_expansion import FeatureExpander
from tu_dataset import TUDatasetExt
import pdb

def get_dataset(name, sparse=True, feat_str="deg+ak3+reall", root=None, pruning_percent=0):
    if root is None or root == '':
        path = osp.join(osp.expanduser('~'), 'pyG_data', name)
    else:
        path = osp.join(root, name)
    degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall("an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    groupd = re.findall("groupd(\d+)", feat_str)
    groupd = int(groupd[0]) if groupd else 0
    remove_edges = re.findall("re(\w+)", feat_str)
    remove_edges = remove_edges[0] if remove_edges else 'none'
    edge_noises_add = re.findall("randa([\d\.]+)", feat_str)
    edge_noises_add = float(edge_noises_add[0]) if edge_noises_add else 0
    edge_noises_delete = re.findall("randd([\d\.]+)", feat_str)
    edge_noises_delete = float(
        edge_noises_delete[0]) if edge_noises_delete else 0
    centrality = feat_str.find("cent") >= 0
    coord = feat_str.find("coord") >= 0
    
    pre_transform = FeatureExpander(
        degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
        centrality=centrality, remove_edges=remove_edges,
        edge_noises_add=edge_noises_add, edge_noises_delete=edge_noises_delete,
        group_degree=groupd).transform

    dataset = TUDatasetExt(
        path, 
        name, 
        pre_transform=pre_transform,
        use_node_attr=True, 
        processed_filename="data_%s.pt" % feat_str, 
        pruning_percent=pruning_percent)

    dataset.data.edge_attr = None
    return dataset

def dataset_split(dataset):
    
    train_set = dataset['house'][:800] + dataset['cycle'][:800]
    val_set = dataset['house'][800:900] + dataset['cycle'][800:900]
    test_set = dataset['house'][900:] + dataset['cycle'][900:]
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    return train_set, val_set, test_set

def get_mnistsp(args):
    from dir_datasets import MNIST75sp
    from torch_geometric.data import DataLoader
    n_train_data, n_val_data = 20000, 5000
    train_val = MNIST75sp('data/MNISTSP/', mode='train')
    perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(0))
    train_val = train_val[perm_idx]
    train_dataset, val_dataset = train_val[:n_train_data], train_val[-n_val_data:]
    test_dataset = MNIST75sp('data/MNISTSP/', mode='test')
    color_noises = torch.load('data/MNISTSP/raw/mnist_75sp_color_noise.pt').view(-1,3)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    noises = []
    noise_level = 0.4
    for graph in test_loader: 
        n_samples = 0
        noises.append(color_noises[n_samples:n_samples + graph.x.size(0), :] * noise_level)
        n_samples += graph.x.size(0)
    test_dataset.data.x[:, :3] = test_dataset.data.x[:, :3] + torch.cat(noises)
    args.feature_dim = 5
    args.num_classes = 10
    return train_dataset, val_dataset,test_dataset



def get_molhiv(args):
    from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
    dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv', root='data') 
    dataset.data.x = dataset.data.x.float()
    split_idx = dataset.get_idx_split() 
    args.feature_dim = 9
    args.num_classes = 2
    return dataset[split_idx["train"]],dataset[split_idx["valid"]],dataset[split_idx["test"]]

def get_SPMotif(args):
    from dir_datasets import SPMotif
    traing_dataset = SPMotif('data/'+ f'SPMotif-{args.bias}/', mode='train')
    val_dataset = SPMotif('data/'+ f'SPMotif-{args.bias}/', mode='val')
    test_dataset = SPMotif('data/'+ f'SPMotif-{args.bias}/', mode='test')
    args.dataset = f'SPMotif-{args.bias}'
    args.feature_dim = 4
    args.num_classes = 3
    return traing_dataset,val_dataset,test_dataset

def get_split_dataset(dataset, degree_bias=True, data_split_ratio=[0.8, 0.1, 0.1], seed=2):
    if degree_bias:
        train, test = [], []
        for g in dataset:
            if g.num_edges <= 2: continue
            degree = float(g.num_edges) / g.num_nodes
            if degree >= 1.76785714:
                train.append(g)
            elif degree <= 1.57142857:
                test.append(g)
        
        eval = train[:int(len(train) * 0.1)]
        train = train[int(len(train) * 0.1):]
        print(len(train), len(eval), len(test))
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test])
    return train, eval, test

def get_sst(args):
    from dir_datasets.graphsst2_dataset import get_dataset, get_dataloader 
    dataset = get_dataset(dataset_dir='data/', dataset_name='Graph_SST2', task=None)
    args.feature_dim = 768
    args.num_classes = 2
    return get_split_dataset(dataset)


def get_dir_dataset(args):
    if args.dataset == 'mnistsp':
        return get_mnistsp(args)
    elif args.dataset == 'molhiv':
        return get_molhiv(args)
    elif args.dataset == 'sst':
        return get_sst(args)
    elif args.dataset == 'Spurious-Motif':
        return get_SPMotif(args)
    


