import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch import tensor
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import pdb
import random
import numpy as np
from torch.autograd import grad
from torch_geometric.data import Batch
from utils import k_fold, num_graphs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def process_dataset(dataset):
    
    num_nodes = max_num_nodes = 0
    for data in dataset:
        num_nodes += data.num_nodes
        max_num_nodes = max(data.num_nodes, max_num_nodes)
    num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)
    transform = T.ToDense(num_nodes)
    new_dataset = []
    
    for data in tqdm(dataset):
        data = transform(data)
        add_zeros = num_nodes - data.feat.shape[0]
        if add_zeros:
            dim = data.feat.shape[1]
            data.feat = torch.cat((data.feat, torch.zeros(add_zeros, dim)), dim=0)
        new_dataset.append(data)
    return new_dataset

def train_baseline_syn(train_set, val_set, test_set, model_func=None, args=None):
    
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree

    best_accs_folds = []
    for fold in range(args.folds):
        model = model_func(args.feature_dim, args.num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
    
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1, verbose=False)
        best_val_acc, update_test_acc, update_train_acc, update_epoch = 0, 0, 0, 0
        stopupdate = 0
        
        for epoch in range(1, args.epochs + 1):
            
            train_loss, train_acc = train(model, optimizer, train_loader, device, args)
            val_acc = eval_acc(model, val_loader, device, args)
            test_acc = eval_acc(model, test_loader, device, args)
            lr_scheduler.step()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                update_test_acc = test_acc
                update_train_acc = train_acc
                update_epoch = epoch
                stopupdate = 0
            else:
                stopupdate+=1
                if  stopupdate > 30:
                    print('early stop |', end = '')
                    break
            print("fold:[{}] | BIAS:[{:.2f}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.2f}] val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Update Test:[{:.2f}] at Epoch:[{}] | lr:{:.6f}"
                    .format(fold,
                            args.bias,
                            args.model,
                            epoch, 
                            args.epochs,
                            train_loss, 
                            train_acc * 100, 
                            val_acc * 100,
                            test_acc * 100, 
                            best_val_acc * 100,
                            update_test_acc * 100, 
                            update_epoch,
                            optimizer.param_groups[0]['lr']))
    
        print("fold:[{}] | syd: BIAS:[{:.2f}] | Best Val acc:[{:.2f}] Test acc:[{:.2f}] at epoch:[{}]"
            .format(fold,
                    args.bias,
                    best_val_acc * 100, 
                    update_test_acc * 100,
                    update_epoch))
        
        best_accs_folds.append(update_test_acc)
        

    best_accs_folds = np.array(best_accs_folds)
    mean = best_accs_folds.mean()*100
    std = best_accs_folds.std()*100
    print('test:[{:.2f} ± {:.2f}]'.format(mean,std))
    
    savefile = 'saved/' + args.dataset + args.model + args.train_model
    if args.double_co_use_mgda:
        savefile = savefile + '2mgda'
    if args.mgda_with_double_co:
        savefile = savefile + '2co'
    torch.save(model, savefile + '.pt')

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        x = data.x if data.x is not None else data.feat
        return x.size(0)
        
def train(model, optimizer, loader, device, args):
    
    model.train()
    total_loss = 0
    correct = 0
    
    for it, data in enumerate(loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_acc(model, loader, device, args):
    
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

def train_baseline_real(dataset=None, model_func=None, args=None):
    train_accs, test_accs = [], []
    random_guess = 1.0 / dataset.num_classes
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.epoch_select))):

        best_test_acc, best_epoch = 0, 0
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx.long()]
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        model = model_func(dataset.num_features, dataset.num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):

            train_loss, train_acc = train(model, optimizer, train_loader, device, args)
            test_acc = eval_acc(model, test_loader, device, args)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
            
            print("Causal | dataset:[{}] fold:[{}] | Epoch:[{}/{}] Loss:[{:.4f}] Train:[{:.4f}] Test:[{:.2f}]  (RG:{:.2f}) | Best Test:[{:.2f}] at Epoch:[{}] "
                    .format(args.dataset,
                            fold,
                            epoch, args.epochs,
                            train_loss, 
                            train_acc * 100,  
                            test_acc * 100, 
                            random_guess*  100,
                            best_test_acc * 100, 
                            best_epoch,
                            ))

        print("syd: Causal fold:[{}] | Dataset:[{}] Model:[{}] | Best Test:[{:.2f}] at epoch [{}] (RG:{:.2f})"
                .format(fold,
                        args.dataset,
                        args.model,
                        best_test_acc * 100, 
                        best_epoch,
                        random_guess*  100))
    
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(args.folds, args.epochs)
    test_acc = test_acc.view(args.folds, args.epochs)
    
    _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
    selected_epoch = selected_epoch.repeat(args.folds)
    

    test_acc = test_acc[torch.arange(args.folds, dtype=torch.long), selected_epoch]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()

    print("=" * 150)
    print('sydall Final: Causal | Dataset:[{}] Model:[{}] seed:[{}]| Test Acc: {:.2f}±{:.2f} (RG:{:.2f}) | [Settings] co:{},c:{},o:{},harf:{},dim:{},fc:{}'
         .format(args.dataset,
                 args.model,
                 args.seed,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 random_guess*  100,
                 args.co,
                 args.c,
                 args.o,
                 args.harf_hidden,
                 args.hidden,
                 args.fc_num))
    print("=" * 150)
