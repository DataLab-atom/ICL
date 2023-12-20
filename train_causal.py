import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import tensor
import numpy as np
from min_norm_solvers import MinNormSolver
from utils import k_fold
import time

from train_epoch import funcs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_causal_syn(train_set, val_set,  test_set, model_func=None, args=None):
    train_causal_epoch = funcs[args.train_model]

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
    temp_att = args.attion_loss
    best_accs_folds = []
    mgda_s = [] 
    for fold in range(args.folds):
        args.attion_loss = temp_att
        model = model_func(args.feature_dim, args.num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1, verbose=False)
        times = []
        lastc = []
        names = ['Train','val','Test']
        best_accs = [0. for _ in names]
        best_epoch = [0 for _ in names]
        stop_updates = [0 for _ in names]
        epoch_mgdas = []
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            mytime,lastc,train_loss, loss_c, loss_o, loss_co, train_acc_co, mgda = train_causal_epoch(model, optimizer, train_loader, device,lastc,args)
            lr_scheduler.step()
            times.append(mytime)
            epoch_mgdas.append(mgda)
            val_acc_co, _, _ = eval_acc_causal(model, val_loader, device, args)
            test_acc_co, _, _ = eval_acc_causal(model, test_loader, device, args)
            accs = [train_acc_co,val_acc_co,test_acc_co]

            if best_accs[2] < accs[2]:
                best_accs = accs
                best_epoch[2] = epoch
                stop_updates[2] = 0
            else:
                stop_updates[2] += 1 
    
            info_out_print = "flod:[{}]| dataset:[{}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f} = {:.4f}+{:.4f}+{:.4f}] \n".format(
                                fold,args.dataset,args.model,epoch, args.epochs,train_loss,loss_c,loss_o,loss_co)
            
            acc_out = "\tACC:"
            best_acc_out = "\n\tBESTACC:"
            for i in  range(len(names)):
                acc_out = acc_out + names[i] + ": [{:.2f}]".format(accs[i]*100)
                best_acc_out = best_acc_out + names[i] + ": [{:.2f}] Update: [{}]".format(best_accs[i]*100,best_epoch[i])
    
            if stop_updates[2]>30:
                break
            print(info_out_print + acc_out + best_acc_out + "\n use time: [{:2f}]".format(mytime) + '\n') 
        
        print(info_out_print + best_acc_out + "\n epoch use time: [{:2f}]".format(sum(times)/len(times)))
        best_accs_folds.append(best_accs)
        mgda_s.append(epoch_mgdas)

    best_accs_folds = np.array(best_accs_folds)[:,2]
    mean = best_accs_folds.mean()*100
    std = best_accs_folds.std()*100
    print('test:[{:.2f} ± {:.2f}]'.format(mean,std))
    
    savefile = 'saved/model/' + args.dataset+ '/' + args.model + args.train_model
    if args.double_co_use_mgda:
        savefile = savefile + '2mgda'
    if args.mgda_with_double_co:
        savefile = savefile + '2co'
    torch.save(model, savefile + '.pt')

def train_causal_real(dataset=None, model_func=None, args=None,test_data_set = None):
    train_accs, test_accs, test_accs_c, test_accs_o = [], [], [], []
    random_guess = 1.0 / dataset.num_classes
    best_epochs = []
    times = []
    losses_saved =  []
    temp_mgda_not_use = [0 for i in range(args.folds)]
    temp_attlossuse = args.attion_loss
    mgda_s = [] 
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.epoch_select))):
        train_causal_epoch = funcs[args.train_model]
        losses = []
        if fold > 1:
            np.save(args.accsave,np.array(best_epochs))
        best_test_acc, best_epoch, best_test_acc_c, best_test_acc_o = 0, 0, 0, 0
        if test_data_set == None :
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx.long()]
        else:
            train_dataset = dataset
            test_dataset = test_data_set
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
        args.attion_loss = temp_attlossuse
        model = model_func(dataset.num_features, dataset.num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lastc = []
        epoch_mgdas = []
        for epoch in range(1, args.epochs + 1): 
            mytime,lastc,train_loss, loss_c, loss_o, loss_co, train_acc,mgda = train_causal_epoch(model, optimizer, train_loader, device,lastc,args)
            test_acc, test_acc_c, test_acc_o = eval_acc_causal(model, test_loader, device, args)
            losses.append([train_loss, loss_c, loss_o, loss_co])
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            test_accs_c.append(test_acc_c)
            test_accs_o.append(test_acc_o)
            epoch_mgdas.append(mgda)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                best_test_acc_c = test_acc_c
                best_test_acc_o = test_acc_o
            
            times.append(mytime)
            best_epochs.append(best_test_acc)
            print("Causal | dataset:[{}] fold:[{}] | Epoch:[{}/{}] Loss:[{:.4f}={:.4f}+{:.4f}+{:.4f}] Train:[{:.4f}] Test:[{:.2f}] Test_o:[{:.2f}] Test_c:[{:.2f}] (RG:{:.2f}) | Best Test:[{:.2f}] at Epoch:[{}] | Test_o:[{:.2f}] Test_c:[{:.2f}] | use time : {:.6f}"
                    .format(args.dataset,
                            fold,
                            epoch, args.epochs,
                            train_loss, loss_c, loss_o, loss_co,
                            train_acc * 100,  
                            test_acc * 100, 
                            test_acc_o * 100,
                            test_acc_c * 100, 
                            random_guess*  100,
                            best_test_acc * 100, 
                            best_epoch,
                            best_test_acc_o * 100,
                            best_test_acc_c * 100,
                            mytime))
        losses_saved.append(losses)
        mgda_s.append(epoch_mgdas)
        print("syd: Causal fold:[{}] | Dataset:[{}] Model:[{}] | Best Test:[{:.2f}] at epoch [{}] | Test_o:[{:.2f}] Test_c:[{:.2f}] (RG:{:.2f})"
                .format(fold,
                        args.dataset,
                        args.model,
                        best_test_acc * 100, 
                        best_epoch,
                        best_test_acc_o * 100,
                        best_test_acc_c * 100,
                        random_guess*  100))
    savefile = 'saved/loss/' + args.dataset+ '/' + args.model + args.train_model
    if args.double_co_use_mgda:
        savefile = savefile + '2mgda'
    if args.mgda_with_double_co:
        savefile = savefile + '2co'
    np.save(savefile,np.array(losses_saved))
    train_acc, test_acc, test_acc_c, test_acc_o = tensor(train_accs), tensor(test_accs), tensor(test_accs_c), tensor(test_accs_o)
    train_acc = train_acc.view(args.folds, args.epochs)
    test_acc = test_acc.view(args.folds, args.epochs)
    test_acc_c = test_acc_c.view(args.folds, args.epochs)
    test_acc_o = test_acc_o.view(args.folds, args.epochs)

    savefile = 'saved/ACC/' + args.dataset+ '/' + args.model + args.train_model
    if args.double_co_use_mgda:
        savefile = savefile + '2mgda'
    if args.mgda_with_double_co:
        savefile = savefile + '2co'
    np.save(savefile + 'train_acc.npy',train_acc.detach().numpy())
    np.save(savefile + 'test_acc.npy',test_acc.detach().numpy())
    np.save(savefile + 'test_acc_c.npy',test_acc_o.detach().numpy())
    np.save(savefile + 'test_acc_o.npy',test_acc_o.detach().numpy())

    
    savefile = 'saved/MGDA/' + args.dataset+ '/' + args.model + args.train_model
    if args.double_co_use_mgda:
        savefile = savefile + '2mgda'
    if args.mgda_with_double_co:
        savefile = savefile + '2co'
        
    np.save(savefile,np.array(mgda_s))
    
    _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
    selected_epoch = selected_epoch.repeat(args.folds)
    
    _, selected_epoch2 = test_acc_o.mean(dim=0).max(dim=0)
    selected_epoch2 = selected_epoch2.repeat(args.folds)

    test_acc = test_acc[torch.arange(args.folds, dtype=torch.long), selected_epoch]
    test_acc_c = test_acc_c[torch.arange(args.folds, dtype=torch.long), selected_epoch]
    test_acc_o = test_acc_o[torch.arange(args.folds, dtype=torch.long), selected_epoch2]

    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    test_acc_c_mean = test_acc_c.mean().item()
    test_acc_c_std = test_acc_c.std().item()
    test_acc_o_mean = test_acc_o.mean().item()
    test_acc_o_std = test_acc_o.std().item()
    best_epochs_mean = sum(best_epochs)/args.folds

    print("=" * 150)
    print('sydall Final: Causal | Dataset:[{}] Model:[{}] seed:[{}]| best_epoch = {:.2f}|Test Acc: {:.2f}±{:.2f} | OTest: {:.2f}±{:.2f}, CTest: {:.2f}±{:.2f} (RG:{:.2f}) | [Settings] co:{},c:{},o:{},harf:{},dim:{},fc:{} | epoch use time :{:.6f}'
         .format(args.dataset,
                 args.model,
                 args.seed,
                 best_epochs_mean,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 test_acc_o_mean * 100, 
                 test_acc_o_std * 100,
                 test_acc_c_mean * 100, 
                 test_acc_c_std * 100,
                 random_guess*  100,
                 args.co,
                 args.c,
                 args.o,
                 args.harf_hidden,
                 args.hidden,
                 args.fc_num,
                 sum(times)/len(times)))
    print("=" * 150)

def eval_acc_causal(model, loader, device, args):
    model.eval()
    eval_random = args.eval_random
    correct = 0
    correct_c = 0
    correct_o = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
            pred = co_logs.max(1)[1]
            pred_c = c_logs.max(1)[1] 
            pred_o = o_logs.max(1)[1] 
        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_c += pred_c.eq(data.y.view(-1)).sum().item()
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()

    acc_co = correct / len(loader.dataset)
    acc_c = correct_c / len(loader.dataset)
    acc_o = correct_o / len(loader.dataset)
    return acc_co, acc_c, acc_o
