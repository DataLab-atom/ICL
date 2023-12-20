import torch
from utils import num_graphs
import numpy as np
from min_norm_solvers import MinNormSolver
import torch.nn.functional as F 
import torch.nn as nn
from torch.autograd import Variable
import time

"""
这个文件存放的都是训练时每个epoch会执行的函数，它们具有相同的输入和输出
"""

def NTXentLoss(zis, zjs, temperature=0.5, use_cosine_similarity=True):
    if use_cosine_similarity:
        # 如果使用余弦相似度
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        similarity_matrix = torch.mm(zis, zjs.t())
    else:
        # 如果使用点积
        similarity_matrix = torch.mm(zis, zjs.t())
            
    # 计算log_prob
    exp_similarity_matrix = torch.exp(similarity_matrix / temperature)
    sum_of_rows = torch.sum(exp_similarity_matrix, dim=1)
    log_prob = similarity_matrix - torch.log(sum_of_rows)
        
    # 计算正样本的log似然均值
    mean_log_prob_pos = torch.mean(torch.diag(log_prob))
        
    # 计算损失
    loss = - mean_log_prob_pos
        
    return loss

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
def get_loss(h1, h2, temperature):
    f = lambda x: torch.exp(x / temperature)
    refl_sim = f(sim_matrix(h1, h1))        # intra-view pairs
    between_sim = f(sim_matrix(h1, h2))     # inter-view pairs
    x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
    loss = -torch.log(between_sim.diag() / x1)
    return loss
    
def unname_loss(h1, h2, temperature):
    loss1 = get_loss(h1,h2,temperature)/2
    loss2 = get_loss(h2,h1,temperature)/2
    return (loss1 + loss2).mean()

def static_weight_loss(model, optimizer, loader, device,lastc, args):
    
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    eval_random=args.with_random

    for it, data in enumerate(loader):
    
        optimizer.zero_grad()
        data = data.to(device)    
        one_hot_target = data.y.view(-1)

        c_logs,o_logs,co_logs = model(data)
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch

        #uniform_target = torch.sigmoid(torch.randn_like(c_logs, dtype=torch.float).to(device)) 
        uniform_target = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes

        c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        o_loss = F.nll_loss(o_logs, one_hot_target) 
        co_loss = F.nll_loss(co_logs, one_hot_target) 
        
        loss =  args.co*co_loss + args.o*o_loss + args.c*c_loss 
        start = time.time()
        loss.backward()
        mytime = (time.time() - start)*1000
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        
        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return mytime,lastc,total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o,0

def get_parameters_grad(model,using_xc = True):
    grads = []
    for param in model.Big.parameters():
        if param.grad is not None:
            grads.append(Variable(param.grad.data.clone(), requires_grad=False))
    return grads

def mgda_loss(model, optimizer, loader, device, lastc , args):
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    eval_random=args.with_random
    criterion = torch.nn.SmoothL1Loss()

    mgda = []
    for it, data in enumerate(loader):
    
        optimizer.zero_grad()
        data = data.to(device)    
        one_hot_target = data.y.view(-1)

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        #+++++++++++++++++++++++++++forward+++++++++++++++++++++++++++
        big_ = model.Big(x,edge_index,model.use_bns_conv)
        
        xo_,xo_edge_att,xo_node_att = model.xo(big_,edge_index,batch,eval_random)
        xc_,xc_edge_att,xc_node_att = model.xc(big_,edge_index,batch,eval_random)
        
        c_logs = model.c(xc_)        
        o_logs = model.o(xo_)
        co_logs = model.co(xc_,xo_,eval_random)

        uniform_target_1 = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        #uniform_target_1 = torch.sigmoid(torch.randn_like(c_logs, dtype=torch.float).to(device)) 
        
        #+++++++++++++++++++++++++++++++++mgda++++++++++++++++++++++

        loss_data = {}
        grads = {}
        start = time.time()
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        c_loss = F.kl_div(c_logs, uniform_target_1, reduction='batchmean')# + F.kl_div(c_logs, uniform_target_2, reduction='batchmean') 
        loss_data['c'] = c_loss.data
        c_loss.backward(retain_graph=True)
        grads['c'] = get_parameters_grad(model)
        model.zero_grad()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  

        o_loss = F.nll_loss(o_logs, one_hot_target)
        loss_data['o'] = o_loss.data
        o_loss.backward(retain_graph=True)
        grads['o'] =  get_parameters_grad(model,False)
        model.zero_grad()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        co_loss = F.nll_loss(co_logs, one_hot_target) 
        
        if args.mgda_with_double_co:
            co_logs_2 = model.co(xc_,xo_,eval_random)
            co_loss_2 = torch.pairwise_distance(co_logs,co_logs_2) #F.kl_div(co_logs - co_logs_2, uniform_target_1, reduction='batchmean')#criterion(,)
            co_loss + co_loss + co_loss_2
        
        loss_data['co'] = co_loss.data
        co_loss.backward(retain_graph=True)
        grads['co'] =  get_parameters_grad(model,False)
        model.zero_grad()
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        loss_name = ['o','c']
        gn = MinNormSolver.gradient_normalizers(grads, loss_data, args.mgda_model)#姑且认为l2不会出问题
        for name in loss_name:
            if gn[name] < 1e-3:
                gn[name] = torch.tensor(1e-3)

        for t in loss_data:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
        sol, _ = MinNormSolver.find_min_norm_element_FW([grads[t] for t in loss_name])
        sol = {k:sol[i] for i, k in enumerate(loss_name)}
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        loss = sol['c']*c_loss +  sol['o']*o_loss + co_loss      
            
        attion_loss = 1/criterion(xo_edge_att,xc_edge_att) + 1/criterion(xo_node_att,xc_node_att)
        if attion_loss > 5 and args.attion_loss:
            loss = loss + attion_loss
       
        loss.backward()
        mytime = (time.time() - start)*1000
        optimizer.step()
        mgda.append([float(sol['c']) , float(sol['o'])])  
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        
    mgda = torch.tensor(mgda)
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return mytime,lastc,total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o,mgda


def mgda_loss_3(model, optimizer, loader, device, lastc , args):
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_o = 0
    total_loss_co = 0
    correct_o = 0
    eval_random=args.with_random
    criterion = torch.nn.SmoothL1Loss()

    for it, data in enumerate(loader):
    
        optimizer.zero_grad()
        data = data.to(device)    
        one_hot_target = data.y.view(-1)

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        #+++++++++++++++++++++++++++forward+++++++++++++++++++++++++++
        big_ = model.Big(x,edge_index,model.use_bns_conv)

        xo_,_,_ = model.xo(big_,edge_index,batch,eval_random)
        xc_,_,_ = model.xc(big_,edge_index,batch,eval_random)
        
        c_logs = model.c(xc_)        
        o_logs = model.o(xo_)
        co_logs = model.co(xc_,xo_,eval_random)

        uniform_target_1 = torch.ones_like(c_logs, dtype=torch.float).to(device) / model.num_classes
        #uniform_target_1 = torch.sigmoid(torch.randn_like(c_logs, dtype=torch.float).to(device)) 
        
        #+++++++++++++++++++++++++++++++++mgda++++++++++++++++++++++

        loss_data = {}
        grads = {}
        start = time.time()
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        c_loss = F.kl_div(c_logs, uniform_target_1, reduction='batchmean')# + F.kl_div(c_logs, uniform_target_2, reduction='batchmean') 
        '''
        if(len(lastc) == it):
            lastc.append(c_logs.detach().clone())
            c_loss = F.kl_div(c_logs, uniform_target, reduction='batchmean')
        else:
            c_loss = criterion(c_logs,co_logs)#F.kl_div(c_logs, uniform_target, reduction='batchmean')# # F.pairwise_distance(c_logs,lastc[it]).mean() #
            lastc[it] = c_logs.detach().clone()
        ''' 
        loss_data['c'] = c_loss.data
        c_loss.backward(retain_graph=True)
        grads['c'] = get_parameters_grad(model)
        model.zero_grad()

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  

        o_loss = F.nll_loss(o_logs, one_hot_target)
        loss_data['o'] = o_loss.data
        o_loss.backward(retain_graph=True)
        grads['o'] =  get_parameters_grad(model,False)
        model.zero_grad()


        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        co_logs_2 = model.co(xc_,xo_,eval_random)
        co_loss_2 = criterion(co_logs,co_logs_2)#F.nll_loss(, .max(1)[1])
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        co_loss = F.nll_loss(co_logs, one_hot_target) 
        
        #两次干扰，对结果取交叉熵
        if args.mgda_with_double_co:
            if args.double_co_use_mgda and co_loss_2 > 1e-3 and c_loss > 1e-3:
                mini_grad = {}
                mini_loss_data = {}
                co_loss.backward(retain_graph=True)
                mini_grad['co1'] = get_parameters_grad(model)
                mini_loss_data['co1'] = co_loss.data
                for param in model.co.parameters():
                    if param.grad is not None:
                        mini_grad['co1'].append(Variable(param.grad.data.clone(), requires_grad=False))

                model.zero_grad()

                co_loss_2.backward(retain_graph=True)
                mini_loss_data['co2'] = co_loss_2.data 
                mini_grad['co2'] = get_parameters_grad(model)
                
                for param in model.co.parameters():
                    if param.grad is not None:
                        mini_grad['co2'].append(Variable(param.grad.data.clone(), requires_grad=False))

                model.zero_grad()            
                
                gn = MinNormSolver.gradient_normalizers(mini_grad, mini_loss_data, args.mgda_model)#姑且认为l2不会出问题
                for t in mini_loss_data:
                    for gr_i in range(len(mini_grad[t])):
                        mini_grad[t][gr_i] = mini_grad[t][gr_i] / gn[t].to(mini_grad[t][gr_i].device)
                sol, _ = MinNormSolver.find_min_norm_element_FW([mini_grad[t] for t in ['co1','co2' ]])
                sol = {k:sol[i] for i, k in enumerate(['co1','co2' ])}

                co_loss = float(sol['co1'])*co_loss + float(sol['co2'])*co_loss_2
            else:
                co_loss = co_loss + co_loss_2
        
        #loss_data['co'] = co_loss.data
        #co_loss.backward(retain_graph=True)
        #grads['co'] =  get_parameters_grad(model)
        #model.zero_grad()

        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        loss_name = ['c','o']
       
        #两次干扰，对结果取欧式距离
        if args.mgda_with_loss_4:
            loss_name.append('co_2')
            loss_data['co_2'] = co_loss_2.data
            co_loss.backward(retain_graph=True)
            grads['co_2'] =  get_parameters_grad(model)
            model.zero_grad()
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        shut_name = []
        temp = []
        for name in loss_name:
            if loss_data[name] > 1e-3:
                temp.append(name)
            else:
                shut_name.append(name)
        
        loss_name = temp
        sol = {}
        if len(loss_name)>2:
            gn = MinNormSolver.gradient_normalizers(grads, loss_data, args.mgda_model)#姑且认为l2不会出问题
            for t in loss_data:
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
            sol, _ = MinNormSolver.find_min_norm_element_FW([grads[t] for t in loss_name])
            sol = {k:sol[i] for i, k in enumerate(loss_name)}
        else:
            for name in loss_name:
                sol[name] = 1
        for name in shut_name:
            sol[name] = 1
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        loss = 0
        loss = float(sol['c']) * c_loss + float(sol['o']) *o_loss +  co_loss        
        if args.mgda_with_loss_4:
            loss = loss + float(sol['co_2']) * co_loss_2
        
        loss.backward()
        mytime = (time.time() - start)*1000
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

        pred_o = o_logs.max(1)[1]
        correct_o += pred_o.eq(data.y.view(-1)).sum().item()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_o += o_loss.item() * num_graphs(data)
        total_loss_co += co_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_o = total_loss_o / num
    total_loss_co = total_loss_co / num
    correct_o = correct_o / num
    return mytime,lastc,total_loss, total_loss_c, total_loss_o, total_loss_co, correct_o


# 函数句柄，用于在外部直接获取这个文件中的函数
funcs = {
    'swl':static_weight_loss,
    'mgda':mgda_loss,
    'mgda3':mgda_loss_3
}

