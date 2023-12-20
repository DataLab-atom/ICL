from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
from gcn_conv import GCNConv
import random
import pdb       
class BIG(torch.nn.Module):
    def __init__(self , num_features,
                        args,
                        bns_conv,
                        convs):
        super(BIG, self).__init__()
        self.args = args
        
        hidden_in = num_features
        hidden = args.hidden
        self.dropout = nn.Dropout(args.drop_out)

        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = bns_conv
        self.convs = convs
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self,x,edge_index,use_bns_conv = True):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        if use_bns_conv :
            for i, conv in enumerate(self.convs):
                x = self.bns_conv[i](x)
                x = F.relu(conv(x, edge_index))
        else:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
        self.dropout(x)
        return x

class Att_cov(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden,
                gfn=False,edge_norm=True):
        super(Att_cov, self).__init__()
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        self.edge_att_mlp = nn.Linear(in_channels*2, 2)
        self.node_att_cov = GConv(in_channels,2)
        
    def forward(self, x,edge_index):
        row, col = edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep),dim = -1)
        node_att = F.softmax(self.node_att_cov(x,edge_index),dim = -1)
        
        return edge_att,node_att

class XC(torch.nn.Module):
    def __init__(self, args,att_cov,gfn=False,edge_norm=True):
        super(XC, self).__init__()
        self.args = args
        
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        hidden = args.hidden
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        self.dropout = nn.Dropout(args.drop_out)
        
        self.att_cov = att_cov
        self.bn = BatchNorm1d(hidden)
        self.conv = GConv(hidden, hidden)
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
    
    def forward(self, x,edge_index, batch,eval_random=True):
        edge_att,node_att = self.att_cov(x,edge_index)
        edge_att,node_att = edge_att[:, 0],node_att[:, 0].view(-1, 1)
        xc = node_att * x
        xc = F.elu(self.conv(self.bn(xc), edge_index, edge_att))
        xc = self.dropout(xc)
        xc = self.global_pool(xc, batch)
        return xc,edge_att,node_att

class XO(torch.nn.Module):
    def __init__(self,args,att_cov,gfn=False,edge_norm=True):
        super(XO, self).__init__()
        self.args = args
        
        self.without_node_attention = args.without_node_attention
        self.without_edge_attention = args.without_edge_attention
        hidden = args.hidden
        self.dropout = nn.Dropout(args.drop_out)
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        
        self.att_cov = att_cov
        self.bn = BatchNorm1d(hidden)
        self.conv = GConv(hidden, hidden)
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self,x,edge_index, batch, eval_random=True):
        edge_att,node_att = self.att_cov(x,edge_index)
        edge_att,node_att = edge_att[:, 1],node_att[:, 1].view(-1, 1)
        xo = node_att * x
        xo = F.elu(self.conv(self.bn(xo), edge_index, edge_att))
        xo = self.dropout(xo)
        xo = self.global_pool(xo, batch)
        return xo,edge_att,node_att

class C(torch.nn.Module):
    def __init__(self, num_classes, args):
        super(C, self).__init__()

        self.args = args
        
        hidden = args.hidden
        hidden_out = num_classes
        
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)


    def forward(self,x):
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

class O(torch.nn.Module):
    def __init__(self, num_classes, args):
        super(O, self).__init__()

        self.args = args
        
        hidden = args.hidden
        hidden_out = num_classes
        self.dorp = torch.nn.Dropout1d(p=0.4)
        
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)


    def forward(self,x):
        x = self.dorp(x)
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

class CO(torch.nn.Module):
    def __init__(self, num_classes, args):
        super(CO, self).__init__()

        hidden = args.hidden
        hidden_out = num_classes
        self.args = args

        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)


    def forward(self,xc,xo,eval_random):
        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo
        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.elu(x)
        x = F.elu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

class CausalGCN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       collapse=False, 
                       residual=False,
                       res_branch="BNConvReLU", 
                       global_pool="sum", 
                       dropout=0.5, 
                       edge_norm=True):
        super(CausalGCN, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.num_classes=num_classes
        self.use_bns_conv = True
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        bns_conv = torch.nn.ModuleList()
        convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            bns_conv.append(BatchNorm1d(hidden))
            convs.append(GConv(hidden, hidden))
        attcov = Att_cov(hidden,hidden)
        self.Big = BIG(num_features,args,bns_conv,convs)
        self.xc = XC(args,attcov,gfn,edge_norm)
        self.xo = XO(args,attcov,gfn,edge_norm)
        self.c = C(num_classes,args)
        self.o = O(num_classes,args)
        self.co = CO(num_classes,args)

    def forward(self, data, eval_random=True):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        x = self.Big(x,edge_index)
        xc_,_,_ = self.xc(x,edge_index,batch,eval_random)
        xo_,_,_ = self.xo(x,edge_index,batch,eval_random)
        
        xc_logis = self.c(xc_)
        xo_logis = self.o(xo_)
        xco_logis = self.co(xc_,xo_,eval_random)

        return xc_logis, xo_logis, xco_logis

class CausalGIN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                gfn=False,
                edge_norm=True):
        super(CausalGIN, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.num_classes=num_classes
        self.use_bns_conv = False
        self.dropout = nn.Dropout(args.drop_out)
        bns_conv = torch.nn.ModuleList()
        convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            convs.append(GINConv(
            Sequential(
                       Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU())))
        attcov = Att_cov(hidden,hidden)
        self.Big = BIG(num_features,args,bns_conv,convs)
        self.xc = XC(args,attcov,gfn,edge_norm)
        self.xo = XO(args,attcov,gfn,edge_norm)
        self.c = C(num_classes,args)
        self.o = O(num_classes,args)
        self.co = CO(num_classes,args)

    def forward(self, data, eval_random=True):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        x = self.Big(x,edge_index,False)
        xc_,_,_ = self.xc(x,edge_index,batch,eval_random)
        xo_,_,_ = self.xo(x,edge_index,batch,eval_random)

        xc_logis = self.c(xc_)
        xo_logis = self.o(xo_)
        xco_logis = self.co(xc_,xo_,eval_random)

        return xc_logis, xo_logis, xco_logis
    
class CausalGAT(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self,  num_features,
                        num_classes, args,
                        head=4, 
                        dropout=0.2,
                        gfn=False,
                        edge_norm=True):
        super(CausalGAT, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.num_classes=num_classes
        self.use_bns_conv = True

        bns_conv = torch.nn.ModuleList()
        convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            bns_conv.append(BatchNorm1d(hidden))
            convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        
        attcov = Att_cov(hidden,hidden)
        self.Big = BIG(num_features,args,bns_conv,convs)
        self.xc = XC(args,attcov,gfn,edge_norm)
        self.xo = XO(args,attcov,gfn,edge_norm)
        self.c = C(num_classes,args)
        self.o = O(num_classes,args)
        self.co = CO(num_classes,args)

    def forward(self, data, eval_random=True):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        x = self.Big(x,edge_index)
        xc_,_,_ = self.xc(x,edge_index,batch,eval_random)
        xo_,_,_ = self.xo(x,edge_index,batch,eval_random)

        xc_logis = self.c(xc_)
        xo_logis = self.o(xo_)
        xco_logis = self.co(xc_,xo_,eval_random)

        return xc_logis, xo_logis, xco_logis

class GCNNet(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, hidden, 
                       num_feat_layers=1, 
                       num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0.2, 
                 edge_norm=True):
        super(GCNNet, self).__init__()

        self.global_pool = global_mean_pool
        self.dropout = nn.Dropout(dropout)
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
            
        x = self.global_pool(x, batch)
        x = self.dropout(x)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))
        
        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

class GINNet(torch.nn.Module):
    def __init__(self, num_features,
                       num_classes,
                       hidden, 
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0):

        super(GINNet, self).__init__()
        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
        
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
            Sequential(Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU())))

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        # x, edge_index, batch = data.feat, data.edge_index, data.batch
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))    
        x = self.bn_hidden(x)
        x = self.lin_class(x)

        prediction = F.log_softmax(x, dim=-1)
        return prediction

class GATNet(torch.nn.Module):
    def __init__(self, num_features, 
                       num_classes,
                       hidden,
                       head=4,
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0.2):

        super(GATNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
   
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)