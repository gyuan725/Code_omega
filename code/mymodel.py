"""
MyModel abd MyLoss

"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import AttentionalAggregation
from torch.nn import Linear
import numpy as np
from torch_geometric.data import Data
from torch_scatter import scatter_sum
from torch_geometric.utils import add_self_loops, softmax



class Comp_value(nn.Module):
## comprehensive value   
    def __init__(self, args, num_g, gama):
        super(Comp_value, self).__init__()
      
        self.pred_edges = args.pred_edges
        self.num_g = num_g
        self.gama = gama
        self.hidden_layer = args.hidden_layer
        self.l0_para = eval(args.l0_para)       
        
        ## additive value function
        self.indi_value = Add_value(num_g, gama)

        if self.pred_edges==1:
            self.linkpred = LinkPred(self.hidden_layer, self.l0_para, gama)
        
        ## interaction effects
        self.int_value = Interact_value(self.hidden_layer, gama)




    def forward(self, data, is_training=True):

        x, edge_index, sr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        ind_value, marginal_value = self.indi_value(x)

        if self.pred_edges ==1:
            sr = torch.transpose(sr, 0, 1)    # [2, num_edges]
            s, l0_penaty = self.linkpred(sr, is_training)
            # s = self.linkpred(sr, is_training)
            pred_edge_index, pred_edge_weight = self.construct_pred_edge(edge_index, s, sr) 
            updated_nodes = self.int_value(x, pred_edge_index, edge_weight=pred_edge_weight)
            num_edges = pred_edge_weight.size(0)
        else:
            updated_nodes = self.int_value(x, edge_index)
            l0_penaty = 0
            num_edges = edge_index.size(1)           

        graph_embedding = global_mean_pool(updated_nodes, batch)
           
        out = graph_embedding + ind_value
        
        return out, l0_penaty, num_edges  

    def construct_pred_edge(self, fe_index, s, sr):
        """
        fe_index: full_edge_index, [2, all_edges_batchwise]
        s: predicted edge value, [all_edges_batchwise, 1]

        construct the predicted edge set and corresponding edge weights
        """
        new_edge_index = [[],[]]
        edge_weight = []
        s = torch.squeeze(s)

        sender = torch.unsqueeze(fe_index[0][s>0], 0)
        receiver = torch.unsqueeze(fe_index[1][s>0], 0)
        pred_index = torch.cat((sender, receiver ), 0)
        pred_weight = s[s>0]
        

        return pred_index, pred_weight


class Interact_value(MessagePassing):
    def __init__(self, hidden_layer, gama):
        super(Interact_value, self).__init__(aggr='add')
       
        self.a = nn.Parameter(torch.zeros(size=(gama * gama, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.lin = torch.nn.Linear(gama * gama, 1, bias = False)

    def forward(self, x, edge_index, edge_weight = None):

        
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight, edge_index):
        # x_i has shape [E, dim]
        # x_j has shape [E, dim]
        
        pairwise_analysis = torch.matmul(x_i.unsqueeze(-1), x_j.unsqueeze(1)).flatten(1)
        
        att_score = self.leakyrelu(torch.mm(pairwise_analysis, self.a))
        sender, receiver  = edge_index
        att_weight = softmax(att_score, receiver)
        pairwise_analysis =  self.lin(att_weight * pairwise_analysis)
        

        if edge_weight != None:
            interaction_analysis = pairwise_analysis * edge_weight.view(-1,1)
        else:
            interaction_analysis = pairwise_analysis

        return interaction_analysis

    def update(self, aggr_out):
        
        # aggr_out has shape [N, dim]
        return aggr_out


class LinkPred(nn.Module):
    def __init__(self, H, l0_para, gama):

        super(LinkPred, self).__init__()
        
        self.linear1 = nn.Linear(gama * gama, H)
        self.linear2 = nn.Linear(H, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
       

        self.temp = l0_para[0]      #temprature
        self.inter_min = l0_para[1] 
        self.inter_max = l0_para[2] 
        self.hardtanh = nn.Hardtanh(0,1)

    def forward(self, sender_receiver, is_training):
        
        sender = sender_receiver[0,:,:].unsqueeze(-1)
        receiver = sender_receiver[1,:,:].unsqueeze(1)
        _input = torch.matmul(sender,receiver).flatten(1)
        h_relu = self.relu(self.linear1(_input))
        h_relu = self.dropout(h_relu)
        loc = self.linear2(h_relu)
        if is_training:
            u = torch.rand_like(loc)
            logu = torch.log2(u)
            logmu = torch.log2(1-u)
            sum_log = loc + logu - logmu
            s = torch.sigmoid(sum_log/self.temp)
            s = s * (self.inter_max - self.inter_min) + self.inter_min
        else:
            s = torch.sigmoid(loc) * (self.inter_max - self.inter_min) + self.inter_min

        s = self.hardtanh(s)

        l0_penaty = torch.sigmoid(loc - self.temp * np.log2(-self.inter_min/self.inter_max)).mean()

        return s, l0_penaty

    def permutate_batch_wise(x, batch):
        """
        x: all feature embeddings all batch
        batch: a list containing feature belongs to which graph
        """
        return


class Add_value(nn.Module):
    def __init__(self, num_g, gama):
        super(Add_value, self).__init__()
        self.num_g = num_g
        self.gama = gama
        self.layers1 = nn.ModuleList()
        
        for i in range(num_g):
            self.layers1.append(nn.Linear(gama, 1))

        self.outlayer = nn.Linear(num_g,1,bias=False)

        
    def forward(self, x):
        
        v = x.reshape([-1, self.num_g* self.gama])        
        out1 = []
        
        for i in range(self.num_g):
            out1.append(self.layers1[i](v[:,i * self.gama:i * self.gama+self.gama]))
        
        ##marginal value
        m_out = torch.cat(out1,dim=1)
        
        ##comprehensive value    
        out = self.outlayer(m_out)
        
        return out, m_out
    
   
    
class MyLoss(nn.Module):
    def __init__(self,num_t):
        super(MyLoss, self).__init__()
        self.num_t = num_t
        
        #class thresholds
        self.coefficient = nn.Parameter(torch.randn(num_t))
        self.coefficient.data = torch.sort(self.coefficient)[0]
        #self.sortlayer = nn.Linear(1,1,bias=False)
        
    def forward(self,s,label):
        label = label.type(torch.long)
       
        t = torch.cat([torch.tensor([float('-inf')]), self.coefficient])
        t = torch.cat([t, torch.tensor([float('inf')])])
        
        flag = np.sign(torch.arange(self.num_t)[:,None] - label + 0.5).t()
        e = s - self.coefficient
        loss = e * flag
        loss = torch.log(1 + torch.exp(loss))
        loss = loss.sum() / label.shape[0] 

        return loss, t

