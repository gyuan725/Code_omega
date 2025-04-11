"""
Dataload

"""
import torch
#from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import pandas as pd
import os
np.random.seed(2024)

class Dataset(InMemoryDataset):
    def __init__(self, root, dataset, gama, pred_edges=1, transform=None, pre_transform=None):


        self.path = root
        self.dataset = dataset
        self.pred_edges= pred_edges 
        self.gama = gama

        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistical_info = torch.load(self.processed_paths[1])
        self.node_num = self.statistical_info['node_num']
        self.data_num = self.statistical_info['data_num']
        self.criteria_num = self.statistical_info['criteria_num']
        

    @property
    def raw_file_names(self):
        return ['{}{}/{}.csv'.format(self.path, self.dataset, self.dataset), \
                '{}{}/{}.edge'.format(self.path, self.dataset, self.dataset)]

    @property
    def processed_file_names(self):
 

        return ['{}/{}.dataset'.format(self.dataset, self.dataset + '_emb' + str(self.gama)), \
                '{}/{}.info'.format(self.dataset, self.dataset + '_emb' + str(self.gama))]
        # return ['{}/{}.dataset'.format(self.dataset, self.dataset + '_emb'), \
        #         '{}/{}.info'.format(self.dataset, self.dataset + '_emb')]


    def download(self):
        # Download to `self.raw_dir`.
        pass

    def read_data(self):
        # handle node and class 
        node_list = []
        label = []
        max_node_index = 0
        data_num = 0
        
        df = pd.read_csv(self.datafile)
       
        node_feature = df[df.columns[0:-1]].values
        node_feature_list = node_feature.tolist()
        label = df.iloc[:, -1].values - 1
        criteria_num = len(node_feature[0])
        c = 0
        for col in df.columns:
            df[col] =  pd.factorize(df[col])[0]
            if c >=1 and c < len(df.columns)-1:
                df[df.columns[c]] = df[df.columns[c]] + (max(df[df.columns[c-1]]) + 1)
            c += 1
            
        node_index = df.values.tolist()
        V = self.piecewise_point(node_feature, self.gama)
            
        for row in range(len(node_index)):
            data_num += 1
            data = node_index[row]
            # label.append(float(data[-1]))
            int_list = [int(data[i]) for i in range(len(data))[0:-1]]
            node_list.append(int_list)
        #     if max_node_index < max(int_list):
        #         max_node_index = max(int_list)            

        max_node_index = df.values.max()

        
        edge_list = []
        sr_list = []    #sender_receiver_list, containing node index
        for nodes in V:
            edge_l, sr_l = self.construct_full_edge_list(nodes)
            edge_list.append(edge_l)
            sr_list.append(sr_l)



        return V, edge_list, label, sr_list, criteria_num, max_node_index + 1, data_num

    def construct_full_edge_list(self, nodes):
        num_node = len(nodes)
        edge_list = [[],[]]         #first for sender, second for receiver
        sender_receiver_list = []
        for i in range(num_node):
            for j in range(num_node)[i+1:]:
                edge_list[0].append(i)
                edge_list[1].append(j)
                sender_receiver_list.append([nodes[i],nodes[j]])

        return edge_list, sender_receiver_list
    
     
    def piecewise_point(self, node_feature, gama):
        
        X = node_feature
        num_g = X.shape[1]
        num_n = X.shape[0]
        z = np.zeros([num_g, gama+1])
        for i in range(num_g):
            for j in range(gama+1):
                z[i,j] = min(X[:,i]) + j / gama * (max(X[:,i]) - min(X[:,i]))
        v = np.zeros([num_n, num_g, gama])
        for i in range(num_n):
            for j in range(num_g):
                for k in range(gama):
                    v[i,j,k] = np.where( z[j,k] <= X[i,j] <= z[j,k+1], (X[i,j]- z[j,k]) / (z[j,k+1] - z[j,k]), np.where( X[i,j] < z[j,k], 0 , 1))
                
        return v.tolist()
        

    def process(self):
        self.datafile, self.edgefile = self.raw_file_names
        self.node, edge, label, self.sr_list, criteria_num, node_num, data_num = self.read_data()

        data_list = []
        sr_data = []
        for i in range(len(self.node)):
            node_features = torch.Tensor(self.node[i])
            # node_features = torch.Tensor(self.node[i]).unsqueeze(1)
            # node_features = torch.LongTensor(self.node[i]).unsqueeze(1)
            x = node_features
            edge_index = torch.LongTensor(edge[i])
            y = torch.tensor(label[i])
            sr = torch.Tensor(self.sr_list[i])
            #y = torch.FloatTensor(label[i])
            # if self.pred_edges == 1:
            #     sr = torch.Tensor(self.sr_list[i])
            #     # sr = torch.LongTensor(self.sr_list[i])     #the sender receiver list, stored in edge_attr
            # else:
            #     sr = []

            data = Data(x=x, edge_index=edge_index, edge_attr=sr,  y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        statistical_info = {'criteria_num': criteria_num,'data_num': data_num, 'node_num': node_num}
        torch.save(statistical_info, self.processed_paths[1])


    def node_M(self):
        return self.node_num
    
    def data_N(self):
        return self.data_num
    
    def criteria_N(self):
        return self.criteria_num

    """
    def len(self):
        return len(self.node)
    def get(self, idx):
        ###
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    """