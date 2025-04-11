import numpy as np
import pandas as pd
import pickle
import torch
from dataload import Dataset
import argparse
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold

from model_train import train
torch.manual_seed(2024)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CEV', help='which dataset to use')
parser.add_argument('--gama', type=int, default=4, help='number of piecewise ponit')
parser.add_argument('--l0_weight', type=float, default= 0.02, help='weight of the l0 regularization term')
parser.add_argument('--l2_weight', type=float, default=0.001, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--l0_para', nargs='?', default='[0.66, -0.1, 1.1]',
                        help="l0 parameters, which are beta (temprature), \
                            zeta (interval_min) and gama (interval_max).")
parser.add_argument('--hidden_layer', type=int, default= 16, help='neural hidden layer')
parser.add_argument('--pred_edges', type=int, default= 1, help='1: predict edges , 0: consider all interactions')
parser.add_argument('--random_seed', type=int, default=2024)
args = parser.parse_args()

# dataset = Dataset('../data/', args.dataset, pred_edges=args.pred_edges)
dataset = Dataset('../data/', args.dataset, args.gama, pred_edges=args.pred_edges)
num_g = dataset.criteria_N()
data_num = dataset.data_N()
t_dim = dataset.y.max()



def cross_val( args, dataset, show_loss):
    
    # dataset = dataset.shuffle()
    skf = StratifiedKFold(n_splits=5,shuffle = True, random_state= args.random_seed) 
    # skf = StratifiedKFold(n_splits=5) 
    metrics = np.zeros([5,4])
    df = pd.DataFrame()
    para = np.array([args.lr,args.l0_weight,args.gama,args.pred_edges])
    
    for i, (train_idx, test_idx) in enumerate(skf.split(dataset, dataset.y)):
        
        train_dataset = dataset[torch.LongTensor(train_idx)]
        test_dataset = dataset[torch.LongTensor(test_idx)]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        datainfo = [train_loader, test_loader, num_g, t_dim]
        df_fold, val_metrics = train(args, datainfo)
        metrics[i,:] = val_metrics
        
        df = pd.concat([df, df_fold])
    
    return metrics, df




dataset = dataset.shuffle()
train_index = int(len(dataset)* 0.8)
train_dataset = dataset[:train_index]
test_dataset = dataset[train_index:]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

datainfo = [train_loader, test_loader, num_g, t_dim]
df, metrics = train(args, datainfo)


# # metrics, df_no = cross_val( args, dataset, show_loss)
