"""
Train model

"""
import numpy as np
import pandas as pd
import torch
from mymodel import Comp_value, MyLoss
from metrics import metric_measures


def train(args, data_info):
    train_loader = data_info[0]
    val_loader = data_info[1]
    num_g = data_info[2]
    t_dim = data_info[3]
    
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'val_acc', 'val_loss','val_edge'])

    model = Comp_value(args, num_g, args.gama)
    loss_func = MyLoss(t_dim)

    optimizer = torch.optim.AdamW([{"params":model.parameters()},{"params":loss_func.parameters()}], lr=args.lr, weight_decay= 1e-2)

    for epoch in range(args.n_epoch):
        # training
        step = 1
        loss_all = 0
        edge_all = 0
        train_acc_sum = 0
        model.train()
        
        for step, data in enumerate(train_loader, 1):
           
            output, l0_penaty, num_edges = model(data)
            label = data.y
            baseloss, t = loss_func(output, label)
            l0_loss = l0_penaty * args.l0_weight 
            loss = baseloss + l0_loss 
            loss_all += loss.item()
            train_acc = metric_measures(output, label, t)[0]
            train_acc_sum += train_acc.item()
            optimizer.zero_grad()             
            loss.backward()
            optimizer.step()


        cur_loss = loss_all / step
        cur_acc = train_acc_sum / step

        val_acc, val_precision, val_f1, val_recall, val_loss, val_edges= evaluate(model, loss_func, val_loader)    
     
        info = (epoch, cur_loss, cur_acc, val_acc, val_loss/step, val_edges)
        dfhistory.loc[epoch] = info
        val_metrics = [val_acc, val_precision, val_f1, val_recall]
        print('Epoch: {:03d}, Loss: {:.4f}, Train_Acc: {:.4f}, Val_Acc: {:.4f}, Val_loss: {:.4f}, Train edges: {:07d}'.
          format(epoch, cur_loss, cur_acc, val_acc, val_loss/step, val_edges))
             
    return dfhistory, val_metrics


def evaluate(model, loss_func, loader):
    
    model.eval()

    predictions = []
    labels = []
    edges_all = 0
    val_loss_sum = 0
    step = 1
    with torch.no_grad():
        for data in loader:
            pred, _, num_edges = model(data)
            # pred, num_edges = model(data)
            # pred = pred.detach().cpu().numpy()
            edges_all += num_edges
            label = data.y
            predictions.append(pred)
            labels.append(label)
            val_loss, t = loss_func(pred, label)
            val_loss_sum += val_loss.item()
    

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    
    acc, precision, f1, recall = metric_measures(predictions, labels, t)
    return acc, precision, f1, recall, val_loss_sum, edges_all

