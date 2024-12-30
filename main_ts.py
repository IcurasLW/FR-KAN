from src.efficient_kan.group_kan import *
from src.efficient_kan.kan import KAN
from src.efficient_kan.rational_kan import Rational_KAN
from src.efficient_kan.rbf_kan import RBF_KAN
from src.efficient_kan.fourier_kan import Fourier_KAN


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import warnings
from datasets.load_dataset_ts import data_provider
from sklearn.metrics import mean_squared_error, median_absolute_error
import math
import numpy as np
import random
import logging


fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
warnings.simplefilter(action='ignore', category=UserWarning)


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model', type=str, default="Knot_KAN", help='[Knot_KAN, KAN, MLP]')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--task_name', type=str, default='TimeSeries')
parser.add_argument('--lr', type=float, default=0.001)


# Model Parameters
parser.add_argument('--grid_size', type=int, default=20)
parser.add_argument('--groups', type=int, default=16)
parser.add_argument('--spline_order', type=int, default=3)
parser.add_argument('--smooth_lambda', type=float, default=0.01)
parser.add_argument('--activation', type=str, default='SiLU')


# parser.add_argument('--grid_range', type=list, default=[-10, 10])
# parser.add_argument('--hidden_layer', type=list, default=[64])
# parser.add_argument('--smooth', default=True)
# parser.add_argument('--need_relu', default=True)


### Runing on scripts
parser.add_argument('--hidden_layer', type=int, nargs='*', help='Arbitary list of hidden layer')
parser.add_argument('--smooth',  action='store_true')
parser.add_argument('--need_relu',  action='store_true')
parser.add_argument('--grid_range', type=int, nargs=2, help='Default as [-1, 1]')


## Dataloader param
parser.add_argument('--root_path', type=str, default='/home/nathan/LLM4TS/datasets/forecasting/ETT-small/')
parser.add_argument('--data_name', type=str, default='ett_h')
parser.add_argument('--seq_len', type=int, default=336)
parser.add_argument('--pred_len', type=int, default=720)
parser.add_argument('--label_len', type=int, default=168)
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--freq', type=str, default='h')


args = parser.parse_args()
train_data, train_loader = data_provider(args, 'train')
val_data, val_loader = data_provider(args, 'test')
in_size = args.seq_len 


results_path = f'/home/nathan/KAN_nathan/efficient-kan/outputs/TS_output/{args.model}/{args.data_name}'

if not os.path.exists(results_path):
    print(f"Path does not exist. Creating: {results_path}")
    os.makedirs(results_path)


logging.basicConfig(filename=f'{results_path}/hid_layer_{args.hidden_layer}_{args.data_name}_{args.model}.log', level=logging.INFO)

fold = 3
val_results = []
mean_val_mse = []
mean_val_mae = []
for k in range(fold):
    
    # I have run once
    # if k == 0:
    #     continue
    
    logging.info(f'====================== Fold {k+1} =====================')
    print(f'====================== Fold {k+1} =====================')
    if args.model == "Knot_KAN":
        model = Knots_KAN(layers_hidden=[in_size] + args.hidden_layer + [args.pred_len],
                            grid_size=args.grid_size,
                            spline_order=args.spline_order,
                            grid_range=args.grid_range,
                            groups=args.groups, 
                            need_relu=args.need_relu)

    elif args.model == 'KAN':
        model = KAN(layers_hidden=[in_size] + args.hidden_layer + [args.pred_len],
                    grid_size=args.grid_size,
                    spline_order=args.spline_order,
                    grid_range=args.grid_range)
        
    elif args.model == 'Rational_KAN':
        model = Rational_KAN(
                            layers_hidden=[in_size] + args.hidden_layer + [args.pred_len],
                            P_order=args.spline_order,
                            Q_order=args.spline_order,
                            groups=args.groups,
                            need_relu=args.need_relu
                            )
        
    elif args.model == 'RBF_KAN':
        model = RBF_KAN(layers_hidden=[in_size] + args.hidden_layer + [args.pred_len],
                        grid_size=args.grid_size,)
        
    elif args.model == 'Fourier_KAN':
        model = Fourier_KAN(layers_hidden=[in_size] + args.hidden_layer + [args.pred_len],
                            grid_size=args.grid_size)
        
    else:
        model = MLP(layers_hidden=[in_size] + args.hidden_layer + [args.pred_len])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.MSELoss()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'{model}')
    logging.info(f'{args}')
    logging.info(f'Number of Trainable Param: {trainable_params}')
    print(f'Number of Trainable Param: {trainable_params}')


    epochs = 10
    best_mse = np.inf
    best_mae = np.inf
    for epoch in range(epochs):
        # Train
        model.train()
        with tqdm(train_loader) as pbar:
            for i, (ts, labels) in enumerate(pbar):
                b_s = ts.shape[0]
                ts = ts.view(b_s, -1).to(device).to(torch.float32)
                optimizer.zero_grad()
                output = model(ts, normalize=True)
                output = output[:, -args.pred_len:]
                labels = labels[:, -args.pred_len:, :].to(device).squeeze(-1).to(torch.float32)
                loss = criterion(output, labels.to(device))
                
                if args.model == 'Knot_KAN' and args.smooth:
                    l2_norm = 0
                    for layer in model.layers[:-1]:
                        f_derive = torch.autograd.grad(loss, layer.spline_lin_c, create_graph=True)[0]
                        s_derive = torch.autograd.grad(f_derive.sum(), layer.spline_lin_c, create_graph=True)[0]
                        l2_norm += torch.norm(s_derive, p=2)
                    loss += args.smooth_lambda * l2_norm
                
                loss.backward()
                optimizer.step()
                mse = ((output - labels)**2).float().mean()
                pbar.set_postfix(loss=loss.item(), mse=mse.item(), lr=optimizer.param_groups[0]['lr'])

        # Validation
        model.eval()
        val_loss = []
        val_y_pred = []
        val_y_true = []
        with torch.no_grad():
            for ts, labels in val_loader:
                b_s = ts.shape[0]
                ts = ts.view(b_s, -1).to(device).to(torch.float32)
                output = model(ts, normalize=True)
                output = output[:, -args.pred_len:]
                labels = labels[:, -args.pred_len:, :].to(device).squeeze(-1).to(torch.float32)
                labels = labels.detach().cpu().numpy()
                output = output.detach().cpu().numpy()
                val_y_pred.append(output)
                val_y_true.append(labels)
                
        val_y_pred = np.concatenate(val_y_pred, axis=0)
        val_y_true = np.concatenate(val_y_true, axis=0)
        
        val_mse = mean_squared_error(val_y_true, val_y_pred)
        val_mae = median_absolute_error(val_y_true, val_y_pred)
        
        scheduler.step()
        
        print(
            f"Epoch {epoch + 1}, Val MSE: {val_mse}, Val MAE: {val_mae}"
        )
        
        logging.info(f"Epoch {epoch + 1}, Val MSE: {val_mse}, Val MAE: {val_mae}")
        
        if best_mse > val_mse:
            best_mse = val_mse
            best_mae = val_mae
            
    mean_val_mse.append(best_mse)
    mean_val_mae.append(best_mae)
    logging.info(f"=============================== Best Val MSE: {best_mse}, Best Val MAE: {best_mae} ===============================")


final_mse = np.mean(mean_val_mse)
final_mae = np.mean(mean_val_mae)

std_mse = np.std(mean_val_mse)
std_mae = np.std(mean_val_mae)
logging.info(f'\nMean MSE:{final_mse}\nMean MAE:{final_mae}')
logging.info(f'\nMean MSE:{std_mse}\nMean MAE:{std_mae}')