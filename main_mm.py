from efficient_kan.konts_kan import *
from src.efficient_kan.kan import KAN
from src.efficient_kan.rational_kan import Rational_KAN
from src.efficient_kan.rbf_kan import RBF_KAN


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
from datasets.load_dataset_mm import get_loader
import math
import random
import logging 
from sklearn.metrics import f1_score, roc_auc_score



fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

warnings.simplefilter(action='ignore', category=UserWarning)



parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model', type=str, default="KAN", help='[Knot_KAN, KAN, MLP]')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_path', type=str, default='/home/nathan/KAN_nathan/efficient-kan/data')
parser.add_argument('--data_name', type=str, default='MIMIC')
parser.add_argument('--task_name', type=str, default='Multimodal')
parser.add_argument('--lr', type=float, default=0.01)


# Model Parameters
parser.add_argument('--grid_size', type=int, default=20)
parser.add_argument('--groups', type=int, default=16)
parser.add_argument('--spline_order', type=int, default=3)
parser.add_argument('--smooth_lambda', type=float, default=0.001)
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
args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if args.data_name == 'MIMIC':
    train_loader, val_loader, num_tokens, txt_len, ts_len, n_vars_ts, n_classes = get_loader(args)
else:
    train_loader, val_loader, input_shape, n_classes, num_mod = get_loader(args)
    in_size = math.prod(input_shape) * num_mod

results_path = f'/home/nathan/KAN_nathan/efficient-kan/outputs/Multimodal/{args.model}/{args.data_name}'
if not os.path.exists(results_path):
    print(f"Path does not exist. Creating: {results_path}")
    os.makedirs(results_path)
logging.basicConfig(filename=f'{results_path}/hid_layer_{args.hidden_layer}_{args.data_name}_{args.model}.log', level=logging.INFO)
fold = 3
val_results = []



for k in range(fold):
    
    
    if args.data_name == 'AVMNIST':
        criterion = nn.CrossEntropyLoss()
    elif args.data_name == 'IMDB':
        im_in = 3*256*160
        au_in = 300
        x_1_encoder = MMEncoder(im_in, math.prod(input_shape)).to(device)
        x_2_encoder = MMEncoder(au_in, math.prod(input_shape)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        
        in_size = math.prod(input_shape) * 2
        
    elif args.data_name == 'MIMIC':
        ts_in = n_vars_ts * ts_len
        embedding_dim = 128
        txt_in = txt_len * embedding_dim
        txt_out = 4 * embedding_dim
        ts_out = 4 * embedding_dim
        in_size = ts_out + txt_out
        embedding_layer = nn.Embedding(num_tokens, embedding_dim).to(device)
        embedding_layer.weight.requires_grad = True
        x_1_encoder = MMEncoder(txt_in, txt_out).to(device)
        x_2_encoder = MMEncoder(ts_in, ts_out).to(device)
        criterion = nn.CrossEntropyLoss()
    
    
    logging.info(f'====================== Fold {k+1} =====================')
    print(f'====================== Fold {k+1} =====================')
    if args.model == "Knot_KAN":
        model = Knots_KAN(layers_hidden=[in_size] + args.hidden_layer + [n_classes],
                            grid_size=args.grid_size,
                            spline_order=args.spline_order,
                            grid_range=args.grid_range,
                            groups=args.groups, 
                            need_relu=args.need_relu)
        
    elif args.model == 'KAN':
        model = KAN(layers_hidden=[in_size] + args.hidden_layer + [n_classes],
                    grid_size=args.grid_size,
                    spline_order=args.spline_order,
                    grid_range=args.grid_range)
        
    elif args.model == 'Rational_KAN':
        model = Rational_KAN(
                            layers_hidden=[in_size] + args.hidden_layer + [n_classes],
                            P_order=args.spline_order,
                            Q_order=args.spline_order,
                            groups=args.groups,
                            need_relu=args.need_relu
                            )
        
    elif args.model == 'RBF_KAN':
        model = RBF_KAN(layers_hidden=[in_size] + args.hidden_layer + [n_classes],
                        grid_size=args.grid_size,
                        grid_range=args.grid_range)
        
    else:
        model = MLP(layers_hidden=[in_size] + args.hidden_layer + [n_classes])


    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'{model}')
    logging.info(f'{args}')
    logging.info(f'Number of Trainable Param: {trainable_params}')



    epochs = 20
    best_metric = 0
    for epoch in range(epochs):
        # Train
        model.train()
        with tqdm(train_loader) as pbar:
            for i, (x_1, x_2, labels) in enumerate(pbar):
                b_s = x_1.shape[0]
                
                if args.data_name == 'AVMNIST':
                    labels = labels.to(device)
                    b_s = x_1.shape[0]
                    x_1, x_2 = x_1.to(torch.float32), x_2.to(torch.float32)
                    x_1 = x_1.view(b_s, -1).to(device)
                    x_2 = x_2.view(b_s, -1).to(device)
                    
                elif args.data_name == 'IMDB':
                    x_1_encoder.train()
                    x_2_encoder.train()
                    labels = labels.to(torch.float32)
                    b_s = x_1.shape[0]
                    labels = labels.to(device)
                    x_1, x_2 = x_1.to(torch.float32), x_2.to(torch.float32)
                    x_1 = x_1.view(b_s, -1).to(device)
                    x_2 = x_2.view(b_s, -1).to(device)
                    x_1 = x_1_encoder(x_1)
                    x_2 = x_2_encoder(x_2)
                    
                elif args.data_name == 'MIMIC':
                    x_1_encoder.train()
                    x_2_encoder.train()
                    embedding_layer.train()
                    
                    # x_1 --> text_list
                    # x_2 --> ts_list
                    # labels --> labels
                    labels = labels.to(device)
                    x_1 = embedding_layer.weight[x_1, :]
                    x_1 = x_1.view(b_s, -1).to(device)
                    x_2 = x_2.view(b_s, -1).to(device)
                    x_1, x_2 = x_1.to(torch.float32), x_2.to(torch.float32)
                    x_1 = x_1_encoder(x_1)
                    x_2 = x_2_encoder(x_2)
                    
                x = torch.concat([x_1, x_2], dim=1)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, labels)
                
                if args.data_name == 'IMDB':
                    output = torch.sigmoid(output)
                    pred = (output > 0.5).int().detach().cpu().numpy()
                    labels = labels.int().detach().cpu().numpy()
                    train_metric = f1_score(labels, pred, average='macro')
                else:
                    train_metric = (output.argmax(dim=1) == labels.to(device)).float().mean()
                    
                if args.model == 'Knot_KAN' and args.smooth:
                    l2_norm = 0
                    for layer in model.layers[:-1]:
                        f_derive = torch.autograd.grad(loss, layer.spline_lin_c, create_graph=True)[0]
                        s_derive = torch.autograd.grad(f_derive.sum(), layer.spline_lin_c, create_graph=True)[0]
                        l2_norm += torch.norm(s_derive, p=2)
                    loss += args.smooth_lambda * l2_norm
                    
                    
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=loss.item(), accuracy=train_metric.item(), lr=optimizer.param_groups[0]['lr'])

        # Validation
        model.eval()
        val_loss = 0
        val_metric = 0

        with torch.no_grad():
            for x_1, x_2, labels in val_loader:
                b_s = x_1.shape[0]
                
                if args.data_name == 'AVMNIST':
                    labels = labels.to(device)
                    b_s = x_1.shape[0]
                    x_1, x_2 = x_1.to(torch.float32), x_2.to(torch.float32)
                    x_1 = x_1.view(b_s, -1).to(device)
                    x_2 = x_2.view(b_s, -1).to(device)
                    
                    
                    
                elif args.data_name == 'IMDB':
                    x_1_encoder.train()
                    x_2_encoder.train()
                    labels = labels.to(torch.float32)
                    b_s = x_1.shape[0]
                    x_1, x_2 = x_1.to(torch.float32), x_2.to(torch.float32)
                    x_1 = x_1.view(b_s, -1).to(device)
                    x_2 = x_2.view(b_s, -1).to(device)
                    x_1 = x_1_encoder(x_1)
                    x_2 = x_2_encoder(x_2)
                    
                    
                    
                elif args.data_name == 'MIMIC':
                    x_1_encoder.train()
                    x_2_encoder.train()
                    embedding_layer.train()
                    
                    # x_1 --> text_list
                    # x_2 --> ts_list
                    # labels --> labels
                    labels = labels.to(device)
                    x_1 = embedding_layer.weight[x_1, :]
                    x_1, x_2 = x_1.to(torch.float32), x_2.to(torch.float32)
                    x_1 = x_1.view(b_s, -1).to(device)
                    x_2 = x_2.view(b_s, -1).to(device)
                    x_1 = x_1_encoder(x_1)
                    x_2 = x_2_encoder(x_2)
                    
                    
                    
                x = torch.concat([x_1, x_2], dim=1)
                output = model(x)
                val_loss += criterion(output, labels).item()
                
                
                if args.data_name == 'IMDB':
                    output = torch.sigmoid(output)
                    pred = (output > 0.5).int().detach().cpu().numpy()
                    labels = labels.int().detach().cpu().numpy()
                    val_metric += f1_score(labels, pred, average='macro')
                    
                elif args.data_name == 'AVMNIST':
                    val_metric += (output.argmax(dim=1) == labels.to(device)).float().mean()
                    
                elif args.data_name == 'MIMIC':
                    pred = nn.functional.softmax(output, dim=1)
                    max_probs, pred_labels = torch.max(pred, dim=1)
                    # val_metric += roc_auc_score(labels.detach().cpu().numpy(), max_probs.detach().cpu().numpy(), average='macro')
                    val_metric += f1_score(labels.detach().cpu().numpy(), pred_labels.detach().cpu().numpy(), average='macro')
                    
                    
        val_loss /= len(val_loader)
        val_metric /= len(val_loader)
        
        
        if best_metric < val_metric:
            best_metric = val_metric.item()
            
        if args.data_name == 'IMDB':
            logging.info(f'val F1:{val_metric}, val loss:{val_loss}')
        else:
            logging.info(f'val Acc:{val_metric}, val loss:{val_loss}')
            
            
            
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Metric: {val_metric}")
    
    
    
    val_results.append(best_metric)
    logging.info(f'=========================================== Best val metric:{best_metric} ===========================================')

mean_metric = np.mean(val_results)
std_metric = np.std(val_results)
logging.info(f'\nMean acc:{mean_metric}\nstd_metric:{std_metric}')