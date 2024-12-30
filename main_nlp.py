from src.efficient_kan.group_kan import *
from src.efficient_kan.kan import KAN
from src.efficient_kan.rational_kan import Rational_KAN


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
from datasets.load_dataset_nlp import get_loader
import math
import random
import logging 
import numpy as np



# For results reproduction
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

warnings.simplefilter(action='ignore', category=UserWarning)



parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model', type=str, default="Knot_KAN", help='[Knot_KAN, KAN, MLP, Rational_KAN]')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_path', type=str, default='/home/nathan/KAN_nathan/efficient-kan/data')
parser.add_argument('--data_name', type=str, default='AG_NEWS')
parser.add_argument('--task_name', type=str, default='NLP')
parser.add_argument('--lr', type=float, default=0.001)

# Model Parameters
parser.add_argument('--grid_size', type=int, default=20)
parser.add_argument('--groups', type=int, default=16)
parser.add_argument('--spline_order', type=int, default=3)
parser.add_argument('--smooth_lambda', type=float, default=0.01)
parser.add_argument('--activation', type=str, default='SiLU')


### Testing deguging 
# parser.add_argument('--grid_range', type=list, default=[-10, 10])
# parser.add_argument('--hidden_layer', type=list, default=[64])
# parser.add_argument('--smooth', default=False)
# parser.add_argument('--need_relu', default=False)


### Runing on scripts
parser.add_argument('--hidden_layer', type=int, nargs='*', help='Arbitary list of hidden layer')
parser.add_argument('--smooth',  action='store_true')
parser.add_argument('--need_relu',  action='store_true')
parser.add_argument('--grid_range', type=int, nargs=2, help='Default as [-1, 1]')
args = parser.parse_args()



# train_dataloader, test_dataloader, ipnut_shape, n_classes
train_loader, val_loader, max_length, num_tokens, embedding_dim, n_classes = get_loader(args)
in_size = max_length * embedding_dim
epochs = 20


results_path = f'/home/nathan/KAN_nathan/efficient-kan/outputs/NLP_output/{args.model}/{args.data_name}'


if not os.path.exists(results_path):
    print(f"Path does not exist. Creating: {results_path}")
    os.makedirs(results_path)


logging.basicConfig(filename=f'{results_path}/hid_layer_{args.hidden_layer}_{args.data_name}_{args.model}.log', level=logging.INFO)

fold = 3
val_results = []

for k in range(fold):
    
    # I have run once
    # if k == 0:
    #     continue
    
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
    else:
        model = MLP(layers_hidden=[in_size] + args.hidden_layer + [n_classes])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    
    # Embeeding layer
    embedding_layer = nn.Embedding(num_tokens, embedding_dim).to(device)
    embedding_layer.weight.requires_grad = True
    
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'{model}')
    logging.info(f'{args}')
    logging.info(f'Number of Trainable Param: {trainable_params}')
    print(f'Number of Trainable Param: {trainable_params}')


    best_acc = 0
    for epoch in range(epochs):
        # Train
        model.train()
        embedding_layer.train()
        with tqdm(train_loader) as pbar:
            for i, (text, labels) in enumerate(pbar):
                text_embed = embedding_layer.weight[text, :]
                b_s = text_embed.shape[0]
                text_embed = text_embed.view(b_s, -1).to(device)
                optimizer.zero_grad()
                output = model(text_embed)
                loss = criterion(output, labels.to(device))
                
                if args.model == 'Knot_KAN' and args.smooth:
                    l2_norm = 0
                    for layer in model.layers[:-1]:
                        f_derive = torch.autograd.grad(loss, layer.spline_lin_c, create_graph=True)[0]
                        s_derive = torch.autograd.grad(f_derive.sum(), layer.spline_lin_c, create_graph=True)[0]
                        l2_norm += torch.norm(s_derive, p=2)
                    loss += args.smooth_lambda * l2_norm
                    
                loss.backward()
                max_norm = 1.0  # Maximum allowed norm of gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
                
                if torch.isnan(loss):
                    break
        
        
        #################### Validation ####################
        model.eval()
        embedding_layer.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for text, labels in val_loader:
                text_embed = embedding_layer.weight[text, :]
                b_s = text_embed.shape[0]
                text_embed = text_embed.view(b_s, -1).to(device)
                output = model(text_embed)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
                
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        
        if best_acc < val_accuracy:
            best_acc = val_accuracy
            # torch.save(model, '../cv_model_mnist.pth')
            
        logging.info(f'val acc:{val_accuracy}, val loss:{val_loss}')
        # Update learning rate
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )
        
    val_results.append(best_acc)
    logging.info(f'=========================================== Best val acc:{best_acc} ===========================================')


# logging.info(f'Fold {k} Best val acc:{best_acc}')
# lst_metric.append(best_acc)
mean_metric = np.mean(val_results)
std_metric = np.std(val_results)
logging.info(f'\nMean acc:{mean_metric}\nstd_metric:{std_metric}')