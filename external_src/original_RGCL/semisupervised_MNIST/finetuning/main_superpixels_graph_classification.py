import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader


from nets.superpixels_graph_classification.load_net import gnn_model, explainer
from data.data import LoadData
from train.train_superpixels_graph_classification import train_epoch, cl_train_epoch, evaluate_network


def cl_train_val_pipeline(MODEL_NAME, dataset, params, net_params, args):
    load_model = args.load_model
    DATASET_NAME = dataset.name

    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    device = net_params['device']

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print('-' * 40 + "Finetune Option" + '-' * 40)
    print("Data  Name:     [{}]".format(DATASET_NAME))
    print("Model Name:     [{}]".format(MODEL_NAME))
    print("Training Graphs:[{}]".format(len(trainset)))
    print("Valid Graphs:   [{}]".format(len(valset)))
    print("Test Graphs:    [{}]".format(len(testset)))
    print("Number Classes: [{}]".format(net_params['n_classes']))
    print("Learning rate:  [{}]".format(params['init_lr']))
    print('-' * 40 + "Contrastive Option" + '-' * 40)
    print("Load model:     [{}]".format(load_model))
    # print("Aug Type:       [{}]".format(aug_type_list[args.aug]))
    print("Aug Type:       [{}]".format(args.aug))
    print("Projection head:[{}]".format(args.head))
    print('-' * 100)

    model = gnn_model(MODEL_NAME, net_params)
    GNN_imp_estimator = explainer(net_params)
    if load_model:
        output_path = './contrastive_models_fi'
        save_model_dir0 = os.path.join(output_path, DATASET_NAME)
        save_model_dir1 = os.path.join(save_model_dir0, args.aug)

        if args.head:
            save_model_dir1 += "_head"
        else:
            save_model_dir1 += "_no_head"
        save_model_dir2 = os.path.join(save_model_dir1, MODEL_NAME)
        load_model_name = save_model_dir2 + args.model_file
        load_explainer_name = save_model_dir2 + args.explainer_file

        checkpoint = torch.load(load_model_name, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
        model.load_state_dict(state_dict)
        print('Success load pre-trained model!: [{}]'.format(load_model_name))

        checkpoint = torch.load(load_explainer_name, map_location=lambda storage, loc: storage)
        GNN_imp_estimator_dict = GNN_imp_estimator.state_dict()
        state_dict = {k: v for k, v in checkpoint.items() if k in GNN_imp_estimator_dict.keys()}
        GNN_imp_estimator.load_state_dict(state_dict)
        print('Success load pre-trained explainer!: [{}]'.format(load_explainer_name))
    else:
        print('No model load!: Test baseline! ')

    model = model.to(device)
    GNN_imp_estimator = GNN_imp_estimator.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'], verbose=True)

    # batching exception for Diffpool
    drop_last = True if MODEL_NAME == 'DiffPool' else False

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last,
                              collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                            collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                             collate_fn=dataset.collate)

    for epoch in range(params['epochs']):

        epoch_train_loss, epoch_train_acc, optimizer = cl_train_epoch(model, GNN_imp_estimator, optimizer, device,
                                                                   train_loader, epoch)
        epoch_val_loss, epoch_val_acc = evaluate_network(model, GNN_imp_estimator, device, val_loader, epoch)
        _, epoch_test_acc = evaluate_network(model, GNN_imp_estimator, device, test_loader, epoch)
        print('-' * 80)
        print("Epoch [{}]  Val Acc: [{:.4f}]".format(epoch + 1, epoch_val_acc))
        print("Epoch [{}]  Test Acc: [{:.4f}]".format(epoch + 1, epoch_test_acc))
        print('-' * 80)
        scheduler.step(epoch_val_loss)

        if optimizer.param_groups[0]['lr'] < params['min_lr']:
            print("\n!! LR EQUAL TO MIN LR SET.")
            break

    _, test_acc = evaluate_network(model, GNN_imp_estimator, device, test_loader, epoch)
    _, train_acc = evaluate_network(model, GNN_imp_estimator, device, train_loader, epoch)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, args):

    load_model = args.load_model
    # aug_type_list = ['drop_nodes', 'drop_add_edges', 'noise', 'mask', 'subgraph', 'new', 'random', 'random2']
    DATASET_NAME = dataset.name
    
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    device = net_params['device']
    
    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    

    print('-'*40 + "Finetune Option" + '-'*40)
    print("Data  Name:     [{}]".format(DATASET_NAME))
    print("Model Name:     [{}]".format(MODEL_NAME))
    print("Training Graphs:[{}]".format(len(trainset)))
    print("Valid Graphs:   [{}]".format(len(valset)))
    print("Test Graphs:    [{}]".format(len(testset)))
    print("Number Classes: [{}]".format(net_params['n_classes']))
    print("Learning rate:  [{}]".format(params['init_lr']))
    print('-'*40 + "Contrastive Option" + '-'*40)
    print("Load model:     [{}]".format(load_model))
    #print("Aug Type:       [{}]".format(aug_type_list[args.aug]))
    print("Aug Type:       [{}]".format(args.aug))
    print("Projection head:[{}]".format(args.head))
    print('-'*100)


    model = gnn_model(MODEL_NAME, net_params)
    GNN_imp_estimator = explainer(net_params)
    if load_model:
        output_path = './contrastive_models_fi'
        save_model_dir0 = os.path.join(output_path, DATASET_NAME)
        save_model_dir1 = os.path.join(save_model_dir0, args.aug)
       
        if args.head:
            save_model_dir1 += "_head"
        else:
            save_model_dir1 += "_no_head"
        save_model_dir2 = os.path.join(save_model_dir1, MODEL_NAME)
        load_model_name = save_model_dir2 + args.model_file
        load_explainer_name = save_model_dir2 + args.explainer_file

        checkpoint = torch.load(load_model_name, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
        model.load_state_dict(state_dict)
        print('Success load pre-trained model!: [{}]'.format(load_model_name))

        checkpoint = torch.load(load_explainer_name, map_location=lambda storage, loc: storage)
        GNN_imp_estimator_dict = GNN_imp_estimator.state_dict()
        state_dict = {k: v for k, v in checkpoint.items() if k in GNN_imp_estimator_dict.keys()}
        GNN_imp_estimator.load_state_dict(state_dict)
        print('Success load pre-trained explainer!: [{}]'.format(load_explainer_name))
    else:
        print('No model load!: Test baseline! ')

    model = model.to(device)
    GNN_imp_estimator = GNN_imp_estimator.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'], verbose=True)
    
    # batching exception for Diffpool
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

    for epoch in range(params['epochs']):

        epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, GNN_imp_estimator, optimizer, device,
                                                                   train_loader, epoch)
        epoch_val_loss, epoch_val_acc = evaluate_network(model, GNN_imp_estimator, device, val_loader, epoch)
        _, epoch_test_acc = evaluate_network(model, GNN_imp_estimator, device, test_loader, epoch)
        print('-'*80)
        print("Epoch [{}]  Val Acc: [{:.4f}]".format(epoch + 1, epoch_val_acc))
        print("Epoch [{}]  Test Acc: [{:.4f}]".format(epoch + 1, epoch_test_acc))  
        print('-'*80)
        scheduler.step(epoch_val_loss)

        if optimizer.param_groups[0]['lr'] < params['min_lr']:
            print("\n!! LR EQUAL TO MIN LR SET.")
            break
        
    _, test_acc = evaluate_network(model, GNN_imp_estimator, device, test_loader, epoch)
    _, train_acc = evaluate_network(model, GNN_imp_estimator, device, train_loader, epoch)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    
    
        
def main():    
    """
        USER CONTROLS
    """
    config_path = ['configs/superpixels_graph_classification_GCN_MNIST.json',
                   'configs/superpixels_graph_classification_GIN_MNIST.json',
                   'configs/superpixels_graph_classification_GAT_MNIST.json',

                   'configs/superpixels_graph_classification_GCN_CIFAR10.json',
                   'configs/superpixels_graph_classification_GIN_CIFAR10.json',
                   'configs/superpixels_graph_classification_GAT_CIFAR10.json']

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_model',  action='store_true',
                        default=True,   help='run type')
    parser.add_argument('--head', action='store_true', default=True,
                        help="use head or not")
    parser.add_argument('--aug', type=str, default='nn',
                        help="Please give a value for gpu id")
    parser.add_argument('--config', type=int, default=1,
                        help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--seed', default=0,
                        help="Please give a value for seed")
    parser.add_argument('--gpu_id', default=0,
                        help="Please give a value for gpu id")
    parser.add_argument('--model_file', type=str, default='/t0.3model_epoch_80.pkl',
                        help="model file path")
    parser.add_argument('--explainer_file', type=str, default='/t0.3explainer_epoch_80.pkl',
                        help="explainer file path")
    parser.add_argument('--init_lr', type=float, default=0.01,
                        help="Please give a value for init_lr")
    
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--graph_norm', help="Please give a value for graph_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', type=bool, default=True, help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    args = parser.parse_args()

    with open(config_path[args.config]) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']

    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.graph_norm is not None:
        net_params['graph_norm'] = True if args.graph_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
        
    # Superpixels
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)
    net_params['in_dim_edge'] = dataset.train[0][0].edata['feat'][0].size(0)
    num_classes = len(np.unique(np.array(dataset.train[:][1])))
    net_params['n_classes'] = num_classes

    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        max_num_nodes_train = max([dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))])
        max_num_nodes_test = max([dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))])
        max_num_node = max(max_num_nodes_train, max_num_nodes_test)
        net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']
    
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    cl_train_val_pipeline(MODEL_NAME, dataset, params, net_params, args)

    
"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


if __name__ == "__main__":
    main()  
