import argparse
import pandas as pd
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from fastai.callbacks import SaveModelCallback
from fastai.basic_data import DataBunch, DatasetType
from fastai.basic_train import Learner
from fastai.train import *
from fastai.distributed import *

from moldataset import MoleculeDataset, collate_parallel_fn
from model import Graph_Transformer
from utils import scale_features, set_seed, store_submit, store_oof
from callbacks import GradientClipping, GroupMeanLogMAE
from losses_and_metrics import rmse, mae, contribs_rmse_loss
import constants as C
import os
import pdb

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from fcnet import FullyConnectedNet, hidden_layer
from scatter import scatter_mean
from layernorm import LayerNorm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.006, help='learning rate')
parser.add_argument('--d_model', type=int, default=256, help='dimenstion of node state vector')
parser.add_argument('--N', type=int, default=2, help='number of encoding layers')
parser.add_argument('--h', type=int, default=4, help='number of attention heads')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--fold_id', type=int, default=1)
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

model_str = f'mol_transformer_v{args.version}_fold{args.fold_id}'

train_df = pd.read_csv(C.PROC_DATA_PATH + 'train_proc_df.csv', index_col=0)   # train_df: (4298749, 32)
test_df = pd.read_csv(C.PROC_DATA_PATH + 'test_proc_df.csv', index_col=0)     # test_df: (359398, 32)
test_df.drop(columns='scalar_coupling_constant', inplace=True)                # test_df: (359398, 31)

atom_df = pd.read_csv(C.PROC_DATA_PATH + 'atom_df.csv', index_col=0)          # atom_df: (1533537, 20)
bond_df = pd.read_csv(C.PROC_DATA_PATH + 'bond_df.csv', index_col=0)          # bond_df: (1586335, 11)
angle_in_df = pd.read_csv(C.PROC_DATA_PATH + 'angle_in_df.csv', index_col=0)  # angle_in_df: (4141741, 3)
angle_out_df = pd.read_csv(C.PROC_DATA_PATH + 'angle_out_df.csv', index_col=0)   # angle_out_df: (1798001, 3)
graph_dist_df = pd.read_csv(C.PROC_DATA_PATH + 'graph_dist_df.csv', index_col=0, dtype=np.int32)
                                                                                 # graph_dist_df: (1533537, 30)

structures_df = pd.read_csv(C.PROC_DATA_PATH + 'structures_proc_df.csv', index_col=0)   # structures_df: (1533537, 7)
train_mol_ids = pd.read_csv(C.PROC_DATA_PATH + 'train_idxs_8_fold_cv.csv',
                            usecols=[0, args.fold_id], index_col=0).dropna().astype(int).iloc[:, 0]
                                                                              # train_mol_ids: (66940,)
val_mol_ids = pd.read_csv(C.PROC_DATA_PATH + 'val_idxs_8_fold_cv.csv',
                          usecols=[0, args.fold_id], index_col=0).dropna().astype(int).iloc[:, 0]
                                                                              # val_mol_ids: (9563,)
test_proc_df = pd.read_csv(C.PROC_DATA_PATH + 'test_proc_df.csv')         # test_proc_df: (359398, 33)
test_mol_ids = pd.Series(test_proc_df['molecule_id'].unique())            # test_mol_ids: (8500,)

train_df, sc_feat_means, sc_feat_stds = scale_features(
    train_df, C.SC_FEATS_TO_SCALE, train_mol_ids, return_mean_and_std=True)    # train_df: (4298749, 32)
test_df = scale_features(test_df, C.SC_FEATS_TO_SCALE, test_mol_ids)           # test_df: (359398, 31)
atom_df = scale_features(atom_df, C.ATOM_FEATS_TO_SCALE, train_mol_ids)        # atom_df: (1533537, 20)
bond_df = scale_features(bond_df, C.BOND_FEATS_TO_SCALE, train_mol_ids)        # bond_df: (1586335, 11)

gb_mol_sc = train_df.groupby('molecule_id')                              # gb_mol_sc: len(76503)
gb_mol_sc_test = test_df.groupby('molecule_id')                          # gb_mol_sc_test: len(8500)
gb_mol_atom = atom_df.groupby('molecule_id')                             # gb_mol_atom:
gb_mol_bond = bond_df.groupby('molecule_id')
gb_mol_struct = structures_df.groupby('molecule_id')
gb_mol_angle_in = angle_in_df.groupby('molecule_id')                     # gb_mol_angle_in: 85003
gb_mol_angle_out = angle_out_df.groupby('molecule_id')                   # gb_mol_angle_out: 84999
gb_mol_graph_dist = graph_dist_df.groupby('molecule_id')
set_seed(100)

train_ds = MoleculeDataset(train_mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_bond, gb_mol_struct,
    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist)                # train_ds: 66940
val_ds = MoleculeDataset(val_mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_bond, gb_mol_struct,
    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist)                # val_ds: 9563
test_ds = MoleculeDataset(test_mol_ids, gb_mol_sc_test, gb_mol_atom, gb_mol_bond, gb_mol_struct,
    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist)                # test_ds: 8500

train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=64, pin_memory=True, drop_last=True)
                                                                         # train_dl: 66940
val_dl = DataLoader(val_ds, args.batch_size, num_workers=64, pin_memory=True, drop_last=True)   # val_dl: 9563
test_dl = DataLoader(test_ds, args.batch_size, num_workers=64, pin_memory=True)                 # test_dl: 8500
db = DataBunch(train_dl, val_dl, test_dl=test_dl, collate_fn=collate_parallel_fn, device=device)

set_seed(100)
d_model = args.d_model  # d_model: 256
enn_args = dict(layers=1 * [d_model], dropout=1 * [0.0], layer_norm=True)
#ann_args = None
ann_args = dict(layers=1 * [d_model], dropout=1 * [0.0], layer_norm=True, out_act=nn.Tanh())
model = Graph_Transformer(
    C.N_ATOM_FEATURES, C.N_BOND_FEATURES, C.N_SC_EDGE_FEATURES,
    C.N_SC_MOL_FEATURES, N=args.N, d_model=d_model, d_ff=d_model * 4,
    d_ff_contrib=d_model // 4, h=args.h, dropout=args.dropout,
    kernel_sz=min(128, d_model), enn_args=enn_args, ann_args=ann_args)

model = nn.DataParallel(model.cuda(), device_ids=[0])

callback_fns = [partial(GradientClipping, clip=10), GroupMeanLogMAE,
    partial(SaveModelCallback, every='improvement', mode='min', monitor='group_mean_log_mae', name=model_str)]

learn = Learner(db, model, metrics=[rmse, mae], callback_fns=callback_fns, wd=args.wd, loss_func=contribs_rmse_loss)
if args.start_epoch > 0:
    learn.load(model_str)
    torch.cuda.empty_cache()

learn.fit_one_cycle(args.epochs, max_lr=args.lr, start_epoch=args.start_epoch)

val_contrib_preds = learn.get_preds(DatasetType.Valid)
val_preds = val_contrib_preds[0][:, -1].detach().numpy() * C.SC_STD + C.SC_MEAN
store_oof(val_preds, model_str, print_head=True)

'''w = torch.load("/home/nesa/fangjia/kaggle-champs-master--0/ax.pt")
metrics = torch.load("/home/nesa/fangjia/kaggle-champs-master--0/metrics.pt")'''
'''def learning_curve_loss(X, Y):
    plt.figure()
    plt.plot(X, Y, 'o-', color='r', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim = (0.6, 1.01)
    plt.title('XGBoost')
    plt.savefig('./model_loss.png')
learning_curve_loss(list(range(args.epochs)), metrics)'''

# make predictions
test_contrib_preds = learn.get_preds(DatasetType.Test)
test_preds = test_contrib_preds[0][:, -1].detach().numpy() * C.SC_STD + C.SC_MEAN

test_contrib_preds_df = pd.DataFrame(list(test_contrib_preds[0]), columns=['scalar_coupling_constant'])
test_contrib_preds_df['scalar_coupling_constant'] = (test_contrib_preds_df['scalar_coupling_constant']) * C.SC_STD + C.SC_MEAN
test_targs = test_df['scalar_coupling_constant']

test_contrib_preds_array = np.array(test_contrib_preds_df, dtype=np.float)
test_targs_array = np.array(test_targs)
test_contrib_preds_tensor = torch.Tensor(test_contrib_preds_array)
test_targs_tensor = torch.Tensor(test_targs_array)

def contribs_rmse_loss(preds, targs):
    return torch.mean((preds - targs) ** 2, dim=0).sqrt().sum()
Loss = contribs_rmse_loss(test_contrib_preds_tensor, test_targs_tensor)
print(Loss)
