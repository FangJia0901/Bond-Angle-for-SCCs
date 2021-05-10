import argparse
import pandas as pd
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from fastai.callbacks import SaveModelCallback
from fastai.basic_data import DataBunch, DeviceDataLoader, DatasetType
from fastai.basic_train import Learner
from fastai.train import *
from fastai.distributed import *

#from moldataset import MoleculeDataset, collate_parallel_fn
from model import Transformer
from utils import scale_features, set_seed, store_submit, store_oof
from callbacks import GradientClipping, GroupMeanLogMAE
from losses_and_metrics import rmse, mae, contribs_rmse_loss
import constants as C
import os

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from fcnet import FullyConnectedNet, hidden_layer
from scatter import scatter_mean
from layernorm import LayerNorm

device = torch.device("cuda:3")

import numpy as np
import torch
from torch.utils.data import Dataset
import constants as C


def _get_existing_group(gb, i):
    group_df = gb.get_group(i)
    return group_df


def get_dist_matrix(struct_df):
    locs = struct_df[['x', 'y', 'z']].values
    n_atoms = len(locs)
    loc_tile = np.tile(locs.T, (n_atoms, 1, 1))
    dist_mat = np.sqrt(((loc_tile - loc_tile.T) ** 2).sum(axis=1))
    return dist_mat


class MoleculeDataset(Dataset):
    """Dataset returning inputs and targets per molecule."""

    def __init__(self, mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_bond,
                 gb_mol_struct, gb_mol_angle_in, gb_mol_angle_out,
                 gb_mol_graph_dist):
        """Dataset is constructed from dataframes grouped by molecule_id."""
        self.n = len(mol_ids)
        self.mol_ids = mol_ids
        self.gb_mol_sc = gb_mol_sc
        self.gb_mol_atom = gb_mol_atom
        self.gb_mol_bond = gb_mol_bond
        self.gb_mol_struct = gb_mol_struct
        self.gb_mol_angle_in = gb_mol_angle_in
        self.gb_mol_angle_out = gb_mol_angle_out
        self.gb_mol_graph_dist = gb_mol_graph_dist

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (self.gb_mol_sc.get_group(self.mol_ids[idx]),
                self.gb_mol_atom.get_group(self.mol_ids[idx]),
                self.gb_mol_bond.get_group(self.mol_ids[idx]),
                self.gb_mol_struct.get_group(self.mol_ids[idx]),
                self.gb_mol_angle_in.get_group(self.mol_ids[idx]),
                _get_existing_group(self.gb_mol_angle_out, self.mol_ids[idx]),
                self.gb_mol_graph_dist.get_group(self.mol_ids[idx]))


def arr_lst_to_padded_batch(arr_lst, dtype=torch.float,
                            pad_val=C.BATCH_PAD_VAL):
    tensor_list = [torch.Tensor(arr).type(dtype) for arr in arr_lst]
    batch = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=pad_val)
    return batch.contiguous()


def collate_parallel_fn(batch, test=False):
    """
    Transforms input dataframes grouped by molecule into a batch of input and 
    target tensors for a 'batch_size' number of molecules. The first dimension 
    is used as the batch dimension.

    Returns:
        - atom_x: features at the atom level
        - bond_x: features at the chemical bond level
        - sc_x: features describing the scalar coupling atom_0 and atom_1 pairs
        - sc_m_x: in addition to the set of features in 'sc_x', includes 
            features at the molecule level.
        - eucl_dists: 3D euclidean distance matrices
        - graph_dists: graph distance matrices
        - angles: cosine angles between all chemical bonds
        - mask: binary mask of dim=(batch_size, max_n_atoms, max_n_atoms),
            where max_n_atoms is the largest number of atoms per molecule in 
            'batch'
        - bond_idx: tensor of dim=(batch_size, max_n_bonds, 2), containing the
            indices of atom_0 and atom_1 pairs that form chemical bonds
        - sc_idx: tensor of dim=(batch_size, max_n_sc, 2), containing the
            indices of atom_0 and atom_1 pairs that form a scalar coupling
            pair
        - angles_idx: tensor of dim=(batch_size, max_n_angles, 1), mapping 
            angles to the chemical bonds in the molecule.
        - sc_types: scalar coupling types
        - sc_vals: scalar coupling contributions (first 4 columns) and constant
            (last column)
    """
    batch_size, n_atom_sum, n_pairs_sum = len(batch), 0, 0
    atom_x, bond_x, sc_x, sc_m_x = [], [], [], []
    eucl_dists, graph_dists = [], []
    angles_in, angles_out = [], []
    mask, bond_idx, sc_idx = [], [], []
    angles_in_idx, angles_out_idx = [], []
    sc_types, sc_vals = [], []

    for b in range(batch_size):
        (sc_df, atom_df, bond_df, struct_df, angle_in_df, angle_out_df,
         graph_dist_df) = batch[b]
        n_atoms, n_pairs, n_sc = len(atom_df), len(bond_df), len(sc_df)
        n_pad = C.MAX_N_ATOMS - n_atoms
        eucl_dists_ = get_dist_matrix(struct_df)
        eucl_dists_ = np.pad(eucl_dists_, [(0, 0), (0, n_pad)], 'constant',
                             constant_values=999)

        atom_x.append(atom_df[C.ATOM_FEATS].values)
        bond_x.append(bond_df[C.BOND_FEATS].values)
        sc_x.append(sc_df[C.SC_EDGE_FEATS].values)
        sc_m_x.append(sc_df[C.SC_MOL_FEATS].values)
        sc_types.append(sc_df['type'].values)
        if not test:
            n_sc_pad = C.MAX_N_SC - n_sc
            sc_vals_ = sc_df[C.CONTRIB_COLS + [C.TARGET_COL]].values
            sc_vals.append(np.pad(sc_vals_, [(0, n_sc_pad), (0, 0)], 'constant',
                                  constant_values=-999))
        eucl_dists.append(eucl_dists_)
        graph_dists.append(graph_dist_df.values[:, :-1])
        angles_in.append(angle_in_df['cos_angle'].values)
        if angle_out_df is not None:
            angles_out.append(angle_out_df['cos_angle'].values)
        else:
            angles_out.append(np.array([C.BATCH_PAD_VAL]))

        mask.append(np.pad(np.ones(2 * [n_atoms]), [(0, 0), (0, n_pad)],
                           'constant'))
        bond_idx.append(bond_df[['idx_0', 'idx_1']].values)
        sc_idx.append(sc_df[['atom_index_0', 'atom_index_1']].values)
        angles_in_idx.append(angle_in_df['b_idx'].values)
        if angle_out_df is not None:
            angles_out_idx.append(angle_out_df['b_idx'].values)
        else:
            angles_out_idx.append(np.array([0.]))

        n_atom_sum += n_atoms
        n_pairs_sum += n_pairs

    atom_x = arr_lst_to_padded_batch(atom_x, pad_val=0.)
    bond_x = arr_lst_to_padded_batch(bond_x)
    max_n_atoms = atom_x.size(1)
    max_n_bonds = bond_x.size(1)
    angles_out_idx = [a + max_n_bonds for a in angles_out_idx]

    sc_x = arr_lst_to_padded_batch(sc_x)
    sc_m_x = arr_lst_to_padded_batch(sc_m_x)
    if not test:
        sc_vals = arr_lst_to_padded_batch(sc_vals)
    else:
        sc_vals = torch.tensor([0.] * batch_size)
    sc_types = arr_lst_to_padded_batch(sc_types, torch.long)
    mask = arr_lst_to_padded_batch(mask, torch.uint8, 0)
    mask = mask[:, :, :max_n_atoms].contiguous()
    bond_idx = arr_lst_to_padded_batch(bond_idx, torch.long, 0)
    sc_idx = arr_lst_to_padded_batch(sc_idx, torch.long, 0)
    angles_in_idx = arr_lst_to_padded_batch(angles_in_idx, torch.long, 0)
    angles_out_idx = arr_lst_to_padded_batch(angles_out_idx, torch.long, 0)
    angles_idx = torch.cat((angles_in_idx, angles_out_idx), dim=-1).contiguous()
    eucl_dists = arr_lst_to_padded_batch(eucl_dists, pad_val=999)
    eucl_dists = eucl_dists[:, :, :max_n_atoms].contiguous()
    graph_dists = arr_lst_to_padded_batch(graph_dists, torch.long, 10)
    graph_dists = graph_dists[:, :, :max_n_atoms].contiguous()
    angles_in = arr_lst_to_padded_batch(angles_in)
    angles_out = arr_lst_to_padded_batch(angles_out)
    angles = torch.cat((angles_in, angles_out), dim=-1).contiguous()

    return (atom_x, bond_x, sc_x, sc_m_x, eucl_dists, graph_dists, angles, mask,
            bond_idx, sc_idx, angles_idx, sc_types), sc_vals


def clones(module, N):
    """Produce N identical layers."""
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


def _gather_nodes(x, idx, sz_last_dim):                   # x: 1*14*64, idx: 1*28, sz_last_dim: 64
    idx = idx.unsqueeze(-1).expand(-1, -1, sz_last_dim)   # idx: 1*28*64
    return x.gather(1, idx)                               # return: 1*28*64


class ENNMessage(nn.Module):
    """
    The edge network message passing function from the MPNN paper.Optionally
    adds and additional cosine angle based attention mechanism over incoming
    messages.
    """
    PAD_VAL = -999

    def __init__(self, d_model, d_edge, kernel_sz, enn_args={}, ann_args=None):
        super().__init__()
        assert kernel_sz <= d_model
        self.d_model, self.kernel_sz = d_model, kernel_sz
        self.enn = FullyConnectedNet(d_edge, d_model * kernel_sz, **enn_args)
        if ann_args:
            self.ann = FullyConnectedNet(1, d_model, **ann_args)
        else:
            self.ann = None

    def forward(self, x, edges, pairs_idx, angles=None, angles_idx=None, t=0):
        """Note that edges and pairs_idx raw inputs are for a unidirectional
        graph. They are expanded to allow bidirectional message passing."""
        if t == 0:
            self.set_a_mat(edges)                       # edges: 1*14*8
            if self.ann: self.set_attn(angles)          # angles: 1*44
            # concat reversed pairs_idx for bidirectional message passing
            self.pairs_idx = torch.cat([pairs_idx, pairs_idx[:, :, [1, 0]]], dim=1) # pairs_idx: 1*14*2 --> self.pairs_idx: 1*28*2
        return self.add_message(torch.zeros_like(x), x, angles_idx)         # torch.zeros_like(x): 1*14*64

    def set_a_mat(self, edges):          # edges: 1*14*8  d_edge: 8
        n_edges = edges.size(1)          # n_edges: 14
        a_vect = self.enn(edges)         # d_model: 64;  kernel_sz: 64  a_vect: 1*14*4096
        a_vect = a_vect / (self.kernel_sz ** .5)  # rescale
        mask = edges[:, :, 0, None].expand(a_vect.size()) == self.PAD_VAL   # mask: 1*14*4096
        a_vect = a_vect.masked_fill(mask, 0.0)   # a_vect: 1*14*4096
        self.a_mat = a_vect.view(-1, n_edges, self.d_model, self.kernel_sz)    # a_mask: 1*14*64*64
        # concat a_mats for bidirectional message passing
        self.a_mat = torch.cat([self.a_mat, self.a_mat], dim=1)       # a_mask: 1*28*64*64

    def set_attn(self, angles):                                  # angles: 1*44
        angles = angles.unsqueeze(-1)                            # angles: 1*44*1
        self.attn = self.ann(angles)                             # self.attn: 1*44*64, d_model: 64
        mask = angles.expand(self.attn.size()) == self.PAD_VAL   # mask: 1*44*64
        self.attn = self.attn.masked_fill(mask, 0.0)             # self.attn: 1*44*64

    def add_message(self, m, x, angles_idx=None):
        """Add message for atom_{i}: m_{i} += sum_{j}[attn_{ij} A_{ij}x_{j}]."""
        # select the 'x_{j}' feeding into the 'm_{i}'
        x_in = _gather_nodes(x, self.pairs_idx[:, :, 1], self.d_model)       # self.pairs_idx: 1*28*2 ; x_in: 1*28*64
                                                                             # self.d_model: 64
        # do the matrix multiplication 'A_{ij}x_{j}'
        if self.kernel_sz == self.d_model:  # full matrix multiplcation      # self.kernel_sz: 64
            ax = (x_in.unsqueeze(-2) @ self.a_mat).squeeze(-2)               # self.a_mat: 1*28*64*64  ax: 1*28*64
        else:  # do a convolution
            x_padded = F.pad(x_in, self.n_pad)
            x_unfolded = x_padded.unfold(-1, self.kernel_sz, 1)
            ax = (x_unfolded * self.a_mat).sum(-1)

        # apply atttention
        if self.ann:
            n_pairs = self.pairs_idx.size(1)                            # n_pairs: 28
            # average all attn(angle_{ijk}) per edge_{ij}.
            # i.e.: attn_{ij} = sum_{k}[attn(angle_{ijk})] / n_angles_{ij}
            ave_att = scatter_mean(self.attn, angles_idx, num=n_pairs, dim=1,
                                   out=torch.ones_like(ax))            # angles_idx: 44;   self.attn: 1*44*64
            ax = ave_att * ax                                          # ax: 1*28*64       ax, ave_att: 1*28*64

        # sum up all 'A_{ij}h_{j}' per node 'i'
        idx_0 = self.pairs_idx[:, :, 0, None].expand(-1, -1, self.d_model)   # idx_0: 1*28*64
        return m.scatter_add(1, idx_0, ax)                  # m=torch.zeros_like(x): 1*14*64;
                                                            # return: 1*14*64
    @property
    def n_pad(self):
        k = self.kernel_sz
        return (k // 2, k // 2 - int(k % 2 == 0))


class MultiHeadedDistAttention(nn.Module):
    """Generalizes the euclidean and graph distance based attention layers."""

    def __init__(self, h, d_model):   # h: 1, d_model: 64
        super().__init__()
        self.d_model, self.d_k, self.h = d_model, d_model // h, h    # self.d_model: 64, self.d_k: 64, self.h: 1
        self.attn = None
        self.linears = clones(nn.Linear(d_model, d_model), 2)

    def forward(self, dists, x, mask):                     # dists: 1*14*14,  x: 1*14*64,  mask: 1*14*14
        batch_size = x.size(0)                             # batch_size: 1
        x = self.linears[0](x).view(batch_size, -1, self.h, self.d_k)   # x: 1*14*64 --> 1*14*1*64
        x, self.attn = self.apply_attn(dists, x, mask)                  # x: 1*14*1*64,  self.attn: 1*1*14*14
        x = x.view(batch_size, -1, self.h * self.d_k)                   # x: 1*14*64
        return self.linears[-1](x)                                      # x: 1*14*64

    def apply_attn(self, dists, x, mask):                    # dists: 1*14*14, x: 1*14*1*64, mask: 1*14*14
        attn = self.create_raw_attn(dists, mask)             # attn: 1*14*14*1
        attn = attn.transpose(-2, -1).transpose(1, 2)        # attn: 1*1*14*14
        x = x.transpose(1, 2)                                # x: 1*1*14*64
        x = torch.matmul(attn, x)                            # x: 1*1*14*64
        x = x.transpose(1, 2).contiguous()                   # x: 1*14*1*64
        return x, attn                                       # x: 1*14*1*64,  attn: 1*1*14*14

    def create_raw_attn(self, dists, mask):
        pass


class MultiHeadedGraphDistAttention(MultiHeadedDistAttention):
    """Attention based on an embedding of the graph distance matrix."""
    MAX_GRAPH_DIST = 10

    def __init__(self, h, d_model):
        super().__init__(h, d_model)                                # h: 1,  d_model: 64
        self.embedding = nn.Embedding(self.MAX_GRAPH_DIST + 1, h)   #

    def create_raw_attn(self, dists, mask):                         # dists: 1*14*14,  mask: 1*14*14
        emb_dists = self.embedding(dists)                           # emb_dists: 1*14*14*1
        mask = mask.unsqueeze(-1).expand(emb_dists.size())          # mask: 1*14*14*1
        emb_dists = emb_dists.masked_fill(mask == 0, -1e9)          # emb_dists: 1*14*14*1
        return F.softmax(emb_dists, dim=-2).masked_fill(mask == 0, 0)   # return: 1*14*14*1


class MultiHeadedEuclDistAttention(MultiHeadedDistAttention):
    """Attention based on a parameterized normal pdf taking a molecule's
    euclidean distance matrix as input."""

    def __init__(self, h, d_model):                              # h: 1,  d_model: 64
        super().__init__(h, d_model)
        self.log_prec = nn.Parameter(torch.Tensor(1, 1, 1, h))   # self.log_prec: 1*1*1*1
        self.locs = nn.Parameter(torch.Tensor(1, 1, 1, h))       # self.locs: 1*1*1*1
        nn.init.normal_(self.log_prec, mean=0.0, std=0.1)        # 初始归一化
        nn.init.normal_(self.locs, mean=0.0, std=1.0)

    def create_raw_attn(self, dists, mask):                      # dists: 1*14*14,  mask: 1*14*14
        dists = dists.unsqueeze(-1).expand(-1, -1, -1, self.h)   # dists: 1*14*14*1
        z = torch.exp(self.log_prec) * (dists - self.locs)       # z: 1*14*14*1
        pdf = torch.exp(-0.5 * z ** 2)                           # pdf: 1*14*14*1
        return pdf / pdf.sum(dim=-2, keepdim=True).clamp(1e-9)   # return: 1*14*14*1/1*14*1*1


def attention(query, key, value, mask=None, dropout=None):  # query: 1*1*14*64, key: 1*1*14*64, value: 1*1*14*64
    """Compute 'Scaled Dot Product Attention'."""           # mask: 1*1*14*14
    d_k = query.size(-1)           # d_k: 64
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: 1*1*14*14
    if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)     #
    p_attn = F.softmax(scores, dim=-1).masked_fill(mask == 0, 0)          # p_attn: 1*1*14*14
    if dropout is not None: p_attn = dropout(p_attn)                      #
    return torch.matmul(p_attn, value), p_attn                            # return: 1*1*14*64, p_attn: 1*1*14*14


class MultiHeadedSelfAttention(nn.Module):
    """Applies self-attention as described in the Transformer paper."""

    def __init__(self, h, d_model, dropout=0.1):                          # d_model: 64,   h: 1
        super().__init__()
        self.d_model, self.d_k, self.h = d_model, d_model // h, h         # self.d_model: 64, self.d_k: 64, self.h: 1
        self.attn = None
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(self, x, mask):                                 # x: 1*14*64,  mask: 1*14*14
        # Same mask applied to all h heads.
        mask = mask.unsqueeze(1)                                # mask: 1*1*14*14
        batch_size = x.size(0)                                  # batch_size: 1

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l in self.linears[:3]
        ]                                                       # query, key, value: 1*1*14*64

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask, self.dropout)   # x: 1*1*14*64, self.attn: 1*1*14*14

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous()                                # x: 1*14*1*64
        x = x.view(batch_size, -1, self.d_model)                          # x: 1*14*64
        return self.linears[-1](x)                                        # x: 1*14*64


class AttendingLayer(nn.Module):
    """Stacks the three attention layers and the pointwise feedforward net."""

    def __init__(self, size, eucl_dist_attn, graph_dist_attn, self_attn, ff,
                 dropout):                      # size: 64
        super().__init__()
        self.eucl_dist_attn = eucl_dist_attn    # eucl_dist_attn:
                                                # ModuleList((0): Linear(in_features=64, out_features=64, bias=True)
                                                # (1): Linear(in_features=64, out_features=64, bias=True))

        self.graph_dist_attn = graph_dist_attn  # graph_dist_attn:
                                                # ModuleList((0): Linear(in_features=64, out_features=64, bias=True)
                                                # (1): Linear(in_features=64, out_features=64, bias=True))
                                                # (embedding): Embedding(11, 1))

        self.self_attn = self_attn              # self_attn:
                                                # ModuleList((0): Linear(in_features=64, out_features=64, bias=True)
                                                # (1): Linear(in_features=64, out_features=64, bias=True)
                                                # (2): Linear(in_features=64, out_features=64, bias=True)
                                                # (3): Linear(in_features=64, out_features=64, bias=True))

        self.ff = ff                            # ff:
                                                # Sequential((0): Linear(in_features=64, out_features=256, bias=True)
                                                # (1): ReLU(inplace=True)
                                                # (2): Linear(in_features=256, out_features=64, bias=True))

        self.subconns = clones(SublayerConnection(size, dropout), 4)
                                                # self.subconns:
                                                # ModuleList(
                                                # (0): SublayerConnection(
                                                # (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                # (dropout): Dropout(p=0.0, inplace=False))

                                                # (1): SublayerConnection(
                                                # (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                # (dropout): Dropout(p=0.0, inplace=False))

                                                # (2): SublayerConnection(
                                                # (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                # (dropout): Dropout(p=0.0, inplace=False))

                                                # (3): SublayerConnection(
                                                # (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                # (dropout): Dropout(p=0.0, inplace=False))

        self.size = size                        # size: 64

    def forward(self, x, eucl_dists, graph_dists, mask):   # x: 1*14*64, eucl_dists, graph_dists: 1*14*14, mask: 1*14*14
        eucl_dist_sub = lambda x: self.eucl_dist_attn(eucl_dists, x, mask)    #
        x = self.subconns[0](x, eucl_dist_sub)             # x: 1*14*64
        graph_dist_sub = lambda x: self.graph_dist_attn(graph_dists, x, mask)
        x = self.subconns[1](x, graph_dist_sub)            # x: 1*14*64
        self_sub = lambda x: self.self_attn(x, mask)
        x = self.subconns[2](x, self_sub)                  # x: 1*14*64
        return self.subconns[3](x, self.ff)                # return: 1*14*64


class MessagePassingLayer(nn.Module):
    """Stacks the bond and scalar coupling pair message passing layers."""

    def __init__(self, size, bond_mess, sc_mess, dropout, N):    # size: 64

                                                                 # bond_mess:
                                                                 # ENNMessage(
                                                                 #   (enn): FullyConnectedNet(
                                                                 #     (layers): Sequential(
                                                                 #       (0): Linear(in_features=8, out_features=64, bias=True)
                                                                 #       (1): ReLU(inplace=True)
                                                                 #       (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                                 #       (3): Linear(in_features=64, out_features=64, bias=True)
                                                                 #       (4): ReLU(inplace=True)
                                                                 #       (5): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                                 #       (6): Linear(in_features=64, out_features=64, bias=True)
                                                                 #       (7): ReLU(inplace=True)
                                                                 #       (8): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                                 #       (9): Linear(in_features=64, out_features=4096, bias=True)
                                                                 #     )
                                                                 #   )
                                                                 #   (ann): FullyConnectedNet(
                                                                 #     (layers): Sequential(
                                                                 #       (0): Linear(in_features=1, out_features=64, bias=True)
                                                                 #       (1): ReLU(inplace=True)
                                                                 #       (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                                 #       (3): Linear(in_features=64, out_features=64, bias=True)
                                                                 #       (4): Tanh()
                                                                 #     )
                                                                 #   )
                                                                 # )

                                                                 # sc_mess:
                                                                 # ENNMessage(
                                                                 #   (enn): FullyConnectedNet(
                                                                 #     (layers): Sequential(
                                                                 #       (0): Linear(in_features=16, out_features=64, bias=True)
                                                                 #       (1): ReLU(inplace=True)
                                                                 #       (2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                                 #       (3): Linear(in_features=64, out_features=64, bias=True)
                                                                 #       (4): ReLU(inplace=True)
                                                                 #       (5): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                                 #       (6): Linear(in_features=64, out_features=64, bias=True)
                                                                 #       (7): ReLU(inplace=True)
                                                                 #       (8): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                                 #       (9): Linear(in_features=64, out_features=4096, bias=True)
                                                                 #     )
                                                                 #   )
                                                                 # )

                                                                 # dropout: 0.0

        super().__init__()
        self.bond_mess = bond_mess
        self.sc_mess = sc_mess
        self.linears = clones(nn.Linear(size, size), 2 * N)      # N = 1
                                                                 # ModuleList(
                                                                 #   (0): Linear(in_features=64, out_features=64, bias=True)
                                                                 #   (1): Linear(in_features=64, out_features=64, bias=True))

        self.subconns = clones(SublayerConnection(size, dropout), 2 * N)    # self.subconns:
                                                                            # ModuleList(
                                                                 #   (0): SublayerConnection(
                                                                 #     (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                                 #     (dropout): Dropout(p=0.0, inplace=False)
                                                                 #   )
                                                                 #   (1): SublayerConnection(
                                                                 #     (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
                                                                 #     (dropout): Dropout(p=0.0, inplace=False)
                                                                 #   )
                                                                 # )

    def forward(self, x, bond_x, sc_pair_x, angles, mask, bond_idx, sc_idx,
                angles_idx, t=0):                       # x: 1*14*64, bond_x: 1*14*8, sc_pair_x: 1*29*16, angles: 1*44
                                                        # mask: 1*14*14, bond_idx: 1*14*2, sc_idx: 1*29*2, angles_idx: 1*44
        bond_sub = lambda x: self.linears[2 * t](
            self.bond_mess(x, bond_x, bond_idx, angles, angles_idx, t))   # bond_sub:
        x = self.subconns[2 * t](x, bond_sub)           # x: 1*14*64
        sc_sub = lambda x: self.linears[(2 * t) + 1](
            self.sc_mess(x, sc_pair_x, sc_idx, t=t))
        return self.subconns[(2 * t) + 1](x, sc_sub)    # return: 1*14*64


class Encoder(nn.Module):
    """Encoder stacks N attention layers and one message passing layer."""

    def __init__(self, mess_pass_layer, attn_layer, N):    #
        super().__init__()
        self.mess_pass_layer = mess_pass_layer
        self.attn_layers = clones(attn_layer, N)
        self.norm = LayerNorm(attn_layer.size)

    def forward(self, x, bond_x, sc_pair_x, eucl_dists, graph_dists, angles,
                mask, bond_idx, sc_idx, angles_idx):      # x: 1*14*64, bond_x: 1*14*8, sc_pair_x: 1*29*16, eucl_dists: 1*14*14
                                                          # graph_dists: 1*14*14, angles: 1*44, mask: 1*14*14
                                                          # bond_idx: 1*14*2, sc_idx: 1*29*2, angles_idx: 1*44
        """Pass the inputs (and mask) through each block in turn. Note that for
        each block the same message passing layer is used."""
        for t, attn_layer in enumerate(self.attn_layers):              # t = 0
            x = self.mess_pass_layer(x, bond_x, sc_pair_x, angles, mask,
                                     bond_idx, sc_idx, angles_idx, t)  # x: 1*14*64
            x = attn_layer(x, eucl_dists, graph_dists, mask)           # x: 1*14*64
        return self.norm(x)                                            # x: 1*14*64


# After N blocks of message passing and attending, the encoded atom states are
# transferred to the head of the model: a customized feed-forward net for
# predicting the scalar coupling (sc) constant.

# First the relevant pairs of atom states for each sc constant in the batch
# are selected, concatenated and stacked. Also concatenated to the encoded
# states are a set of raw molecule and sc pair specific features. These states
# are fed into a residual block comprised of a dense layer followed by a type
# specific dense layer of dimension 'd_ff' (the same as the dimension used for
# the pointwise feed-forward net).

# The processed states are passed through to a relatively small feed-forward
# net, which predicts each sc contribution seperately plus a residual.
# Ultimately, the predictions of these contributions and the residual are summed
# to predict the sc constant.

def create_contrib_head(d_in, d_ff, act, dropout=0.0, layer_norm=True):  # d_in: 153, d_ff: 16, act: ReLU(inplace=True)
    layers = hidden_layer(d_in, d_ff, False, dropout, layer_norm, act)   #
    layers += hidden_layer(d_ff, 1, False, 0.0)  # output layer
    return nn.Sequential(*layers)          # Sequential((0): Linear(in_features=153, out_features=16, bias=True)
                                           # (1): ReLU(inplace=True)
                                           # (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
                                           # (3): Linear(in_features=16, out_features=1, bias=True))

class ContribsNet(nn.Module):
    """The feed-forward net used for the sc contribution and final sc constant
    predictions."""
    N_CONTRIBS = 5
    CONTIB_SCALES = [1, 250, 45, 35, 500]  # scales used to make the 5 predictions of similar magnitude

    def __init__(self, d_in, d_ff, vec_in, act, dropout=0.0, layer_norm=True):
        super().__init__()           # d_in: 153, d_ff: 16, vec_in: 256, act: ReLU(inplace=True)
        contrib_head = create_contrib_head(d_in, d_ff, act, dropout, layer_norm)   # contrib_head:
                                                                                   # Sequential(
                                                                                   # (0): Linear(in_features=153, out_features=16, bias=True)
                                                                                   # (1): ReLU(inplace=True)
                                                                                   # (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
                                                                                   # (3): Linear(in_features=16, out_features=1, bias=True))
        self.blocks = clones(contrib_head, self.N_CONTRIBS)     # ModuleList(
                                                                # (0): Sequential(
                                                                # (0): Linear(in_features=153, out_features=16, bias=True)
                                                                # (1): ReLU(inplace=True)
                                                                # (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
                                                                # (3): Linear(in_features=16, out_features=1, bias=True)
                                                                # )
                                                                # (1): Sequential(
                                                                # (0): Linear(in_features=153, out_features=16, bias=True)
                                                                # (1): ReLU(inplace=True)
                                                                # (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
                                                                # (3): Linear(in_features=16, out_features=1, bias=True)
                                                                # )
                                                                # (2): Sequential(
                                                                # (0): Linear(in_features=153, out_features=16, bias=True)
                                                                # (1): ReLU(inplace=True)
                                                                # (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
                                                                # (3): Linear(in_features=16, out_features=1, bias=True)
                                                                # )
                                                                # (3): Sequential(
                                                                # (0): Linear(in_features=153, out_features=16, bias=True)
                                                                # (1): ReLU(inplace=True)
                                                                # (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
                                                                # (3): Linear(in_features=16, out_features=1, bias=True)
                                                                # )
                                                                # (4): Sequential(
                                                                # (0): Linear(in_features=153, out_features=16, bias=True)
                                                                # (1): ReLU(inplace=True)
                                                                # (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
                                                                # (3): Linear(in_features=16, out_features=1, bias=True)
                                                                # )
                                                                # )

    def forward(self, x):                                                          # x: 29*153
        ys = torch.cat(
            [b(x) / s for b, s in zip(self.blocks, self.CONTIB_SCALES)], dim=-1)   # self.CONTIB_SCALES: [1, 250, 45, 35, 500]
                                                                                   # ys: 29*5
        return torch.cat([ys[:, :-1], ys.sum(dim=-1, keepdim=True)], dim=-1)       # ys.sum(dim=-1, keepdim=True): 29*1
                                                                                   # ys[:, :-1]: 29*4
                                                                                   # return: 29*5

class MyCustomHead(nn.Module):
    """Joins the sc type specific residual block with the sc contribution
    feed-forward net."""
    PAD_VAL = -999
    N_TYPES = 8

    def __init__(self, d_input, d_ff, d_ff_contribs, pre_layers=[],
                 post_layers=[], act=nn.ReLU(True), dropout=3 * [0.], norm=False): # d_input: 153, d_ff: 256, d_ff_contribs: 16
        super().__init__()
        fc_pre = hidden_layer(d_input, d_ff, False, dropout[0], norm, act)  # fc_pre:
                                                                            # [Linear(in_features=153, out_features=256, bias=True),
                                                                            # ReLU(inplace=True),
                                                                            # LayerNorm((256,), eps=1e-05, elementwise_affine=True)]
        self.preproc = nn.Sequential(*fc_pre)                               # Sequential(
                                                                            # (0): Linear(in_features=153, out_features=256, bias=True)
                                                                            # (1): ReLU(inplace=True)
                                                                            # (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True))
        fc_type = hidden_layer(d_ff, d_input, False, dropout[1], norm, act)     # fc_type:[Linear(in_features=256, out_features=153, bias=True),
                                                                                # ReLU(inplace=True),
                                                                                # LayerNorm((153,), eps=1e-05, elementwise_affine=True)]
        self.types_net = clones(nn.Sequential(*fc_type), self.N_TYPES)      # self.N_TYPES: 8
        self.contribs_net = ContribsNet(
            d_input, d_ff_contribs, d_ff, act, dropout[2], layer_norm=norm)  # self.contribs_net:
                                                                             # ContribsNet(
        #   (blocks): ModuleList(
        #     (0): Sequential(
        #       (0): Linear(in_features=153, out_features=16, bias=True)
        #       (1): ReLU(inplace=True)
        #       (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
        #       (3): Linear(in_features=16, out_features=1, bias=True)
        #     )
        #     (1): Sequential(
        #       (0): Linear(in_features=153, out_features=16, bias=True)
        #       (1): ReLU(inplace=True)
        #       (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
        #       (3): Linear(in_features=16, out_features=1, bias=True)
        #     )
        #     (2): Sequential(
        #       (0): Linear(in_features=153, out_features=16, bias=True)
        #       (1): ReLU(inplace=True)
        #       (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
        #       (3): Linear(in_features=16, out_features=1, bias=True)
        #     )
        #     (3): Sequential(
        #       (0): Linear(in_features=153, out_features=16, bias=True)
        #       (1): ReLU(inplace=True)
        #       (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
        #       (3): Linear(in_features=16, out_features=1, bias=True)
        #     )
        #     (4): Sequential(
        #       (0): Linear(in_features=153, out_features=16, bias=True)
        #       (1): ReLU(inplace=True)
        #       (2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
        #       (3): Linear(in_features=16, out_features=1, bias=True)
        #     )
        #   )
        # )

    def forward(self, x, sc_types):            #x: 1*29*153, sc_types: 1*29
        # stack inputs with a .view for easier processing
        x, sc_types = x.view(-1, x.size(-1)), sc_types.view(-1)     # x: 29*153, sc_types: 29
        mask = sc_types != self.PAD_VAL           # mask : 29
        x, sc_types = x[mask], sc_types[mask]     # x: 29*153, sc_types: 29

        x_ = self.preproc(x)           # x_: 29*256
        x_types = torch.zeros_like(x)  # x_types: 29*256
        for i in range(self.N_TYPES):  #
            t_idx = sc_types == i      # t_idx: 29
            if torch.any(t_idx):
                x_types[t_idx] = self.types_net[i](x_[t_idx])    #
            else:
                x_types = x_types + 0.0 * self.types_net[i](x_)  # fake call
            # (only necessary for distributed training - to make sure all processes have gradients for all parameters)
        x = x + x_types       # x, x_types: 29*153
        return self.contribs_net(x)     # return: 29*5


class Transformer(nn.Module):
    """Molecule transformer with message passing."""

    def __init__(self, d_atom, d_bond, d_sc_pair, d_sc_mol, N=6, d_model=512,
                 d_ff=2048, d_ff_contrib=128, h=8, dropout=0.1, kernel_sz=128,
                 enn_args={}, ann_args={}):     # d_atom: 21, d_bond: 8, d_sc_mol: 25
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        c = copy.deepcopy
        bond_mess = ENNMessage(d_model, d_bond, kernel_sz, enn_args, ann_args)   #
        sc_mess = ENNMessage(d_model, d_sc_pair, kernel_sz, enn_args)
        eucl_dist_attn = MultiHeadedEuclDistAttention(h, d_model)
        graph_dist_attn = MultiHeadedGraphDistAttention(h, d_model)
        self_attn = MultiHeadedSelfAttention(h, d_model, dropout)
        ff = FullyConnectedNet(d_model, d_model, [d_ff], dropout=[dropout])

        message_passing_layer = MessagePassingLayer(
            d_model, bond_mess, sc_mess, dropout, N)
        attending_layer = AttendingLayer(
            d_model, c(eucl_dist_attn), c(graph_dist_attn), c(self_attn), c(ff),
            dropout
        )

        self.projection = nn.Linear(d_atom, d_model)
        self.encoder = Encoder(message_passing_layer, attending_layer, N)
        self.write_head = MyCustomHead(
            2 * d_model + d_sc_mol, d_ff, d_ff_contrib, norm=True)

    def forward(self, atom_x, bond_x, sc_pair_x, sc_mol_x, eucl_dists,
                graph_dists, angles, mask, bond_idx, sc_idx, angles_idx,
                sc_types):   # atom_x: 1*14*21, bond_x: 1*14*8, sc_pair_x: 1*29*16, sc_mol_x: 1*29*25, eucl_dists: 1*14*14
                             # graph_dists: 1*14*14, angles: 1*44, mask: 1*14*14, bond_idx: 1*14*2,
                             # sc_idx: 1*29*2, angles_idx: 1*44, sc_types: 1*29
        x = self.encoder(
            self.projection(atom_x), bond_x, sc_pair_x, eucl_dists, graph_dists,
            angles, mask, bond_idx, sc_idx, angles_idx
        )                              # 1*14*64
        # for each sc constant in the batch select and concat the relevant pairs
        # of atom  states.
        x = torch.cat(
            [_gather_nodes(x, sc_idx[:, :, 0], self.d_model),
             _gather_nodes(x, sc_idx[:, :, 1], self.d_model),
             sc_mol_x], dim=-1
        )                              # x: 1*29*153
        return self.write_head(x, sc_types)     # return: 29*5



# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=4e-5, help='learning rate')
parser.add_argument('--d_model', type=int, default=64,
                    help='dimenstion of node state vector')
parser.add_argument('--N', type=int, default=1,
                    help='number of encoding layers')
parser.add_argument('--h', type=int, default=1,
                    help='number of attention heads')
parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--fold_id', type=int, default=1)
parser.add_argument('--version', type=int, default=1)
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

# check if distributed training is possible and set model description
#distributed_train = torch.cuda.device_count() > 1
model_str = f'mol_transformer_v{args.version}_fold{args.fold_id}'

# import data
train_df = pd.read_csv(C.PROC_DATA_PATH+'train_proc_df.csv', index_col=0)      # train_df: 4658147*36
                                                                               # ['atom_index_0', 'atom_index_1', 'scalar_coupling_constant', 'type',
#        'atom_0', 'atom_1', 'dist', 'normed_dist', 'dist_min_rad',
#        'dist_electro_neg_adj', 'type_0', 'type_1', 'type_2', 'type_3',
#        'type_4', 'type_5', 'type_6', 'type_7', 'diangle', 'cos_angle',
#        'cos_angle0', 'cos_angle1', 'ave_bond_length', 'std_bond_length',
#        'ave_atom_weight', 'num_atoms', 'num_C_atoms', 'num_F_atoms',
#        'num_H_atoms', 'num_N_atoms', 'num_O_atoms', 'molecule_id', 'fc', 'sd',
#        'pso', 'dso']
test_df = pd.read_csv(C.PROC_DATA_PATH+'test_proc_df.csv', index_col=0)        # test_df: 2505542*31
                                                                               # ['atom_index_0', 'atom_index_1', 'type', 'atom_0', 'atom_1', 'dist',
#        'normed_dist', 'dist_min_rad', 'dist_electro_neg_adj', 'type_0',
#        'type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'type_6', 'type_7',
#        'diangle', 'cos_angle', 'cos_angle0', 'cos_angle1', 'ave_bond_length',
#        'std_bond_length', 'ave_atom_weight', 'num_atoms', 'num_C_atoms',
#        'num_F_atoms', 'num_H_atoms', 'num_N_atoms', 'num_O_atoms',
#        'molecule_id']
atom_df = pd.read_csv(C.PROC_DATA_PATH+'atom_df.csv', index_col=0)             # atom_df: 2358657*22
                                                                               # ['type_H', 'type_C', 'type_N', 'type_O', 'type_F', 'degree_1',
#        'degree_2', 'degree_3', 'degree_4', 'degree_5', 'SP', 'SP2', 'SP3',
#        'hybridization_unspecified', 'aromatic', 'formal_charge', 'atomic_num',
#        'ave_bond_length', 'ave_neighbor_weight', 'donor', 'acceptor',
#        'molecule_id']
bond_df = pd.read_csv(C.PROC_DATA_PATH+'bond_df.csv', index_col=0)             # bond_df: 2439811*11
                                                                               # ['single', 'double', 'triple', 'aromatic', 'conjugated', 'in_ring',
#        'dist', 'normed_dist', 'idx_0', 'idx_1', 'molecule_id']
angle_in_df = pd.read_csv(C.PROC_DATA_PATH+'angle_in_df.csv', index_col=0)     # angle_in_df: 6368267*3
                                                                               # ['molecule_id', 'b_idx', 'cos_angle']
angle_out_df = pd.read_csv(C.PROC_DATA_PATH+'angle_out_df.csv', index_col=0)   # angle_out_df: 2764727*3
                                                                               # ['molecule_id', 'b_idx', 'cos_angle']
graph_dist_df = pd.read_csv(
    C.PROC_DATA_PATH+'graph_dist_df.csv', index_col=0, dtype=np.int32)         # graph_dist_df: 2358657*30
                                                                               # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
#        '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
#        '25', '26', '27', '28', 'molecule_id']
structures_df = pd.read_csv(
    C.PROC_DATA_PATH+'structures_proc_df.csv', index_col=0)                    # structures_df: 2358657*7
                                                                               # ['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z', 'molecule_id']

train_mol_ids = pd.read_csv(C.PROC_DATA_PATH+'train_idxs_8_fold_cv.csv',
                            usecols=[0, args.fold_id], index_col=0
                            ).dropna().astype(int).iloc[:,0]                   # train_mol_ids: 74378*8
                                                                               # columns: ['0', '1', '2', '3', '4', '5', '6', '7']
                                                                               # usecols=[0, args.fold_id]: 74378*1
                                                                               # columns: '0' (args.fold_id=1)
                                                                               # train_mol_ids: (74377,)
val_mol_ids = pd.read_csv(C.PROC_DATA_PATH+'val_idxs_8_fold_cv.csv',
                            usecols=[0, args.fold_id], index_col=0
                            ).dropna().astype(int).iloc[:,0]                   # val_mol_ids: 10626*8
                                                                               # columns: ['0', '1', '2', '3', '4', '5', '6', '7']
                                                                               # usecols=[0, args.fold_id]: 10626*1
                                                                               # columns: '0' (args.fold_id=1)
                                                                               # val_mol_ids: (10626,)

test_mol_ids = pd.Series(test_df['molecule_id'].unique())                      # test_mol_ids: (45772,)


# scale features
train_df, sc_feat_means, sc_feat_stds = scale_features(
    train_df, C.SC_FEATS_TO_SCALE, train_mol_ids, return_mean_and_std=True)    # SC_FEATS_TO_SCALE = ['dist', 'dist_min_rad', 'dist_electro_neg_adj',
                                                                               # 'num_atoms', 'num_C_atoms', 'num_F_atoms', 'num_H_atoms',
                                                                               # 'num_N_atoms', 'num_O_atoms', 'ave_bond_length',
                                                                               # 'std_bond_length', 'ave_atom_weight']

                                                                               # train_df: 4658147*36
                                                                               # sc_feat_means: (12,)
                                                                               # sc_feat_stds: (12,)


test_df = scale_features(test_df, C.SC_FEATS_TO_SCALE, means=sc_feat_means, stds=sc_feat_stds)      # test_df: 2505542*31
atom_df = scale_features(atom_df, C.ATOM_FEATS_TO_SCALE, train_mol_ids)        # atom_df: 2358657*22
bond_df = scale_features(bond_df, C.BOND_FEATS_TO_SCALE, train_mol_ids)        # bond_df: 2439811*11

# group data by molecule id                           # <class 'pandas.core.groupby.groupby.DataFrameGroupBy'>
gb_mol_sc = train_df.groupby('molecule_id')                                    # gb_mol_sc: 85003*36
test_gb_mol_sc = test_df.groupby('molecule_id')                                # test_mol_sc: 45772*31
gb_mol_atom = atom_df.groupby('molecule_id')                                   # gb_mol_atom: 130775*22
gb_mol_bond = bond_df.groupby('molecule_id')                                   # gb_mol_bond: 130775*11
gb_mol_struct = structures_df.groupby('molecule_id')                           # gb_mol_struct: 130775*7
gb_mol_angle_in = angle_in_df.groupby('molecule_id')                           # gb_mol_angle_in: 130775*3
gb_mol_angle_out = angle_out_df.groupby('molecule_id')                         # gb_mol_angle_out: 130771*3
gb_mol_graph_dist = graph_dist_df.groupby('molecule_id')                       # gb_mol_graph_dist: 130775*30

# create dataloaders and fastai DataBunch
set_seed(100)
train_ds = MoleculeDataset(
    train_mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_bond, gb_mol_struct,
    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist
)                                                                             # <class 'moldataset.MoleculeDataset'>
                                                                              # train_ds: 74377
val_ds = MoleculeDataset(
    val_mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_bond, gb_mol_struct,
    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist
)                                                                             # val_ds: 10626
test_ds = MoleculeDataset(
    test_mol_ids, test_gb_mol_sc, gb_mol_atom, gb_mol_bond, gb_mol_struct,
    gb_mol_angle_in, gb_mol_angle_out, gb_mol_graph_dist
)                                                                             # test_ds: 45772

train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=8)    # <class 'torch.utils.data.dataloader.DataLoader'>
                                                                                 # train_dl: 74377
val_dl = DataLoader(val_ds, args.batch_size, num_workers=8)                      # val_dl: 10626
test_dl = DeviceDataLoader.create(
    test_ds, args.batch_size, num_workers=8,
    collate_fn=partial(collate_parallel_fn, test=True), device=device)           # test_df: 45772

db = DataBunch(train_dl, val_dl, collate_fn=collate_parallel_fn, device=device)  # <class 'fastai.basic_data.DataBunch'>
db.test_dl = test_dl

# set up model
set_seed(100)
d_model = args.d_model                                                           # d_model: 64
enn_args = dict(layers=3*[d_model], dropout=3*[0.0], layer_norm=True)  # {'layers': [64, 64, 64], 'dropout': [0.0, 0.0, 0.0], 'layer_norm': True}
ann_args = dict(layers=1*[d_model], dropout=1*[0.0], layer_norm=True,
                out_act=nn.Tanh())           # {'layers': [64], 'dropout': [0.0], 'layer_norm': True, 'out_act': Tanh()}
model = Transformer(
    C.N_ATOM_FEATURES, C.N_BOND_FEATURES, C.N_SC_EDGE_FEATURES,
    C.N_SC_MOL_FEATURES, N=args.N, d_model=d_model, d_ff=d_model*4,
    d_ff_contrib=d_model//4, h=args.h, dropout=args.dropout,
    kernel_sz=min(128, d_model), enn_args=enn_args, ann_args=ann_args)  # C.N_ATOM_FEATURES: 21, C.N_BOND_FEATURES: 8,
                                                                        # C.N_SC_EDGE_FEATURES: 16, C.N_SC_MOL_FEATURES: 25
                                                                        # args.N: 1, d_model: 64, args.h: 1, args.dropout: 0.0

# model cuda
#model = nn.DataParallel(model,device_ids=[3])
#model = model.to(device)

# initialize distributed
''''if distributed_train:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')'''

# train model
callback_fns = [
    partial(GradientClipping, clip=10), GroupMeanLogMAE,
    partial(SaveModelCallback, every='improvement', mode='min',
            monitor='group_mean_log_mae', name=model_str)      # <class 'callbacks.GradientClipping'>
]
learn = Learner(db, model, metrics=[rmse, mae], callback_fns=callback_fns,
                wd=args.wd, loss_func=contribs_rmse_loss)      # args.wd: 0.01

if args.start_epoch > 0:
    learn.load(model_str, device=device)                      # model_str: mol_transformer_v1_fold1
    torch.to(device).empty_cache()
#if distributed_train: learn = learn.to_distributed(args.local_rank)

learn.fit_one_cycle(args.epochs, max_lr=args.lr, start_epoch=args.start_epoch)   #


# make predictions
val_contrib_preds = learn.get_preds(DatasetType.Valid)
test_contrib_preds = learn.get_preds(DatasetType.Test)
val_preds = val_contrib_preds[0][:,-1].detach().numpy() * C.SC_STD + C.SC_MEAN
test_preds = test_contrib_preds[0][:,-1].detach().numpy() * C.SC_STD + C.SC_MEAN


# store results
store_submit(test_preds, model_str, print_head=True)
store_oof(val_preds, model_str, print_head=True)
