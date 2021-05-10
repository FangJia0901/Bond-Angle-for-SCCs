import numpy as np
import torch
from torch.utils.data import Dataset
import constants as C

def _get_existing_group(gb, i):  # gb: 130771 <DataFrameGroupBy object>, i: 52664
    try: group_df = gb.get_group(i)   # group_df: 17*3, columns: molecule_id, b_idx, cos_angles
    except KeyError: group_df = None
    return group_df           # group_df: 17*3

def get_dist_matrix(struct_df):  # struct_df: 16*7, columns:[molecule_name, atom_index, atom, x, y, z, molecule_id]
    locs = struct_df[['x','y','z']].values    # locs: array(16*3), [[], [], ...]
    n_atoms = len(locs)   # n_atoms: 16
    loc_tile = np.tile(locs.T, (n_atoms,1,1))  # locs.T: array(3*16),  loc_tile: array(16*3*16)
    dist_mat = np.sqrt(((loc_tile - loc_tile.T)**2).sum(axis=1))   # dist_mat: 16*16, (loc_tile - loc_tile.T == 16*3*16 - 16*3*16)
    return dist_mat    # return 16*16

class MoleculeDataset(Dataset):   # gb_mol_sc: <class 'pandas.core.groupby.groupby.DataFrameGroupBy'>,
    """Dataset returning inputs and targets per molecule."""
    def __init__(self, mol_ids, gb_mol_sc, gb_mol_atom, gb_mol_bond, 
                 gb_mol_struct, gb_mol_angle_in, gb_mol_angle_out, 
                 gb_mol_graph_dist):  # mol_ids: Series: (74377,);  gb_mol_sc: 85003*36, gb_mol_atom: 130775*22, gb_mol_bond: 130775*11
        """Dataset is constructed from dataframes grouped by molecule_id."""  # gb_mol_struct: 130775*7,  gb_mol_angle_in: 130775*3
        self.n = len(mol_ids)      # self.n: 74377              # gb_mol_graph_dist: 130775*30        gb_mol_angle_in: 130771*3
        self.mol_ids = mol_ids     # self.mol_ids: Series: (74377,)
        self.gb_mol_sc = gb_mol_sc           # gb_mol_sc: <class 'pandas.core.groupby.groupby.DataFrameGroupBy'>, 85003*36
        self.gb_mol_atom = gb_mol_atom
        self.gb_mol_bond = gb_mol_bond
        self.gb_mol_struct = gb_mol_struct
        self.gb_mol_angle_in = gb_mol_angle_in
        self.gb_mol_angle_out = gb_mol_angle_out
        self.gb_mol_graph_dist = gb_mol_graph_dist

    def __len__(self):
        return self.n

    def __getitem__(self, idx):     # <__main__.MoleculeDataset object at 0x7f762ba4ba978>, idx: 29957
        return (self.gb_mol_sc.get_group(self.mol_ids[idx]),
                self.gb_mol_atom.get_group(self.mol_ids[idx]), 
                self.gb_mol_bond.get_group(self.mol_ids[idx]), 
                self.gb_mol_struct.get_group(self.mol_ids[idx]), 
                self.gb_mol_angle_in.get_group(self.mol_ids[idx]), 
                _get_existing_group(self.gb_mol_angle_out, self.mol_ids[idx]),
                self.gb_mol_graph_dist.get_group(self.mol_ids[idx]))     #

def arr_lst_to_padded_batch(arr_lst, dtype=torch.float, 
                            pad_val=C.BATCH_PAD_VAL):       # arr_lst: [array(16*21),], pad_val=0
    tensor_list = [torch.Tensor(arr).type(dtype) for arr in arr_lst]  # arr: 16*21,
    batch = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=pad_val)   # batch: 1*16*21
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

    for b in range(batch_size):    # sc_df: 37*36, atom_df: 16*22, bond_df:16*11, struct_df: 16*7
        (sc_df, atom_df, bond_df, struct_df, angle_in_df, angle_out_df, 
         graph_dist_df) = batch[b]    # angle_in_df: 39*3, angle_out_df: 17*3, graph_dist_df: 16*30(padding到30)
        n_atoms, n_pairs, n_sc = len(atom_df), len(bond_df), len(sc_df)   # n_atoms: 16, n_pairs: 16, n_sc: 37
        n_pad = C.MAX_N_ATOMS - n_atoms     # n_pad: 29-16=13
        eucl_dists_ = get_dist_matrix(struct_df)   # eucl_dists_: array(16*16)
        eucl_dists_ = np.pad(eucl_dists_, [(0, 0), (0, n_pad)], 'constant', 
                             constant_values=999)    # padding, eucl_dists_: 16*29
        
        atom_x.append(atom_df[C.ATOM_FEATS].values)   # atom_x: [array(16*21), ...]
        bond_x.append(bond_df[C.BOND_FEATS].values)   # bond_x: [array(16*8), ...]
        sc_x.append(sc_df[C.SC_EDGE_FEATS].values)    # sc_x: [array(37*16), ...]
        sc_m_x.append(sc_df[C.SC_MOL_FEATS].values)   # sc_m_x: [array(37*25), ...]
        sc_types.append(sc_df['type'].values)         # sc_types: [array(37,), ...]
        if not test: 
            n_sc_pad = C.MAX_N_SC - n_sc              # n_sc_pad=135-37=98
            sc_vals_ = sc_df[C.CONTRIB_COLS+[C.TARGET_COL]].values     # sc_vals_: 37*5, columns: ['fc', 'sd', 'pso', 'dso', 'sc']
            sc_vals.append(np.pad(sc_vals_, [(0, n_sc_pad), (0, 0)], 'constant', 
                                  constant_values=-999))    # sc_vals: 135*5, padding至135
        eucl_dists.append(eucl_dists_)                      # eucl_dists: [array(16*29), array(), ...]
        graph_dists.append(graph_dist_df.values[:,:-1])     # graph_dists: [array(16*30), ...]
        angles_in.append(angle_in_df['cos_angle'].values)   # angles_in: [array(39,)]   angle_in_df: df, (39*3)
        if angle_out_df is not None: 
            angles_out.append(angle_out_df['cos_angle'].values)   # angles_out: [array(17,)],  angle_out_df: df, (17*3)
        else: 
            angles_out.append(np.array([C.BATCH_PAD_VAL]))
        
        mask.append(np.pad(np.ones(2 * [n_atoms]), [(0, 0), (0, n_pad)], 
                           'constant'))                           # mask: [array(16*29), ...]
        bond_idx.append(bond_df[['idx_0', 'idx_1']].values)       # bond_idx: [array(16*2), ...]
        sc_idx.append(sc_df[['atom_index_0', 'atom_index_1']].values)   # sc_idx: [array(37*2), ...]
        angles_in_idx.append(angle_in_df['b_idx'].values)         # angles_in_idx: [array(39,)]
        if angle_out_df is not None: 
            angles_out_idx.append(angle_out_df['b_idx'].values)   # angles_out_idx: [array(17,)]
        else:
            angles_out_idx.append(np.array([0.]))
        
        n_atom_sum += n_atoms      # n_atom_sum: 16
        n_pairs_sum += n_pairs     # n_pairs_sum: 16
        
    atom_x = arr_lst_to_padded_batch(atom_x, pad_val=0.)       # atom_x: tensor(1*16*21)  padding至batch中原子个数最大的分子
    bond_x = arr_lst_to_padded_batch(bond_x)                   # bond_x: tensor(1*16*8)
    max_n_atoms = atom_x.size(1)                               # max_n_atoms: 16
    max_n_bonds = bond_x.size(1)                               # max_n_bonds: 16
    angles_out_idx = [a + max_n_bonds for a in angles_out_idx]  # angles_out_idx: [array(17,)]
    
    sc_x = arr_lst_to_padded_batch(sc_x)                       # sc_x: tensor(1*37*16)
    sc_m_x =arr_lst_to_padded_batch(sc_m_x)                    # sc_m_x: tensor(1*37*25)
    if not test: sc_vals = arr_lst_to_padded_batch(sc_vals)    # sc_vals: tensor(1*135*5)
    else: sc_vals = torch.tensor([0.] * batch_size)
    sc_types = arr_lst_to_padded_batch(sc_types, torch.long)   # sc_types: tensor(1*37)

    #
    mask = arr_lst_to_padded_batch(mask, torch.uint8, 0)       # mask: tensor(1*16*29)   mask第二维度是分子中原子个数
    mask = mask[:,:,:max_n_atoms].contiguous()                 # mask: tensor(1*16*16)  切片

    bond_idx = arr_lst_to_padded_batch(bond_idx, torch.long, 0) # bond_idx: tensor(1*16*2)
    sc_idx = arr_lst_to_padded_batch(sc_idx, torch.long, 0)    # sc_idx: tensor(1*37*2)

    angles_in_idx = arr_lst_to_padded_batch(angles_in_idx, torch.long, 0)  # angles_in_idx: tensor(1*39)
    angles_out_idx = arr_lst_to_padded_batch(angles_out_idx, torch.long, 0)  # angles_out_idx: tensor(1*17)
    angles_idx = torch.cat((angles_in_idx, angles_out_idx), dim=-1).contiguous()  # angles_idx: tensor(1*56) 39+17=56

    eucl_dists = arr_lst_to_padded_batch(eucl_dists, pad_val=999)     # eucl_dists: tensor(1*16*29)
    eucl_dists = eucl_dists[:,:,:max_n_atoms].contiguous()            # eucl_dists: tensor(1*16*16)  切片
    graph_dists = arr_lst_to_padded_batch(graph_dists, torch.long, 10)  # graph_dists: tensor(1*16*29)
    graph_dists = graph_dists[:,:,:max_n_atoms].contiguous()            # graph_dists: tensor(1*16*16)  切片

    angles_in = arr_lst_to_padded_batch(angles_in)                    # angles_in: tensor(1*39)
    angles_out = arr_lst_to_padded_batch(angles_out)                  # angles_out: tensor(1*17)
    angles = torch.cat((angles_in, angles_out), dim=-1).contiguous()  # angles: tensor(1*56)
    
    return (atom_x, bond_x, sc_x, sc_m_x, eucl_dists, graph_dists, angles, mask, 
            bond_idx, sc_idx, angles_idx, sc_types), sc_vals          #  sc_vals: tensor(1*135*5)

# atom_x: tensor(1*16*21),  bond_x: tensor(1*16*8), sc_x: tensor(1*37*16),  sc_m_x: tensor(1*37*25)
# eucl_dists: tensor(1*16*16),   graph_dists: tensor(1*16*16)  angles: tensor(1*56),  mask: tensor(1*16*16)
# bond_idx:  tensor(1*16*2),   sc_idx: tensor(1*37*2),  angles_idx: tensor(1*56), sc_types: tensor(1*37)