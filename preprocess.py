import pdb
import gc
import numpy as np
import pandas as pd
from itertools import combinations
from glob import glob
import deepchem as dc
from rdkit.Chem import rdmolops, ChemicalFeatures
from xyz2mol import read_xyz_file, xyz2mol
from utils import print_progress
import constants as C
from pandas.testing import assert_frame_equal

train_df = pd.read_csv(C.RAW_DATA_PATH + 'train.csv', index_col=0)
a = train_df['molecule_name'].unique()
mol_feat_columns = ['ave_bond_length', 'std_bond_length', 'ave_atom_weight']
xyz_filepath_list = list(glob(C.RAW_DATA_PATH + 'structures/*.xyz'))
xyz_filepath_list.sort()
xyz_filepath_list_1 = []
for i in range(len(xyz_filepath_list)):
    filepath = xyz_filepath_list[i]
    mol_name = filepath.split('/')[-1][:-4]
    if (a == mol_name).any():
        xyz_filepath_list_1.append(filepath)
xyz_filepath_list = xyz_filepath_list_1

## Functions to create the RDKit mol objects
def mol_from_xyz(filepath, add_hs=True, compute_dist_centre=False):
    """Wrapper function for calling xyz2mol function."""
    charged_fragments = True  # alternatively radicals are made
    # quick is faster for large systems but requires networkx
    # if you don't want to install networkx set quick=False and
    # uncomment 'import networkx as nx' at the top of the file
    quick = True

    atomicNumList, charge, xyz_coordinates = read_xyz_file(filepath)
    mol, dMat = xyz2mol(atomicNumList, charge, xyz_coordinates,
                        charged_fragments, quick, check_chiral_stereo=False)  #The molecular structure mol and distance matrix can be obtained by reading the XYZ file
    return mol, np.array(xyz_coordinates), dMat

def get_molecules():
    """
    Constructs rdkit mol objects derrived from the .xyz files. Also returns:
        - mol ids (unique numerical ids)
        - set of molecule level features
        - arrays of xyz coordinates
        - euclidean distance matrices
        - graph distance matrices.
    All objects are returned in dictionaries with 'mol_name' as keys.
    """
    mols, mol_ids, mol_feats = {}, {}, {}
    xyzs, dist_matrices, graph_dist_matrices = {}, {}, {}
    print('Create molecules and distance matrices.')
    for i in range(C.N_MOLS):
        print_progress(i, C.N_MOLS)
        filepath = xyz_filepath_list[i]
        mol_name = filepath.split('/')[-1][:-4]
        mol, xyz, dist_matrix = mol_from_xyz(filepath)    #Read the XYZ file to get the structure mol and the distance matrix, coordinates
        mols[mol_name] = mol
        xyzs[mol_name] = xyz
        dist_matrices[mol_name] = dist_matrix
        mol_ids[mol_name] = i                            # The molecular number in the data set is used as the molecular id

        # make padded graph distance matrix dataframes
        n_atoms = len(xyz)
        graph_dist_matrix = pd.DataFrame(np.pad(
            rdmolops.GetDistanceMatrix(mol),
            [(0, 0), (0, C.MAX_N_ATOMS - n_atoms)], 'constant'))    #obtain distance matrix through ramolops.GetDistanceMatrix
        graph_dist_matrix['molecule_id'] = n_atoms * [i]          
        graph_dist_matrices[mol_name] = graph_dist_matrix           #dict：value: dataframe

        # compute molecule level features
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)      #通过ramolops.GetDistanceMatrix获取 图邻接矩阵
        atomic_num_list, _, _ = read_xyz_file(filepath)    #读取XYZ文件获取分子中各原子的原子序数和坐标
        dists = dist_matrix.ravel()[np.tril(adj_matrix).ravel()==1]   #通过邻接矩阵的下三角获取与相邻原子之间的距离
        mol_feats[mol_name] = pd.Series(
            [np.mean(dists), np.std(dists), np.mean(atomic_num_list)],
            index=mol_feat_columns)                                  #获取与领接原子之间距离均值和标准差、原子序数的均值(分子级特征)
    return mols, mol_ids, mol_feats, xyzs, dist_matrices, graph_dist_matrices  #返回训练集所有分子结构mol和分子ids,分子级特征,原子坐标,距离矩阵，图距离矩阵


## Functions to create features at the scalar coupling level.
def map_atom_info(df, atom_idx, struct_df):
    """Adds xyz-coordinates of atom_{atom_idx} to 'df'."""
    df = pd.merge(df, struct_df, how = 'left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])       # 合成训练(测试) 和 结构文件
    df = df.drop('atom_index', axis=1)                              # 删除struct_df中的列，df中的atom_index列带有后缀
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})                  # 添加index原子的后缀
    return df

def add_dist(df, struct_df):
    """Adds euclidean distance between scalar coupling atoms to 'df'."""
    df = map_atom_info(df, 0, struct_df)                           # 添加atom_idx_0中的结构信息
    df = map_atom_info(df, 1, struct_df)                           # 添加atom_idx_1中的结构信息
    p_0 = df[['x_0', 'y_0', 'z_0']].values                         # 获取index_0原子的坐标，返回numpy
    p_1 = df[['x_1', 'y_1', 'z_1']].values
    df['dist'] = np.linalg.norm(p_0 - p_1, axis=1)                 # 计算原子对之间欧式距离
    df.drop(columns=['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], inplace=True)     # 获取原子对之间距离后，删除原子的坐标
    return df

def transform_per_atom_group(df, a_idx, col='dist', trans='mean'):
    """Apply transformation 'trans' on feature in 'col' to scalar coupling
    constants grouped at the atom level."""
    return df.groupby(
        ['molecule_name', f'atom_index_{a_idx}'])[col].transform(trans)

def inv_dist_per_atom(df, a_idx, d_col='dist', power=3):
    """Compute sum of inverse distances of scalar coupling constants grouped at
    the atom level."""
    trans = lambda x: 1 / sum(x ** -power)
    return transform_per_atom_group(df, a_idx, d_col, trans=trans)

def inv_dist_harmonic_mean(df, postfix=''):
    """Compute the harmonic mean of inverse distances of atom_0 and atom_1."""
    c0, c1 = 'inv_dist0' + postfix, 'inv_dist1' + postfix
    return (df[c0] * df[c1]) / (df[c0] + df[c1])

def add_atom_counts(df, struct_df):
    """Add atom counts (total and per type) to 'df'."""
    pd.options.mode.chained_assignment = None
    atoms_per_mol_df = struct_df.groupby(['molecule_name', 'atom']).count()     # 计算每个分子中各个原子个数
    atoms_per_mol_map = atoms_per_mol_df['atom_index'].unstack().fillna(0)      #
    atoms_per_mol_map = atoms_per_mol_map.astype(int).to_dict()
    df['num_atoms'] = 0
    for a in atoms_per_mol_map:
        df[f'num_{a}_atoms'] = df['molecule_name'].map(atoms_per_mol_map[a])
        df['num_atoms'] += df[f'num_{a}_atoms']
    return df

# source: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def dihedral(p):
    """Praxeolitic formula: 1 sqrt, 1 cross product"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)

def cosine_angle(p):
    p0, p1, p2 = p[0], p[1], p[2]
    v1, v2 = p0 - p1, p2 - p1
    return np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))

def add_sc_angle_features(df, xyzs, dist_matrices):
    """
    Adds the following angle features to 'df':
    - diangle: for 3J couplings
    - cos_angle: for 2J couplings, angle between sc atom 0, atom in between sc
        atoms and sc atom 1
    - cos_angle0: for all coupling types, cos angle between sc atoms and atom
        closest to atom 0 (except for 1J coupling)
    - cos_angle1: for all coupling types, cos angle between sc atoms and atom
        closest to atom 1
    """
    df['diangle'] = 0.0
    df['cos_angle'] = 0.0
    df['cos_angle0'] = 0.0
    df['cos_angle1'] = 0.0
    diangles, cos_angles, cos_angles0, cos_angles1 = {}, {}, {}, {}
    print('Add scalar coupling angle based features.')
    n = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        print_progress(i, n, 500000)
        #if row['molecule_name'] == 'dsgdb9nsd_086797':
        #    pdb.set_trace()
        mol_name = row['molecule_name']
        mol, xyz = mols[mol_name], xyzs[mol_name]
        dist_matrix = dist_matrices[mol_name]
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        idx0, idx1 = row['atom_index_0'], row['atom_index_1']
        atom_ids = rdmolops.GetShortestPath(mol, idx0, idx1)

        if len(atom_ids)==4:
            diangles[idx] = dihedral(xyz[atom_ids,:])
        elif len(atom_ids)==3:
            cos_angles[idx] = cosine_angle(xyz[atom_ids,:])

        if row['type'] not in [0, 2]:
            neighbors0 = np.where(adj_matrix[idx0]==1)[0]           ###
            if len(neighbors0) > 0:
                idx0_closest = neighbors0[
                    dist_matrix[idx0][neighbors0].argmin()]
                cos_angles0[idx] = cosine_angle(
                    xyz[[idx0_closest, idx0, idx1],:])
        neighbors1 = np.setdiff1d(np.where(adj_matrix[idx1]==1)[0], [idx0])
        if len(neighbors1) > 0:
            idx1_closest = neighbors1[
                dist_matrix[idx1][neighbors1].argmin()]
            cos_angles1[idx] = cosine_angle(
                xyz[[idx0, idx1, idx1_closest],:])

    df['diangle'] = pd.Series(diangles).abs()
    df['cos_angle'] = pd.Series(cos_angles)
    df['cos_angle0'] = pd.Series(cos_angles0)
    df['cos_angle1'] = pd.Series(cos_angles1)
    df.fillna(0., inplace=True)
    return df

def add_sc_features(df, structures_df, mol_feats, xyzs, dist_matrices, mol_ids):
    """Add scalar coupling edge and molecule level features to 'df'."""
    # add euclidean distance between scalar coupling atoms
    df = add_dist(df, structures_df)                   # obtain the distance of atomic pairs

    # compute distance normalized by scalar coupling type mean and std
    gb_type_dist = df.groupby('type')['dist']           ###获取耦合种类相同的组信息, 返回特定的SeriesGroupby数据格式
    df['normed_dist'] = ((df['dist'] - gb_type_dist.transform('mean'))
                         / gb_type_dist.transform('std'))      # Normalize the distance of the same coupling type

    # add distance features adjusted for atom radii and electronegativity
    df['R0'] = df['atom_0'].map(C.ATOMIC_RADIUS)                # add the radius of the atom
    df['R1'] = df['atom_1'].map(C.ATOMIC_RADIUS)
    df['E0'] = df['atom_0'].map(C.ELECTRO_NEG)                  # add electronegativity
    df['E1'] = df['atom_1'].map(C.ELECTRO_NEG)
    df['dist_min_rad'] = df['dist'] - df['R0'] - df['R1']       # obtain the value of the surface distance
    df['dist_electro_neg_adj'] = df['dist'] * (df['E0'] + df['E1']) / 2   # obtain the electrode distance of the coupling pair
    df.drop(columns=['R0','R1','E0','E1'], inplace=True)        # Deletes the radii and electronegativity of the atoms

    # map scalar coupling types to integers and add dummy variables
    df['type'] = df['type'].map(C.TYPES_MAP)                   # 映射耦合类型为数值0～7
    df = pd.concat((df, pd.get_dummies(df['type'], prefix='type')), axis=1)   # 通过pd.get_dummies函数 将耦合类型映射为one-hot编码

    # add angle related features
    df = add_sc_angle_features(df, xyzs, dist_matrices)        # 增加了diangle、cos_angle、cos_angle0、cos_angle1四列特征

    # add molecule level features
    mol_feat_df = pd.concat(mol_feats, axis=1).T       ####### mol_feats是{},cancat在列方向堆叠，再转置。将字典型数据转换成dataframe型
    mol_feat_dict = mol_feat_df.to_dict()
    for f in mol_feat_columns:
        df[f] = df['molecule_name'].map(mol_feat_dict[f])     # 把mol_feats字典型数据映射到df中去

    # add atom counts per molecule
    df = add_atom_counts(df, structures_df)            # 将分子中原子个数加入到df中

    # add molecule ids
    df['molecule_id'] = df['molecule_name'].map(mol_ids)      # 将分子名编码成分子id加到df中

    return df

def store_train_and_test(all_df):
    """Split 'all_df' back to train and test and store the resulting dfs."""
    train_df = all_df.iloc[:C.N_SC_TRAIN]
    test_df = all_df.iloc[C.N_SC_TRAIN:]
    train_df.drop(columns='molecule_name', inplace=True)
    test_df.drop(columns='molecule_name', inplace=True)
    test_df.to_csv(C.PROC_DATA_PATH + 'test_proc_df.csv')

    train_df[C.TARGET_COL] = (train_df[C.TARGET_COL] - C.SC_MEAN) / C.SC_STD
    train_df.to_csv(C.PROC_DATA_PATH + 'train_proc_df.csv')

#Functions to create atom and bond level features
def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot

def get_bond_features(mol, eucl_dist):
    """
    Compute the following features for each bond in 'mol':
        - bond type: categorical {1: single, 2: double, 3: triple,
            4: aromatic} (one-hot)
        - is conjugated: bool {0, 1}
        - is in ring: bool {0, 1}
        - euclidean distance: float
        - normalized eucl distance: float
    """
    n_bonds = mol.GetNumBonds()                                    # 获取分子中键的个数
    features = np.zeros((n_bonds, C.N_BOND_FEATURES))              # 初始化键特征数组
    bond_idx = np.zeros((n_bonds, 2))                              # 初始化键索引数组
    for n, e in enumerate(mol.GetBonds()):                         # 遍历分子中每个键
        i = e.GetBeginAtomIdx()                                    # 获取键中初始原子索引
        j = e.GetEndAtomIdx()                                      # 获取键中末尾原子索引
        dc_e_feats = dc.feat.graph_features.bond_features(e).astype(int)      # 通过内置函数获取键的特征，包含单、双、三键、是否共轭、是否成环
        features[n, :6] = dc_e_feats                               # 填充键特征数组
        features[n, 6] = eucl_dist[i, j]                           # 填充键长
        bond_idx[n] = i, j                                         # 获取成键原子的编号
    sorted_idx = bond_idx[:,0].argsort()                           # 获取数值中的第一列，即非H
    dists = features[:, 6]
    features[:, 7] = (dists - dists.mean()) / dists.std() # normed_dist
    return features[sorted_idx], bond_idx[sorted_idx]              # 给每个分子按照初始原子索引进行排序

def get_atom_features(mol, dist_matrix):
    """
    Compute the following features for each atom in 'mol':
        - atom type: H, C, N, O, F (one-hot)
        - degree: 1, 2, 3, 4, 5 (one-hot)
        - Hybridization: SP, SP2, SP3, UNSPECIFIED (one-hot)
        - is aromatic: bool {0, 1}
        - formal charge: int
        - atomic number: float
        - average bond length: float
        - average weight of neigboring atoms: float
    """
    n_atoms = mol.GetNumAtoms()                                # 获取原子个数
    features = np.zeros((n_atoms, C.N_ATOM_FEATURES))          # 初始化原子特征数组
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)              # 通过rdmolops.GetAdjacencyMatrix函数获取领接矩阵
    for a in mol.GetAtoms():
        idx = a.GetIdx()                                       # 通过mol结构内置函数GetIdx()获取id
        if sum(adj_matrix[idx]) > 0:                           #
            ave_bond_length = np.mean(dist_matrix[idx][adj_matrix[idx]==1])    # 获取键长的均值
            ave_neighbor_wt = np.mean(
                [n.GetAtomicNum() for n in a.GetNeighbors()])          # 获取周边原子的质子数的均值
        else:
            ave_bond_length, ave_neighbor_wt = 0.0, 0.0                # 如果没有周边原子，赋值为0

        sym = a.GetSymbol()                                            # 获取原子的标记symbol
        a_feats = one_hot_encoding(sym, C.SYMBOLS) \
            + one_hot_encoding(a.GetDegree(), C.DEGREES) \
            + one_hot_encoding(a.GetHybridization(), C.HYBRIDIZATIONS) \
            + [a.GetIsAromatic(), a.GetFormalCharge(), a.GetAtomicNum(),
               ave_bond_length, ave_neighbor_wt]      # one-hot编码分子中原子的类型、度矩阵、杂化类型； 添加分子中是否有芳香环、部分电荷、原子个数、键长均值、周边原子的质子均值
        features[idx, :len(a_feats)] = np.array(a_feats)          # 填充features数组
    return features

def get_atom_and_bond_features(mols, mol_ids, dist_matrices):       # mol_ids: {}
    atom_features, bond_features = [], []
    bond_idx, atom_to_m_id, bond_to_m_id = [], [], []
    print('Get atom and bond features.')
    for it, m_name in enumerate(mols):
        print_progress(it, C.N_MOLS)
        m_id, mol = mol_ids[m_name], mols[m_name]                   # m_id: int
        dist_matrix = dist_matrices[m_name]                         # 获取分子的距离矩阵
        n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()     # 根据mol结构文件获取分子中 原子的个数和 键的个数
                                                                    # mol.GetNumAtoms(), mol.GetNumBonds()
        atom_features.append(get_atom_features(mol, dist_matrix))   # 针对每个分子文件，获取分子中原子特征

        e_feats, b_idx = get_bond_features(mol, dist_matrix)        # 针对每个分子文件，获取键的特征
        bond_features.append(e_feats)
        bond_idx.append(b_idx)                                      # b_idx: array(n_bonds, 2)

        atom_to_m_id.append(np.repeat(m_id, n_atoms))               # 根据原子个数产生对应个数的分子id, [array, array, array, ...]
        bond_to_m_id.append(np.repeat(m_id, n_bonds))               # 根据键个数产生对应个数的分子id
    atom_features = pd.DataFrame(
        np.concatenate(atom_features), columns=C.ATOM_FEATS)   # atom_features:[np.array(n_atoms*atom_feats), np.array(), ...]
    bond_features = pd.DataFrame(
        np.concatenate(bond_features), columns=C.BOND_FEATS)   # bond_features:[np.array(n_bonds*bond_feats), np.array(), ...]
    bond_idx = np.concatenate(bond_idx)                        # bond_idx: [np.array(n_bonds*2), np.array(), ...]
    bond_features['idx_0'] = bond_idx[:,0]                     # 给bond_features——df添加成键原子的索引
    bond_features['idx_1'] = bond_idx[:,1]                     #
    atom_features['molecule_id'] = np.concatenate(atom_to_m_id)   # 给atom_features--df添加列
    bond_features['molecule_id'] = np.concatenate(bond_to_m_id)   # 给bond_features--df添加列

    return atom_features, bond_features                           # 返回 atom_features--df, bond_features--df

def store_atom_and_bond_features(atom_df, bond_df):               # 将atom_features, bond_features保存成csv文件
    atom_df.to_csv(C.PROC_DATA_PATH + 'atom_df.csv')
    bond_df.to_csv(C.PROC_DATA_PATH + 'bond_df.csv')


## Functions to store distance matrices
def store_graph_distances(graph_dist_matrices):
    graph_dist_df = pd.concat(graph_dist_matrices)       # 将graph_dist_matrices字典型数据{dataframe, dataframe, ...}转换成dataframe
    graph_dist_df.reset_index(drop=True, inplace=True)   # 重新设置索引
    graph_dist_df.replace(1e8, 10, inplace=True) # fix for one erroneous atom
    graph_dist_df = graph_dist_df.astype(int)                    # 将数值转化为整数型
    graph_dist_df.to_csv(C.PROC_DATA_PATH + 'graph_dist_df.csv')    # 将 图距离矩阵df 转化成 csv文件

def store_eucl_distances(dist_matrices, atom_df):
    dist_df = pd.DataFrame(np.concatenate(
        [np.pad(dm, [(0,0), (0, C.MAX_N_ATOMS-dm.shape[1])], mode='constant')
        for dm in dist_matrices.values()]           # dist_matrices: {k:array, K:array, ...} #array是个方阵，需要将列padding到29
    ))
    dist_df['molecule_id'] = atom_df['molecule_id']  # Add another column to the dist_df up to 30 columns
    dist_df.to_csv(C.PROC_DATA_PATH + 'dist_df.csv')   # Convert dist.df to a CSV file

## Functions to compute cosine angles for all bonds
def _get_combinations(idx_0_group):                    #
    s = list(idx_0_group['idx_1'])[1:]
    return [list(combinations(s, r))[-1] for r in range(len(s), 0, -1)]

def get_all_cosine_angles(bond_df, structures_df, mol_ids, store=True):
    """Compute cosine angles between all bonds. Grouped at the bond level."""
    bond_idx = bond_df[['molecule_id', 'idx_0', 'idx_1']].astype(int)
    in_out_idx = pd.concat((
        bond_idx,
        bond_idx.rename(columns={'idx_0': 'idx_1', 'idx_1': 'idx_0'})
    ), sort=False)
    gb_mol_0_bond_idx = in_out_idx.groupby(['molecule_id', 'idx_0'])

    angle_idxs = []
    print('Get cosine angle indices.')
    for it, (mol_id, idx_0) in enumerate(gb_mol_0_bond_idx.groups):
        # iterate over all atoms (atom_{idx0})
        print_progress(it, gb_mol_0_bond_idx.ngroups, print_iter=500000)
        idx_0_group = gb_mol_0_bond_idx.get_group((mol_id, idx_0))
        combs = _get_combinations(idx_0_group)
        for i, comb in enumerate(combs):
            # iterate over all bonds of the atom_{idx0} (bond_{idx_0, idx_1})
            idx_1 = idx_0_group['idx_1'].iloc[i]
            for idx_2 in comb:
                # iterate over all angles between bonds with bond_{idx_0, idx_1}
                # as base
                angle_idxs.append((mol_id, idx_0, idx_1, idx_2))
    angle_cols = ['molecule_id', 'atom_index_0', 'atom_index_1', 'atom_index_2']
    angle_df = pd.DataFrame(angle_idxs, columns=angle_cols)
    angle_df['molecule_name'] = angle_df['molecule_id'].map(
        {v:k for k,v in mol_ids.items()})
    angle_df.drop(columns='molecule_id', inplace=True)

    for i in range(3): angle_df = map_atom_info(angle_df, i, structures_df)
    drop_cols = ['atom_0', 'atom_1', 'atom_2', 'molecule_id_x', 'molecule_id_y']
    angle_df.drop(columns=drop_cols, inplace=True)

    for c in ['x', 'y', 'z']:
        angle_df[f'{c}_0_1'] = \
            angle_df[f'{c}_0'].values - angle_df[f'{c}_1'].values
        angle_df[f'{c}_0_2'] = \
            angle_df[f'{c}_0'].values - angle_df[f'{c}_2'].values

    def cos_angles(v1, v2):
        return (v1 * v2).sum(1) / np.sqrt((v1 ** 2).sum(1) * (v2 ** 2).sum(1))

    angle_df['cos_angle'] = cos_angles(
        angle_df[['x_0_1', 'y_0_1', 'z_0_1']].values,
        angle_df[['x_0_2', 'y_0_2', 'z_0_2']].values
    )
    angle_df = angle_df[['molecule_id', 'atom_index_0', 'atom_index_1',
                         'atom_index_2', 'cos_angle']]
    angle_df.to_csv(C.PROC_DATA_PATH + 'angle_all_df.csv')
    gb_mol_angle = angle_df.groupby('molecule_id')
    gb_mol_bond_idx = bond_idx.groupby('molecule_id')

    angle_to_in_bond, angle_to_out_bond = [], []
    print('Get cosine angles.')
    for i, mol_id in enumerate(mol_ids.values()):
        print_progress(i, C.N_MOLS)
        b_df = gb_mol_bond_idx.get_group(mol_id)
        a_df = gb_mol_angle.get_group(mol_id)
        b_in_idxs = b_df[['idx_0', 'idx_1']].values
        b_out_idxs = b_df[['idx_1', 'idx_0']].values
        a1 = a_df[['atom_index_0', 'atom_index_1', 'cos_angle']].values
        a2 = a_df[['atom_index_0', 'atom_index_2', 'cos_angle']].values
        for a in np.concatenate((a1, a2)):
            if any(np.all(b_in_idxs==a[:2], axis=1)):
                a_to_in_idx = np.where(np.all(b_in_idxs==a[:2], axis=1))[0][0]
                angle_to_in_bond.append((mol_id, a_to_in_idx, a[-1]))
            if any(np.all(b_out_idxs==a[:2], axis=1)):
                a_to_out_idx = np.where(np.all(b_out_idxs==a[:2], axis=1))[0][0]
                angle_to_out_bond.append((mol_id, a_to_out_idx, a[-1]))
    angle_in_df = pd.DataFrame(
        angle_to_in_bond, columns=['molecule_id', 'b_idx', 'cos_angle'])
    angle_out_df = pd.DataFrame(
        angle_to_out_bond, columns=['molecule_id', 'b_idx', 'cos_angle'])

    if store: store_angles(angle_in_df, angle_out_df)
    return angle_in_df, angle_out_df

def store_angles(angle_in_df, angle_out_df):
    angle_in_df.to_csv(C.PROC_DATA_PATH + 'angle_in_df.csv')
    angle_out_df.to_csv(C.PROC_DATA_PATH + 'angle_out_df.csv')


def process_and_store_structures(structures_df, mol_ids):
    structures_df['molecule_id'] = structures_df['molecule_name'].map(mol_ids)
    structures_df.to_csv(C.PROC_DATA_PATH + 'structures_proc_df.csv')
    return structures_df

def _clear_memory(var_strs):
    for var_str in var_strs: del globals()[var_str]
    gc.collect()


if __name__=='__main__':
    # import data
    all_df = pd.read_csv(C.RAW_DATA_PATH + 'train.csv', index_col=0)
    structures_df = pd.read_csv(C.RAW_DATA_PATH + 'structures.csv')
    all_df_mol_name = all_df['molecule_name'].unique()
    structures_df = structures_df[structures_df.molecule_name.isin(all_df_mol_name)]
    if 'id' in all_df.columns: all_df.drop(columns='id', inplace=True)

    # create molecules
    mols, mol_ids, mol_feats, xyzs, dist_matrices, graph_dist_matrices = \
        get_molecules()   # return mol ids、molecule level features、xyz coordinates、euclidean distance、graph distance
    # mol_feats:(dist_mean, dist_std, atom_label_number_mean) 
   
    # create and store features
    all_df = add_sc_features(all_df, structures_df, mol_feats, xyzs, dist_matrices, mol_ids)
    # add euclidean distance、normalized dist、atom radii and electronegativity、coupling types、
    # angle related features、molecule level features、atom counts per molecule、molecule ids
    all_df.to_csv(C.PROC_DATA_PATH + 'all_df.csv')
    store_train_and_test(all_df)
    _clear_memory(['all_df'])

    atom_df, bond_df = get_atom_and_bond_features(mols, mol_ids, dist_matrices)
    store_atom_and_bond_features(atom_df, bond_df)

    store_graph_distances(graph_dist_matrices)
    store_eucl_distances(dist_matrices, atom_df)   # only used for MPNN model

    structures_df = process_and_store_structures(structures_df, mol_ids)
    _, _ = get_all_cosine_angles(bond_df, structures_df, mol_ids, store=True)
