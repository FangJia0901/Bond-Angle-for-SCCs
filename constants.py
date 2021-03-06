import os
import numpy as np
from rdkit import Chem, RDConfig

TYPES = np.array(['1JHC', '1JHN', '2JHC', '2JHN', '2JHH', '3JHC', '3JHN', '3JHH'])
TYPES_MAP = {t: i for i, t in enumerate(TYPES)}
SYMBOLS = ['H', 'C', 'N', 'O', 'F']
DEGREES = [1, 2, 3, 4, 5]
HYBRIDIZATIONS = [Chem.rdchem.HybridizationType.SP, 
                  Chem.rdchem.HybridizationType.SP2, 
                  Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.UNSPECIFIED]
ATOMIC_RADIUS = {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71}
ELECTRO_NEG = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

# feature definition file
FDEF = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')

SC_EDGE_FEATS = ['type_0', 'type_1', 'type_2', 'type_3', 'type_4', 'type_5', 
                 'type_6', 'type_7', 'dist', 'dist_min_rad', 
                 'dist_electro_neg_adj', 'normed_dist', 'diangle', 'cos_angle', 'cos_angle0', 'cos_angle1']
SC_MOL_FEATS = ['type_0', 'type_1', 'type_2', 'type_3', 'type_4', 'type_5',
                 'type_6', 'type_7', 'dist', 'dist_min_rad', 
                 'dist_electro_neg_adj', 'normed_dist', 'diangle', 'cos_angle', 
                 'cos_angle0', 'cos_angle1', 'num_atoms', 'num_C_atoms', 
                 'num_F_atoms', 'num_H_atoms', 'num_N_atoms', 'num_O_atoms', 
                 'std_bond_length', 'ave_bond_length', 'ave_atom_weight']
ATOM_FEATS    = ['type_H', 'type_C', 'type_N', 'type_O', 'type_F', 'degree_1', 
                 'degree_2', 'degree_3', 'degree_4', 'degree_5', 'SP', 'SP2', 
                 'SP3', 'hybridization_unspecified', 'aromatic', 
                 'formal_charge', 'atomic_num', 'ave_bond_length', 'ave_neighbor_weight']
BOND_FEATS    = ['single', 'double', 'triple', 'aromatic', 'conjugated', 'in_ring', 'dist', 'normed_dist']

TARGET_COL   = 'scalar_coupling_constant'

N_TYPES            = 8
N_SC_EDGE_FEATURES = 16
#N_SC_EDGE_FEATURES = 12
N_SC_MOL_FEATURES  = 25
#N_SC_MOL_FEATURES  = 21
N_ATOM_FEATURES    = 19
N_BOND_FEATURES    = 8
MAX_N_ATOMS        = 29
MAX_N_SC           = 135
BATCH_PAD_VAL      = -999

N_SC               = 4658147
N_SC_TRAIN         = 3762130
N_SC_VAL           = 536619
N_SC_TEST          = 359398

N_MOLS             = 85003
N_TRAIN_MOLS       = 66951
N_VAL_MOLS         = 9552
N_TEST_MOLS        = 8500

N_FOLDS = 8

SC_MEAN = 16
SC_STD = 35
SC_FEATS_TO_SCALE = ['dist', 'dist_min_rad', 'dist_electro_neg_adj', 'num_atoms', 'num_C_atoms', 'num_F_atoms', 'num_H_atoms', 
                       'num_N_atoms', 'num_O_atoms', 'ave_bond_length', 'std_bond_length', 'ave_atom_weight']
ATOM_FEATS_TO_SCALE = ['atomic_num', 'ave_bond_length', 'ave_neighbor_weight']
BOND_FEATS_TO_SCALE = ['dist']

RAW_DATA_PATH = '/home/nesa/fangjia/champs-scalar-coupling/'
PATH = '/home/nesa/fangjia/champs-scalar-coupling/tmp/'
PROC_DATA_PATH = '/home/nesa/fangjia/champs-scalar-coupling/proc_data/'
SUB_PATH = '/home/nesa/fangjia/champs-scalar-coupling/submissions/'
OOF_PATH = '/home/nesa/fangjia/champs-scalar-coupling/oofs/'
