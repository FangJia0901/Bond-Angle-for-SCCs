import pandas as pd
import constants as C
import pdb
import numpy as np

all_df = pd.read_csv(C.PROC_DATA_PATH + 'all_df.csv', index_col=0)
test_df = pd.read_csv(C.PROC_DATA_PATH + 'test_proc_df.csv', index_col=0)

target_df = pd.read_csv('/home/nesa/fangjia/champs-scalar-coupling/oofs/mol_transformer_v1_fold1-submit_targs.csv', index_col=0)
pred_df = pd.read_csv('/home/nesa/fangjia/champs-scalar-coupling/oofs/mol_transformer_v1_fold1-submit.csv', index_col=0)

#Generate the relationship between ID and Name
mol_id_name = all_df[['molecule_id', 'molecule_name']]
mol_id_name.molecule_id = (all_df['molecule_id']).astype(str)
mol_id_name.molecule_name = (all_df['molecule_name']).astype(str)
mol_id_name_dict = mol_id_name.set_index('molecule_id')['molecule_name'].to_dict()

test_df_index = test_df[['atom_index_0', 'atom_index_1','atom_0', 'atom_1']]
test_type = test_df['type']
TYPES = np.array(['1JHC', '1JHN', '2JHC', '2JHN', '2JHH', '3JHC', '3JHN', '3JHH'])
#Generate the molecular Name
test_mol_id = test_df['molecule_id']
test_df['molecule_name'] = test_df['molecule_id'].astype(str).map(mol_id_name_dict)
test_mol_name = test_df['molecule_name']
print(pred_df.shape, target_df.shape, test_type.shape, test_mol_name.shape, test_df_index.shape)

pred_df_list_1J, target_df_list_1J, test_type_list_1J, test_mol_name_list_1J, test_df_index_list_1J = [], [], [], [], []
pred_df_list_2J, target_df_list_2J, test_type_list_2J, test_mol_name_list_2J, test_df_index_list_2J = [], [], [], [], []
pred_df_list_3J, target_df_list_3J, test_type_list_3J, test_mol_name_list_3J, test_df_index_list_3J = [], [], [], [], []
for i in range(pred_df.shape[0]):
    if test_type.iloc[i] <= 1:
        pred_df_list_1J.append(float(pred_df.iloc[i].item()[7:-1]))
        target_df_list_1J.append(float(target_df.iloc[i].item()[7:-1]))
        test_type_list_1J.append(TYPES[test_type.iloc[i]])
        test_mol_name_list_1J.append(test_mol_name.iloc[i])
        test_df_index_list_1J.append(test_df_index.iloc[i])
    if (test_type.iloc[i] >= 2) & (test_type.iloc[i] <= 4):
        pred_df_list_2J.append(float(pred_df.iloc[i].item()[7:-1]))
        target_df_list_2J.append(float(target_df.iloc[i].item()[7:-1]))
        test_type_list_2J.append(TYPES[test_type.iloc[i]])
        test_mol_name_list_2J.append(test_mol_name.iloc[i])
        test_df_index_list_2J.append(test_df_index.iloc[i])
    if test_type.iloc[i] >= 5:
        pred_df_list_3J.append(float(pred_df.iloc[i].item()[7:-1]))
        target_df_list_3J.append(float(target_df.iloc[i].item()[7:-1]))
        test_type_list_3J.append(TYPES[test_type.iloc[i]])
        test_mol_name_list_3J.append(test_mol_name.iloc[i])
        test_df_index_list_3J.append(test_df_index.iloc[i])
print(len(pred_df_list_1J), len(pred_df_list_2J), len(pred_df_list_3J))

pred_df_list_1J_df = pd.DataFrame(data={'Prediction':pred_df_list_1J,
                                    'Target':target_df_list_1J,
                                    'Type':test_type_list_1J,
                                    'Mol_Name':test_mol_name_list_1J,
                                    'Atom_index_0':[test_df_index_list_1J[i][0] for i in range(len(test_df_index_list_1J))],
                                    'Atom_index_1':[test_df_index_list_1J[i][1] for i in range(len(test_df_index_list_1J))],
                                    'Atom_0':[test_df_index_list_1J[i][2] for i in range(len(test_df_index_list_1J))],
                                    'Atom_1':[test_df_index_list_1J[i][3] for i in range(len(test_df_index_list_1J))]})
#pdb.set_trace()
pred_df_list_1J_df.to_csv('./pred_df_list_1J_df.csv')
pred_df_list_2J_df = pd.DataFrame(data={'Prediction':pred_df_list_2J,
                                    'Target':target_df_list_2J,
                                    'Type':test_type_list_2J,
                                    'Mol_Name':test_mol_name_list_2J,
                                    'Atom_index_0':[test_df_index_list_2J[i][0] for i in range(len(test_df_index_list_2J))],
                                    'Atom_index_1':[test_df_index_list_2J[i][1] for i in range(len(test_df_index_list_2J))],
                                    'Atom_0':[test_df_index_list_2J[i][2] for i in range(len(test_df_index_list_2J))],
                                    'Atom_1':[test_df_index_list_2J[i][3] for i in range(len(test_df_index_list_2J))]})
pred_df_list_2J_df.to_csv('./pred_df_list_2J_df.csv')
pred_df_list_3J_df = pd.DataFrame(data={'Prediction':pred_df_list_3J,
                                    'Target':target_df_list_3J,
                                    'Type':test_type_list_3J,
                                    'Mol_Name':test_mol_name_list_3J,
                                    'Atom_index_0':[test_df_index_list_3J[i][0] for i in range(len(test_df_index_list_3J))],
                                    'Atom_index_1':[test_df_index_list_3J[i][1] for i in range(len(test_df_index_list_3J))],
                                    'Atom_0':[test_df_index_list_3J[i][2] for i in range(len(test_df_index_list_3J))],
                                    'Atom_1':[test_df_index_list_3J[i][3] for i in range(len(test_df_index_list_3J))]})
pred_df_list_3J_df.to_csv('./pred_df_list_3J_df.csv')
#pdb.set_trace()
print("111")
