import pandas as pd
from matplotlib import pyplot as plt
import constants as C
import pdb
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
import matplotlib as mpl
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["axes.linewidth"] = 2

all_df = pd.read_csv(C.PROC_DATA_PATH + 'all_df.csv', index_col=0)
test_df = pd.read_csv(C.PROC_DATA_PATH + 'test_proc_df.csv', index_col=0)

target_df = pd.read_csv('/home/nesa/fangjia/champs-scalar-coupling/oofs/mol_transformer_v1_fold1-submit_targs.csv', index_col=0)
pred_df = pd.read_csv('/home/nesa/fangjia/champs-scalar-coupling/oofs/mol_transformer_v1_fold1-submit1.csv', index_col=0)
test_sc_type = test_df['type']
#pdb.set_trace()
test_targs_array = np.array([float(i[7:-1]) for i in target_df['scalar_coupling_constants'].values])
test_contrib_preds_array = np.array([float(i[7:-1]) for i in pred_df['scalar_coupling_constants'].values])

x = test_targs_array
y = test_contrib_preds_array
R_square = r2_score(x, y, multioutput= 'uniform_average')
print('The R^2 of prediction is:', R_square)

g =sns.JointGrid(x=x, y=y, height=10, ratio=4, xlim=(-25, 165), ylim=(-25, 165), space=0)
g.set_axis_labels(xlabel="True", ylabel="Predtion", fontsize=18)
sns.regplot(x=x, y=y, ax=g.ax_joint, scatter_kws={'s':8}, ci=99)
colors = ['#2F5597', '#A12525', '#61397F']
labels = ['1J', '2J', '3J']
for i in range(3):
    if i == 0:
        x = test_targs_array[test_sc_type<=1]
        y = test_contrib_preds_array[test_sc_type<=1]
        print(x.shape)
    elif i == 1:
        x = test_targs_array[(test_sc_type>=2)&(test_sc_type<=4)]
        y = test_contrib_preds_array[(test_sc_type>=2)&(test_sc_type<=4)]
        print(x.shape)
    else:
        x = test_targs_array[test_sc_type>=5]
        y = test_contrib_preds_array[test_sc_type>=5]
        print(x.shape)
    sns.histplot(x=x, bins=40, linewidth=0, ax=g.ax_marg_x, color=colors[i], kde=True, label=labels[i])
    sns.histplot(y=y, bins=40, linewidth=0, ax=g.ax_marg_y, color=colors[i], kde=True, label=labels[i], legend=True)
plt.legend(fontsize=20, markerscale=1)
plt.savefig("./scatter_MPNN_transformer_all_2.png".format(i), dpi=900, bbox_inches='tight')
