import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import constants as C
import numpy as np
import pdb

all_df = pd.read_csv(C.PROC_DATA_PATH + 'all_df.csv', index_col=0)
TYPES = np.array(['1JHC', '1JHN', '2JHC', '2JHN', '2JHH', '3JHC', '3JHN', '3JHH'])
TYPES_MAP = {i: t for i, t in enumerate(TYPES)}
all_df['type'] = all_df['type'].map(TYPES_MAP)
all_df_1 = all_df[['type', 'scalar_coupling_constant']]
all_df_1.to_csv('./all_df_1.csv')
pdb.set_trace()
#fig, ax = plt.subplots(figsize = (11, 8))
x = all_df['type']
y = all_df['scalar_coupling_constant']
g =sns.JointGrid(data=all_df, height=10, ratio=4, space=0)
sns.violinplot(x=x, y=y, data=all_df, ax=g.ax_joint,
               linewidth=2, order=['1JHC', '1JHN', '2JHC', '2JHN', '2JHH', '3JHC', '3JHN', '3JHH'])
#g =sns.JointGrid(data=all_df,y=y,height=10, ratio=4, space=0)
sns.histplot(y, ax=g.ax_marg_y, color='red',linewidth=2, kde=True)
plt.xlabel('type')
#plt.title('Violinplot of scalar coupling constant by type')
plt.savefig('./Violinplot_4.png', dpi=800)
