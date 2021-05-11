import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import constants as C
import pdb

all_df = pd.read_csv(C.PROC_DATA_PATH + 'all_df.csv', index_col=0)
plt.figure(figsize=(10,4))
all_df.groupby('type').count()['molecule_id'].sort_values().plot(kind='bar',
                                                               figsize=(11, 8),
                                                               title='Count of Coupling Types')
plt.savefig('./Coupling_Count.png', dpi=800, bbox_inches='tight')
