import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
sns.set()

x_embedding_1 = torch.load("/home/nesa/fangjia/kaggle-champs-master--0/x_embedding-1.pt").cpu()
x_embedding_0 = torch.load("/home/nesa/fangjia/kaggle-champs-master--0/x_embedding-0.pt").cpu()
print(x_embedding_1.shape, x_embedding_0.shape)
x_embedding = torch.cat([x_embedding_1, x_embedding_0], dim=0)
print(x_embedding.shape)
pca = PCA(n_components=2, whiten=False)
embedding = pca.fit_transform(np.array(x_embedding.detach()))
print(embedding.shape)

sc_types_embedding_1 = torch.load("/home/nesa/fangjia/kaggle-champs-master--0/sc_types_embedding-1.pt").cpu()
sc_types_embedding_0 = torch.load("/home/nesa/fangjia/kaggle-champs-master--0/sc_types_embedding-0.pt").cpu()
sc_types_embedding = torch.cat([sc_types_embedding_1, sc_types_embedding_0], dim=0)
print(sc_types_embedding.shape)
sc_types_embedding = sc_types_embedding.numpy()
print(sc_types_embedding.shape)

#colors = ['blue', 'blueviolet', 'brown', 'green', 'cadetblue', 'chartreuse', 'red', 'yellow'] #'#91D1C2'
#colors = ['#8491B4','#4DBBD5','#3C5488','#DC0000','#F39B7F','cadetblue','#B09C85','#91D1C2']  #'#00A087',
colors = ['#631779','#4DBBD5','#00A087','#DC0000','#F39B7F','#3C5488','#EFC000','#7E6148']
Label = ['1JHC', '1JHN', '2JHC', '2JHN', '2JHH', '3JHC', '3JHN', '3JHH']

fig1 = plt.figure(1, figsize=(6, 4))
X_0, Y_0 = [], []
X_1, Y_1 = [], []
X_2, Y_2 = [], []

X_4, Y_4 = [], []
X_5, Y_5 = [], []
X_6, Y_6 = [], []

for index in range(8):
    X = embedding[sc_types_embedding == np.array(index)][:, 0]
    Y = embedding[sc_types_embedding == np.array(index)][:, 1]
    plt.scatter(X, Y, c=colors[index], cmap='brg', s=5, alpha=0.8, marker='o', linewidth=0, label=Label[index])
    ax = sns.kdeplot(x=X, y=Y, shade=False, levels=1, color=colors[index], thresh=.5, alpha=1, linewidths=1)
    if index <= 1:
        X_0.append(X)
        Y_0.append(Y)
    elif index >= 5:
        X_1.append(X)
        Y_1.append(Y)
    elif 1:
        X_2.append(X)
        Y_2.append(Y)
    ax.patch.set_facecolor('white')
print(len(X_0), len(X_1), len(X_2))
print(np.array(X_0)[0].shape, np.array(X_1)[0].shape, np.array(X_2)[0].shape)
print(np.array(X_0)[1].shape, np.array(X_1)[1].shape, np.array(X_2)[1].shape)
print(np.array(X_1)[2].shape, np.array(X_2)[2].shape)

X_4 = np.append(X_0[0], X_0[1])
X_5 = np.concatenate((X_1[0], X_1[1], X_1[2]), axis=0)
X_6 = np.concatenate((X_2[0], X_2[1], X_2[2]), axis=0)
Y_4 = np.append(Y_0[0], Y_0[1])
Y_5 = np.concatenate((Y_1[0], Y_1[1], Y_1[2]), axis=0)
Y_6 = np.concatenate((Y_2[0], Y_2[1], Y_2[2]), axis=0)

for i in range(3):
    if i == 0:
        x, y = X_4, Y_4
        print(x.shape, y.shape)
    elif i == 1:
        x, y = X_5, Y_5
        print(x.shape, y.shape)
    else:
        print('I Love you!')
        x, y = X_6, Y_6
        print(x.shape, y.shape)
    ax = sns.kdeplot(x=x, y=y, shade=True, levels=2, color=colors[i],thresh=.3, alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right', fontsize=6)
plt.savefig('./scatter_types_212.png', dpi=900, bbox_inches='tight', transparent=False)
plt.show()
#handles,labels = ax.get_legend_handles_labels()
#plt.legend(handles, labels = Label[2:], loc='upper right', fontsize=6)
