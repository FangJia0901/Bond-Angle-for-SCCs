# Bond-Angle-for-SCCs
First：The data that support the findings of this study are available in https://www.kaggle.com/c/champs-scalar-coupling/data. we mainly focused on the features related to molecular structure in the feature engineering, ignoring the electronic or magnetic features such as Mulliken charge and magnetic shielding tensor. We believe that the structural characteristics of molecules are the result of the integrated manifestation of all the influencing factors. The influence of these factors is included in the structural information.

Second:Data preprocessing.All features which potentially influence SCCs are extracted and engineered with the guidance of prior physicochemical knowledge!The data can first be preprocessed through preprocess.py

Third:The model can be run via train.py.If you want to understand the structure of the model, you can refer to model.py.We train the model by parallel computation
.The GAANN model has been trained by Fastai library in PyTorch framework.

Fourth:In addition, we provide Gaann performance measurement code（such as scatter_type.py),drawing Molecular Structure code.

The code is being sorted out. If you have any questions, please don't hesitate to contact me.(email: fangjia0901@csu.edu.cn)
