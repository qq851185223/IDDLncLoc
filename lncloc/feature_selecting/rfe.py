import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from scipy.io import loadmat
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#rfe选择的过程
kmer = np.load(r'8merbino.npy')
DACC = np.genfromtxt(r'C:\Users\Amber\Desktop\feature\DACC.txt',delimiter=' ')
CTD = np.genfromtxt(r'C:\Users\Amber\Desktop\feature\cpp.txt',delimiter=' ')
label = np.genfromtxt(r'C:\Users\Amber\Desktop\feature\label.txt',delimiter=' ')
data = np.hstack((kmer,DACC,CTD))

model = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=10000)
rfe = RFE(model,n_features_to_select=4000)

data = MinMaxScaler().fit_transform(data,label)
data = rfe.fit_transform(data,label)

print(cross_val_score(model,data,label))
np.save('../data/f_rfe',arr=data)