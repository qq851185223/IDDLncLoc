from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
# load label

#Y = np.genfromtxt(r'C:\Users\Amber\Desktop\label.txt',delimiter=' ')
# datadict = loadmat(r"D:\matlab\bin\8merbino.mat")
# print(datadict)
# data = datadict['kmerCLorder']

# data = np.genfromtxt(r'C:\Users\Amber\Desktop\gragh.txt',delimiter=' ')
# Y = [0]*426+[1]*30+[2]*156+[3]*43
# Y = np.array(Y)
# Y= Y.reshape((655))
data = np.load('../data/8merbino.npy')
print(data.shape)
label = np.genfromtxt('../data/label.txt',delimiter=' ')
label=np.array(label)
label = label.reshape((len(label)))
fea_len = data.shape[1]
print(data)
percent = [x / 100 for x in range(2, 102, 2)]
Num = []
for i in range(len(percent)):
    Num.append(math.ceil(fea_len * percent[i]))

for i in range(len(Num)):
    print(i,Num[i])
ifs8merout1 = list()
ifs8merout2 = list()
ifs8merout3 = list()
num_folds = 5
for i in range(len(Num)):
    X = data[:, 0:Num[i]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(X)

    # resultIgs = []
    # resultIparams = []
    # for x in range(5):
    #     num_folds = 5
    #     kf = KFold(n_splits=num_folds, shuffle=True)
    #     param_grid = {}
    #     param_grid['C'] = [1000]
    #     param_grid['solver'] = [ 'lbfgs']
    #     param_grid['multi_class'] = ['multinomial']
    #     model = LogisticRegression()
    #     grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kf)
    #     grid_result = grid.fit(rescaledX, label)
    #     print('最优 : %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
    #     resultIgs.append(grid_result.best_score_)
    #     resultIparams.append(grid_result.best_params_)
    # IFS00 = list(zip(resultIgs, resultIparams))
    # IFS01 = np.argsort(-np.array(resultIgs))
    # params = resultIparams[IFS01[0]]

    resultI = []
    for j in range(5):
        kf = KFold(n_splits=num_folds, shuffle=True)
        #model = LogisticRegression(**params)
        model = LogisticRegression(C=1000,solver='lbfgs',multi_class='multinomial',max_iter=10000)
        resultj = cross_val_score(model, rescaledX, label, cv=kf)
        resultI.append(resultj.mean())
    acci = np.array(resultI)
    print('加入第%d个准确率 : %s' % (i, acci.mean()))
    ifs8merout1.append(Num[i])
    #ifs8merout2.append(params)
    ifs8merout3.append(acci.mean())

IFS00 = np.vstack((ifs8merout1,ifs8merout3)).T
np.save('cdkmerifs',IFS00)
IFS01 = np.argsort(-IFS00[:, 1])
IFSorder0 = IFS00[IFS01].tolist()
ifs8perorder0 = IFSorder0
np.save('cdkmerifsoder',ifs8perorder0)