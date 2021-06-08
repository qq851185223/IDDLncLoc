import numpy as np
from evalutefunc import metric
cm = np.load('data/confuse_matrix.npy')#保存的结果的混淆矩阵

Sn,Sp,Mcc = metric(cm)

OA = (cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3])/655

print(Sn)
print(Sp)
print(Mcc)
print(OA)