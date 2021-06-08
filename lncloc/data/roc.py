import numpy as np
from tensorflow.python.keras.utils import to_categorical
def softmax(z):
    z = np.array(z)
    z = z - max(z)
    z = np.exp(z)   #求e^zi值
    softmax_z = z / np.sum(z)
    return softmax_z

list = np.array([1,2,3,4])


prob0 = np.load('prob-0.npy')
prob1 = np.load('prob-1.npy')
prob2 = np.load('prob-2.npy')
prob3 = np.load('prob-3.npy')
prob4 = np.load('prob-4.npy')


label0 = np.load('label-0.npy')
label1 = np.load('label-1.npy')
label2 = np.load('label-2.npy')
label3 = np.load('label-3.npy')
label4 = np.load('label-4.npy')

label = np.hstack((label0,label1,label2,label3,label4))

prob_list = []
for i in range(prob0.shape[1]):
    prob = (prob0[0][i]*8+prob0[1][i])/9
    prob_list.append(prob)
for i in range(prob1.shape[1]):
    prob = (prob1[0][i]*8+prob1[1][i])/9
    prob_list.append(prob)
for i in range(prob2.shape[1]):
    prob = (prob2[0][i]*8+prob2[1][i])/9
    prob_list.append(prob)
for i in range(prob3.shape[1]):
    prob = (prob3[0][i]*8+prob3[1][i])/9
    prob_list.append(prob)
for i in range(prob4.shape[1]):
    prob = (prob4[0][i]*8+prob4[1][i])/9
    prob_list.append(prob)

prob_list = np.array(prob_list)
#print(prob_list)

label = to_categorical(label,4)
pred_0 = [y[0] for y in prob_list]  # 取出y中的一列
label_0 = [y[0] for y in label]
pred_1 = [y[1] for y in prob_list]  # 取出y中的一列
label_1 = [y[1] for y in label]
pred_2 = [y[2] for y in prob_list]  # 取出y中的一列
label_2 = [y[2] for y in label]
pred_3 = [y[3] for y in prob_list]  # 取出y中的一列
label_3 = [y[3] for y in label]

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from scipy import interp

fpr0, tpr0, thresholds_keras = roc_curve(label_0, pred_0)
fpr1, tpr1, thresholds_keras = roc_curve(label_1, pred_1)
fpr2, tpr2, thresholds_keras = roc_curve(label_2, pred_2)
fpr3, tpr3, thresholds_keras = roc_curve(label_3, pred_3)
label=label_0+label_1+label_2+label_3
label=np.array(label)
print(label)
pred = pred_0+pred_1+pred_2+pred_3
pred = np.array(pred)
print(pred)
fpr_micro, tpr_micro, _ = roc_curve(label.ravel(), pred.ravel())
all_fpr = np.unique(np.concatenate((fpr0,fpr1,fpr2,fpr3)))

mean_tpr = np.zeros_like(all_fpr)
mean_tpr +=interp(all_fpr,fpr0,tpr0)
mean_tpr +=interp(all_fpr,fpr1,tpr1)
mean_tpr +=interp(all_fpr,fpr2,tpr2)
mean_tpr +=interp(all_fpr,fpr3,tpr3)
fpr_macro = all_fpr
tpr_macro = mean_tpr/4
auc0 = auc(fpr0, tpr0)
print("AUC0 : ", auc0)
auc1 = auc(fpr1, tpr1)
print("AUC1 : ", auc1)
auc2 = auc(fpr2, tpr2)
print("AUC2 : ", auc2)
auc3 = auc(fpr3, tpr3)
print("AUC3 : ", auc3)
auc_micro = auc(fpr_micro,tpr_micro)
print("AUC_micro",auc_micro)
auc_macro = auc(fpr_macro,tpr_macro)
print("AUC_macro",auc_macro)


plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_micro, tpr_micro, label='micro-average ROC curve  (area = {:.3f})'.format(auc_micro))
plt.plot(fpr_micro, tpr_micro, label='macro-average ROC curve  (area = {:.3f})'.format(auc_macro))
plt.plot(fpr0, tpr0, label='ROC curve of class nucleus  (area = {:.3f})'.format(auc0))
plt.plot(fpr1, tpr1, label='ROC curve of class cytoplasm (area = {:.3f})'.format(auc1))
plt.plot(fpr2, tpr2, label='ROC curve of class ribosome (area = {:.3f})'.format(auc2))
plt.plot(fpr3, tpr3, label='ROC curve of class exosome (area = {:.3f})'.format(auc3))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve of lncLoc')
plt.legend(loc='best',fontsize=8)
plt.savefig('myroc.jpg',dpi=600)

plt.show()