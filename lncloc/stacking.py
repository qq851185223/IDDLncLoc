# coding=utf8

from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from evalutefunc import metric
from smote.smote_sampling import model
from sklearn.model_selection import LeaveOneOut
import tensorflow as tf

'''开启GPU加速'''
# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.allow_growth = True
# with tf.Session(config=config) as sess:
# sess = tf.Session(config=config)
# KTF.set_session(sess)

# data = np.load(r'C:\Users\Amber\Desktop\feature\feature_rfe.npy')
# label  =np.genfromtxt(r'C:\Users\Amber\Desktop\label.txt',delimiter=' ')

data = np.load(r'C:\Users\Amber\Desktop\feature\8merbinorfe.npy')
label = np.genfromtxt(r'C:\Users\Amber\Desktop\feature\label.txt')

label = label.astype('int64')

skf = StratifiedKFold(random_state=30).split(data, label)
cm_list = []
test = []
pred = []
i = 0
# 五折交叉验证
final_acc = 0

# for train_index,test_index in skf:
#     if i==2:
#         x_train, x_test = data[train_index], data[test_index]
#         y_train, y_test = label[train_index], label[test_index]
#         cm,y_true,y_pred,prob = model(x_train, x_test, y_train, y_test)
#         final_acc += accuracy_score(y_test,y_pred)
#         print('第'+str(i)+'折准确率：',accuracy_score(y_test,y_pred))
#         test.append(y_true)
#         pred.append(y_pred)
#         cm_list.append(cm)
#         np.save('data/cm-'+str(i),arr=cm)
#         np.save('data/prob-'+str(i),arr=prob)
#     i+=1

# jacknife(leaveone-out)
pred_list = []
y_true_list = []
# prob_list = []
loo = LeaveOneOut().split(data, label)
for train_index, test_index in loo:
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]
    y_true, y_pred = model(x_train, x_test, y_train, y_test)
    pred_list.append(y_pred[0])
    y_true_list.append(y_true[0])
    # prob.append(prob_list)
    # print('第'+str(i)+'折准确率：',accuracy_score(y_test,y_pred))
confusion_matrix = confusion_matrix(y_true=y_true_list, y_pred=pred_list)
print('acc=', accuracy_score(y_true=y_true_list, y_pred=pred_list))
Sn, Sp, Mcc = metric(confusion_matrix)
print('Sn=', Sn)
print('Sp=', Sp)
print('Mcc', Mcc)
# for train_index,test_index in skf:
#     if i == 0:
#         x_train, x_test = data[train_index], data[test_index]
#         y_train, y_test = label[train_index], label[test_index]
#         cm,y_true,y_pred,prob = model(x_train, x_test, y_train, y_test)
#         com = [test_index,y_test,y_pred]
#         np.save('data/compare'+str(i),com)
#     i+=1
