from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from tensorflow.python.keras.utils import to_categorical

from AFCNN.AFCNN import AFCNN
from sample_weight.sapmle_weight import sample_weight
from smote.random_sampling import random_sampling
from util import dataset_split


def model(ori_x_train, x_test, ori_y_train, y_test):
    count = Counter(ori_y_train)
    print(count)

    pred_set = []  # 预测结果集
    # prob_set = []  # 概率集合，用于计算ROC

    methodlist =[sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,sample_weight,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN,AFCNN]
    # methodlist = [sample_weight, AFCNN]
    index = 0
    for method in methodlist:
        print('正在处理', index, '个子数据集')

        smote = SMOTE(random_state=20, sampling_strategy={1: count[1], 0: count[0], 2: 100, 3: 100})  # 过采样
        x_train, y_train = smote.fit_sample(ori_x_train, ori_y_train)
        print(Counter(y_train))

        c_data, rest_data, rest_label = dataset_split(x_train, y_train)  # 分离ncleus的样本
        num_c = len(c_data) // 3
        rs_c_data = random_sampling(c_data, num_c)  # 随机采样1/3的样本
        new_dataset = np.concatenate((rs_c_data, rest_data))  # 组建新的子集
        c_label = num_c * [1]
        new_label = np.concatenate((c_label, rest_label))
        print(Counter(new_label))
        # pred, prob = method(new_dataset, new_label, x_test, y_test)
        pred = method(new_dataset, new_label, x_test, y_test)
        pred_set.append(np.array(pred).reshape(-1, 1))
        # prob_set.append(prob)
        index += 1
    res = []
    i = 0
    for pred in pred_set:
        if i == 0:
            res = pred
        else:
            res = np.hstack((res, pred))
        i += 1

    y_pred = []
    for tmp in res:
        tmp_onehot = to_categorical(tmp, 4)
        pred_onehot = [0, 0, 0, 0]

        # tmp_index=0
        # for tmp in tmp_onehot:
        #     if tmp_index >= 20:
        #         pred_onehot += tmp*0.9
        #     else:
        #         pred_onehot += tmp
        #     tmp_index+=1
        # y_pred.append(np.argmax(pred_onehot))

        for tmp in tmp_onehot:
            pred_onehot += tmp
        y_pred.append(np.argmax(pred_onehot))

    # return confusion_matrix(y_test,y_pred),y_test,y_pred,prob_set
    return y_test, y_pred
