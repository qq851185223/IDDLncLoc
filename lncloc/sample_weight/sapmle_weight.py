import numpy as np
from sklearn.svm import SVC


def sample_weight(x_train, y_train, x_test, y_test):
    # 按照公式计算权重
    weight_0 = (1 / 156) * 655 / 4
    weight_1 = (1 / 426) * 655 / 4
    weight_2 = (1 / 43) * 655 / 4
    weight_3 = (1 / 30) * 655 / 4
    model = SVC(C=1000, gamma=0.001, max_iter=10000, class_weight={0: weight_0, 1: weight_1, 2: weight_2, 3: weight_3},
                probability=True)

    y_train = np.ravel(y_train)  # 向量转成数组
    sample_weights = []
    # 加入sample_weight
    for sw in range(len(y_train)):
        if y_train[sw] == 0:
            sample_weights.append(0.3)
        if y_train[sw] == 1:
            sample_weights.append(0.1)
        if y_train[sw] == 2:
            sample_weights.append(0.8)
        if y_train[sw] == 3:
            sample_weights.append(1)
    sample_weights = np.array(sample_weights)

    model.fit(X=x_train, y=y_train, sample_weight=sample_weights)

    # 训练集预测结果
    y_pred = model.predict(x_test)
    # y_prob = model.predict_proba(x_test)

    #print('终测试集的准确率 ： ', accuracy_score(y_test, y_pred))

    # return y_pred, y_prob
    return y_pred
