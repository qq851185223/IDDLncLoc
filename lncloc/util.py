import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.layers import Dense,Dropout,Layer,Lambda,Conv2D,Concatenate,multiply,MaxPooling2D,Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import backend as K
from AFCNN.loss import MyLoss
from sklearn import clone
def dataset_split(data,label):
    c_data = []
    rest_data =[]
    rest_label =[]
    for i in range(len(label)):
        if label[i]==1:
            c_data.append(data[i])
        else:
            rest_data.append(data[i])
            rest_label.append(label[i])

    return c_data,rest_data,rest_label

def stacking(model, train_data, train_target, test_data,test_target,n_fold):
    """
    :param model:  模型算法
    :param train_data:  训练集(不含带预测的目标特征)
    :param train_target:  需要预测的目标特征
    :param test_data:   测试集
    :param n_fold:   交叉验证的折数
    :return:
    """
    train_data = pd.DataFrame(train_data)
    train_target = pd.DataFrame(train_target)
    test_data = pd.DataFrame(test_data)
    skf = StratifiedKFold(n_splits=n_fold,shuffle=True)  # StratifiedKFold 默认分层采样
    train_pred = np.zeros((train_data.shape[0], 1), int)  # 存储训练集预测结果
    test_pred = np.zeros((test_data.shape[0], 1), int)  # 存储测试集预测结果 行数：len(test_data) ,列数：1列
    for skf_index, (train_index, val_index) in enumerate(skf.split(train_data, train_target)):
        print('第 ', skf_index + 1, ' 折交叉验证开始... ')
        # 训练集划分
        new_model = clone(model)
        # print('pre-model',model)
        # print('new-model',new_model)
        x_train, x_val = train_data.iloc[train_index], train_data.iloc[val_index]
        y_train, y_val = train_target.iloc[train_index], train_target.iloc[val_index]
        # 模型构建
        y_train = np.ravel(y_train)  # 向量转成数组
        new_model.fit(X=x_train, y=y_train)
        # 模型预测
        accs = accuracy_score(y_val, new_model.predict(x_val))
        print('第 ', skf_index + 1, ' 折交叉验证 :  accuracy ： ', accs)

        # 训练集预测结果
        val_pred = new_model.predict(x_val)
        for i in range(len(val_index)):
            train_pred[val_index[i]] = val_pred[i]
        # 保存测试集预测结果

        print('第 ', skf_index + 1, ' 折accuracy ： ', accuracy_score(test_target, new_model.predict(test_data)))
        test_pred = np.column_stack((test_pred, new_model.predict(test_data)))  # 将矩阵按列合并

    test_pred_mean = np.mean(test_pred, axis=1)  # 按行计算均值(会出现小数)
    test_pred_mean = pd.DataFrame(test_pred_mean)  # 转成DataFrame
    test_pred_mean = test_pred_mean.apply(lambda x: round(x))  # 小数需要四舍五入成整数
    test_set = np.ravel(test_pred_mean)
    return test_set.reshape(test_set.shape[0],1), np.array(train_pred)
class SpatialAttention(Layer):

    def __init__(self):
        super().__init__()

    def __call__(self, input_feature):
        kernel_size = 7

        # if K.image_data_format() == "channels_first":
        #     channel = input_feature._keras_shape[1]
        #     cbam_feature = Permute((2, 3, 1))(input_feature)
        # else:
        #     channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

        avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
        #assert avg_pool._keras_shape[-1] == 1
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
        #assert max_pool._keras_shape[-1] == 1
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        #assert concat._keras_shape[-1] == 2
        cbam_feature = Conv2D(filters=1,
                              kernel_size=kernel_size,
                              activation='hard_sigmoid',
                              strides=1,
                              padding='same',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        #assert cbam_feature._keras_shape[-1] == 1

        #if K.image_data_format() == "channels_first":
        #    cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])

def create_model():
    #CNN模型
    model = Sequential()
    model.add(Conv2D(8, (1, 4), activation='relu', input_shape=(1, 4144, 1)))
    # model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(SpatialAttention())
    #model.add(Dropout(0.5))
    model.add(Conv2D(16, (1, 4), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 2)))
    # model.add(Conv2D(150, (1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    #model.add(SpatialAttention())
    #model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dense(int(4), activation='softmax'))

    model.compile(loss=MyLoss().loss_func, optimizer='adam', metrics=['accuracy'])#加入focal_loss
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model