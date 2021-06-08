import numpy as np
# from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils import to_categorical

from util import create_model


def AFCNN(x_train, y_train, x_test, y_test):
    class_num = 4  # 分类的种类

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], 1)

    # one-hot
    Y_train_one_hot = to_categorical(y_train, int(class_num))  # four labels
    Y_test_one_hot = to_categorical(y_test, int(class_num))  # four labels

    model = create_model()  # 创建CNN模型
    # model.fit(x_train, y_train,batch_size = 32, epochs = 10,verbose = 1,validation_data=(test_data,test_target),callbacks=callbacks_list)
    model.fit(x_train, Y_train_one_hot, batch_size=32, epochs=5, verbose=1, validation_data=(x_test, Y_test_one_hot))

    # test_pred,train_pred = stacking(train_data=x_train,train_target=Y_train_one_hot,test_data=x_test,test_target=Y_test_one_hot, n_fold=5)
    test_prob = model.predict(x_test)
    test_p = [np.argmax(item) for item in test_prob]  # 将矩阵按列合并
    score = model.evaluate(x_test, Y_test_one_hot, verbose=1)
    # print('acc==', score)
    # tmp_target = [np.argmax(item) for item in Y_test_one_hot]
    # print(confusion_matrix(tmp_target, test_p))
    # return test_p, test_prob
    return test_p
