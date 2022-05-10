import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def train_CNN(data_path):
    # 加载数据
    arr = np.load(data_path)
    img_list = arr['arr_0']
    label_list = arr['arr_1']
    # 将label转化为独热码
    onehot = OneHotEncoder()
    y_onehot = onehot.fit_transform(label_list.reshape(-1,1))
    y_onehot_arr = y_onehot.toarray()
    # 分成训练集和验证集
    x_train,x_test,y_train,y_test = train_test_split(img_list,y_onehot_arr,test_size=0.2,random_state=42)
    # 搭建LeNet5卷积神经网络
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', input_shape=(100,100,3), activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(166, activation='relu'),
        layers.Dense(22, activation='relu'),
        layers.Dense(3, activation='sigmoid')
    ])
    # 预览模型
    model.summary()
    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    # 训练模型
    history = model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test), batch_size=30, epochs=10)
    # 保存模型
    model.save('./face_mask_model.h5')
    return history


if __name__ == '__main__':
    data_path = './facemask_detection/data/imageData.npz'
    # 训练模型
    history = train_CNN(data_path)

    loss = history.history['loss']
    accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']

    plt.figure(1)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss Curve')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.figure(2)
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.title('Model acc Curve')
    plt.legend(['train_acc', 'val_acc'], loc='lower right')
    plt.show()