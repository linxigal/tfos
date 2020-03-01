# -*- coding:utf-8 _*-

from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense, Flatten, MaxPool2D
from tensorflow.python.keras.models import Sequential

from tfos.base import *


class AlexNet(BaseLayer):

    @ext_exception('AlexNet model')
    def add(self, input_shape, reshape, out_dense):
        model = create_alex_net(input_shape, out_dense, scale=True)
        return self.model2df(model)


def create_alex_net(input_shape, out_dense, scale=True):
    model = Sequential()

    # model.add(input_shape=(32, 32, 3)))
    # Layer 1
    # model.add( Conv2D(48, kernel_size=(3,3),strides=(1,1), activation='relu', padding='same', input_shape=x_train.shape[1:] ) )
    model.add(
        Conv2D(48, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(input_shape)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 2
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 3
    model.add(Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same'))

    # Layer 4
    model.add(Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 5
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    # Layer 6
    model.add(Dense(512, activation='tanh'))

    # Layer 7
    model.add(Dense(256, activation='tanh'))

    # Prediction
    model.add(Dense(out_dense, activation='softmax'))

    model.summary()

    return model

# outputdf = sqlc.createDataFrame([Row(model_type=0, model_config=json.dumps(model.get_config()))])
#
#
# outputRDD('<#zzjzRddName#>_dense', outputdf)
# outputdf.show()
