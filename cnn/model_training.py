import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import argparse

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import buildPosesDataset as dataset

def train(args):

    epochSize = args["epoch"]
    model = args["model"].lower()

    batch_size = 64
    epochs = int(epochSize)
    learning_rate = 0.001
    model_name = "cnn/models/" + model + "_" + str(epochs) + ".h5"
    # model_name = "cnn/models/hand_detector_" + str(epochs) + ".h5"

    # input image dimensions
    img_rows, img_cols = 128, 128

    # the data, shuffled and split between train and test sets
    x_train, y_train, x_test, y_test = dataset.load_data(poses=["all"])

    num_classes = len(np.unique(y_test))

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    ####### Model structure #######
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=learning_rate),
                metrics=['accuracy'])

    ####### TRAINING #######
    hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))
    # Evaluation
    score = model.evaluate(x_test, y_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(model_name)

    # plotting the metrics
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="hand_detector", help="Model file to be created")
    ap.add_argument("-e", "--epoch", default="15", help="Epoch size")
    
    args = vars(ap.parse_args())
    train(args)
