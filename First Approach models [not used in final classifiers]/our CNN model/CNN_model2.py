from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.model_selection import train_test_split
from accuracy import non_neutral_accuracy
from accuracy import get_actual_labels

saved_weights_path = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\saved_weights\\'

class cifar100vgg:
    def __init__(self,train=True):
        self.num_classes = 3
        self.weight_decay = 0.005
        # self.weight_decay = 0
        self.x_shape = [288,512,3]

        self.model = self.build_model()

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(4, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(4, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(8, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(8, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(16, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
##########
        model.add(Dense(256, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.4))
##########
        ##########
        model.add(Dense(64, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.3))
        ##########
        ##########
        model.add(Dense(32, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.3))
        ##########
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


if __name__ == '__main__':

    data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\'

    images1 = np.load(data_folder+'image_array1.npy')
    labels1 = np.load(data_folder+'labels_array1.npy')
    ball_coords1 = np.load(data_folder+'ball_array1.npy')
    print(images1.shape)
    images2 = np.load(data_folder + 'image_array2.npy')
    labels2 = np.load(data_folder + 'labels_array2.npy')
    ball_coords2 = np.load(data_folder + 'ball_array2.npy')
    print(images2.shape)
    images = np.concatenate((images1, images2), axis=0)
    print('concatenated images shape= ', images.shape)
    labels = np.concatenate((labels1, labels2), axis=0)
    ball_coords = np.concatenate((ball_coords1, ball_coords2), axis=0)

    X = images
    labels[labels=='Real'] = 0
    labels[labels=='Barca'] = 1
    labels[labels=='Other'] = 2

    labels = keras.utils.to_categorical(labels, 3)

    X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.1, random_state=42)

    model = cifar100vgg()
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # print(model.model.summary())
    # exit()

    weights_file_name = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(saved_weights_path + weights_file_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.model.load_weights('weights-improvement-06-0.81.hdf5')
    ##############
    # print(model.model.summary())
    # exit()
    print('normal validation accuracy=', model.model.evaluate(X_val, y_val))
    print('non_neutral validation accuracy=', non_neutral_accuracy(get_actual_labels(y_val), get_actual_labels(model.model.predict(X_val))))
    exit()
    ##############
    model.model.fit(X_train, y_train, batch_size=64, epochs=50, callbacks=callbacks_list, validation_data=(X_val, y_val), verbose=2)
