import keras
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, regularizers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self,train=True):
        self.num_classes = 3
        self.weight_decay = 0.9
        # self.weight_decay = 0
        self.x_shape = [9, 16, 512]

        self.model = self.build_model()

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Flatten(input_shape=self.x_shape))
        model.add(Dense(8, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
##########
        # model.add(Dense(16, kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        #
        # model.add(Dropout(0.5))
##########
        # ##########
        # model.add(Dense(16, kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        #
        # model.add(Dropout(0.5))
        # ##########
        # ##########
        # model.add(Dense(8, kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        #
        # model.add(Dropout(0.5))
        # ##########
        model.add(Dense(self.num_classes, kernel_regularizer=regularizers.l2(weight_decay)))
        # model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


if __name__ == '__main__':
    # load original labels:
    labels_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\'
    labels1 = np.load(labels_folder+'labels_array1.npy')
    labels2 = np.load(labels_folder + 'labels_array2.npy')
    labels = np.concatenate((labels1, labels2), axis=0)
    labels[labels == 'Real'] = 0
    labels[labels == 'Barca'] = 1
    labels[labels == 'Other'] = 2
    labels = keras.utils.to_categorical(labels, 3)

    # load vgg features:
    data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\vgg_features_model\\'
    features = np.load(data_folder+'vgg_features_new.npy')
    features = np.reshape(features, (5202, 9, 16, 512))  # remove added dimension

    model = Model().model

    adam = keras.optimizers.Adam(lr=0.000003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print(model.summary())
    # exit()
    # 1.17M params
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=42)

    saved_weights_path = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\vgg_features_model\\weights\\'
    weights_file_name = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(saved_weights_path + weights_file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.load_weights('weights-improvement-101-0.88.hdf5')
    model.fit(X_train, y_train, batch_size=512, epochs=10000, callbacks=callbacks_list,
                    validation_data=(X_val, y_val), verbose=2)
