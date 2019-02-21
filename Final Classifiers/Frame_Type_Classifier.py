from keras.applications import VGG16
import cv2
import keras
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, regularizers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score


class FrameTypeClf:
    def __init__(self):
        self.img_shape = (288, 512, 3)
        self.weight_decay = 0.05

        vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=self.img_shape)

        # Freeze the VGG layers
        for layer in vgg_conv.layers:
            layer.trainable = False

        # Create the model
        self.model = Sequential()

        # Add the vgg convolutional base model
        self.model.add(vgg_conv)

        # Add new layers
        self.model.add(Flatten())
        self.model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Dense(2, activation='softmax'))

        # Show a summary of the model. Check the number of trainable parameters
        self.model.summary()
        # Compile the model
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=0.001),
                      metrics=['accuracy'])

    def load(self):
        saved_weights_path = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\fine_tuning_approach\\FrameType_weights\\'
        self.model.load_weights(saved_weights_path + 'weights-improvement-13-0.95.hdf5')
        # self.model.load_weights(saved_weights_path + 'weights-improvement-17-0.86.hdf5')
        # def get_labels(self, output):
        #     predicted_class_indices = np.argmax(output, axis=1)
        #     predicted_classes = ['Real' if x == 0 else 'Barca' for x in predicted_class_indices]
        #     return predicted_classes

    def preprocess_frames(self, frames):
        processed = []
        for frame in frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.rectangle(img, (40, 16), (70, 21), (255, 255, 255), 10)
            img = cv2.resize(img, (512, 288))
            img = preprocess_input(img)
            processed += [img]
        return np.array(processed)

    def get_labels(self, output):
        predicted_class_indices = np.argmax(output, axis=1)
        predicted_classes = ['On' if x == 0 else 'Off' for x in predicted_class_indices]
        return predicted_classes

    def predict_labels(self, videoPath):
        # cap = cv2.VideoCapture('C:\\Users\\Rani\\Desktop\\AI project\\data (video)\\Test'+str(num)+'.mp4')
        cap = cv2.VideoCapture(videoPath)
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        i = 0
        labels = []
        while i < totalFrames:
            frames = []
            for j in range(i, i + 100):
                ret, frame = cap.read()
                if ret:
                    frames += [frame]
                else:
                    print('video read error (or finished reading)')
            frames = self.preprocess_frames(frames)
            predictions = self.get_labels(self.model.predict(frames, batch_size=16))
            labels += list(predictions)
            print(i)
            i += 100

        np.save('Test'+'_predicted_labels_FrameTypeClf.npy', np.array(labels))
        return labels

    def get_accuracy(self):
        self.load()

        # test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\test2\\'
        test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\test2 strict labels\\'
        X_test = np.load(test_folder + 'image_array.npy')
        X_test_new = []
        for image in X_test:
            cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
            X_test_new += [preprocess_input(image)]
        X_test = np.array(X_test_new)

        y_test = np.load(test_folder + 'labels_array.npy')
        y_test[y_test != 'Other'] = 'On'
        y_test[y_test == 'Other'] = 'Off'

        # X_test = X_test[y_test != 'Other']  ##################### dropping Other frames
        # y_test = y_test[y_test != 'Other']  ##################### dropping Other frames

        pred = self.model.predict(X_test, batch_size=8)
        predictions = self.get_labels(pred)

        print('manual acc calculation: ', sum(predictions == y_test) / y_test.shape[0])
        print('sklearn acc calculation: ', accuracy_score(y_test, predictions))

    def train(self):
        saved_weights_path = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\fine_tuning_approach\\FrameType_weights\\'

        train_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\frame_type_data\\Train\\'
        val_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\frame_type_data\\Validation\\'

        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            zoom_range=[1, 1.25],
            preprocessing_function=preprocess_input)

        validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        # Change the batchsize according to your system RAM
        train_batchsize = 8
        val_batchsize = 8

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_shape[0], self.img_shape[1]),
            batch_size=train_batchsize,
            classes=['On', 'Off'],
            class_mode='categorical', shuffle=True)

        print(train_generator.class_indices)

        validation_generator = validation_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_shape[0], self.img_shape[1]),
            batch_size=val_batchsize,
            classes=['On', 'Off'],
            class_mode='categorical',
            shuffle=False)

        # Compile the model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(lr=0.0005),
                           metrics=['accuracy'])

        # code for saving weights
        weights_file_name = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(saved_weights_path + weights_file_name, monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        #########################
        #########################
        # model.load_weights(saved_weights_path + 'weights-improvement-01-0.84.hdf5')
        # Train the model
        history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples / train_generator.batch_size,
            epochs=15,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples / validation_generator.batch_size,
            verbose=1, callbacks=callbacks_list)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
        cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.waitKey(0)


def watch_demo(num, create=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 600)
    fontScale = 2
    fontColor = (255, 255, 255)
    lineType = 2
    predicted = np.load('Test'+str(num)+'_predicted_labels.npy')
    cap = cv2.VideoCapture('C:\\Users\\Rani\\Desktop\\AI project\\data (video)\\Test'+str(num)+'.mp4')
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if create:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280, 720))

    j = 0
    while j < totalFrames:
        ret, frame = cap.read()
        cv2.putText(frame, predicted[j],
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        if create:
            out.write(frame)
        cv2.imshow('window', frame)
        cv2.waitKey(20)
        j += 1
    cap.release()
    if create:
        out.release()


if __name__ == '__main__':
    model = FrameTypeClf()
    # model.train()
    model.get_accuracy()
    exit()

    model.load()
    model.predict_labels(2)
    watch_demo(2, create=True)
