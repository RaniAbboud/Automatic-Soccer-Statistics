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


class TeamClf:
    def __init__(self):
        img_shape = (288, 512, 3)
        weight_decay = 0.5

        vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=img_shape)

        # Freeze the layers except the last 2 layers
        for layer in vgg_conv.layers[:-2]:
            layer.trainable = False

        # Create the model
        self.model = Sequential()

        # Add the vgg convolutional base model
        self.model.add(vgg_conv)

        # Add new layers
        self.model.add(Flatten())
        self.model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Dense(2, activation='softmax'))

        # Show a summary of the model. Check the number of trainable parameters
        # self.model.summary()
        # Compile the model
        self.model.compile(loss='categorical_crossentropy',
                      # optimizer=optimizers.RMSprop(lr=1e-4),
                      optimizer=keras.optimizers.Adam(lr=0.00002),
                      metrics=['accuracy'])

    def load(self):
        saved_weights_path = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\fine_tuning_approach\\weights\\'
        self.model.load_weights(saved_weights_path + 'weights-improvement-11-0.83.hdf5')

    def get_labels(self, output):
        predicted_class_indices = np.argmax(output, axis=1)
        predicted_classes = ['Real' if x == 0 else 'Barca' for x in predicted_class_indices]
        return predicted_classes

    def preprocess_frames(self, frames):
        processed = []
        for frame in frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.rectangle(img, (40, 16), (70, 21), (255, 255, 255), 10)
            img = cv2.resize(img, (512, 288))
            img = preprocess_input(img)
            processed += [img]
        return np.array(processed)

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

        np.save('Test'+'_predicted_labels_TeamClf.npy', np.array(labels))
        return labels

    def get_accuracy(self):
        self.load()

        # test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\test2\\'
        test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\clasico2 labelled data\\'

        X_test = np.load(test_folder + 'image_array.npy')
        X_test_new = []
        for image in X_test:
            cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
            X_test_new += [preprocess_input(image)]
        X_test = np.array(X_test_new)

        y_test = np.load(test_folder + 'labels_array.npy')
        X_test = X_test[y_test != 'Other']  ##################### dropping Other frames
        y_test = y_test[y_test != 'Other']  ##################### dropping Other frames

        # #####
        # X_test = X_test[:120] #out of 180
        # y_test = y_test[:120] #out of 180
        # #####

        pred = self.model.predict(X_test, batch_size=8)
        predictions = self.get_labels(pred)

        # ###################################### only for testing, to delete
        #
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # bottomLeftCornerOfText = (10, 600)
        # fontScale = 2
        # fontColor = (255, 255, 255)
        # lineType = 2
        # X_test = np.load(test_folder + 'image_array.npy')
        # y_test = np.load(test_folder + 'labels_array.npy')
        # X_test = X_test[y_test != 'Other']
        # i = 0
        # for frame in X_test:
        #     im = frame.copy()
        #     cv2.putText(im, predictions[i],
        #                 bottomLeftCornerOfText,
        #                 font,
        #                 fontScale,
        #                 fontColor,
        #                 lineType)
        #     cv2.imshow('window', im)
        #     print(predictions[i])
        #     cv2.waitKey(500)
        #     i += 1
        # ######################################

        print('manual acc calculation: ', sum(predictions == y_test) / y_test.shape[0])
        print('sklearn acc calculation: ', accuracy_score(y_test, predictions))


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
    model = TeamClf()
    model.load()

    ##########################
    train_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\data\\Train\\'
    val_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\data\\Validation\\'

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        zoom_range=[1, 1.25],
        preprocessing_function=preprocess_input)

    validation_datagen = ImageDataGenerator(horizontal_flip=True,
                                            preprocessing_function=preprocess_input)

    train_batchsize = 8
    val_batchsize = 8
    img_shape = (288, 512, 3)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_shape[0], img_shape[1]),
        batch_size=train_batchsize,
        classes=['Real', 'Barca'],
        class_mode='categorical', shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(img_shape[0], img_shape[1]),
        batch_size=val_batchsize,
        classes=['Real', 'Barca'],
        class_mode='categorical',
        shuffle=False)

    print(model.model.evaluate_generator(train_generator))
    print(model.model.evaluate_generator(validation_generator))
    exit()
    ##########################

    model.get_accuracy()
    exit()
    model.predict_labels(2)
    watch_demo(2, create=True)
