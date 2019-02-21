from keras import Model
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from keras.optimizers import Adam
from accuracy import get_actual_labels, non_neutral_accuracy
from vgg_features_model.features4096.model4096_smaller import Model as final_model
import cv2
import numpy as np
from vgg_features_model.features4096.prepare4096 import get_features, preprocess_img


def get_image_features(images, vgg):
    return np.squeeze(get_features(np.array([preprocess_img(img) for img in images]), vgg, flip=False))


def predict_aux(images, model, vgg):
    features = get_image_features(images, vgg)
    return np.array(model.predict(features))
    # return np.array(model.predict(features, batch_size=16))


def calculate_test1_accuracy():
    # testing on the labelled Test1 video
    vgg_model = final_model().model
    # vgg_model.load_weights('weights-improvement-16-0.87.hdf5')
    vgg_model.load_weights('weights-improvement-257-0.80.hdf5')

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\test\\'
    test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data 2\\test1\\'
    X_test1 = np.load(test_folder + 'image_array.npy')
    y_test1 = np.load(test_folder + 'labels_array.npy')

    y_test1[y_test1 == 'Real'] = 0
    y_test1[y_test1 == 'Barca'] = 1
    y_test1[y_test1 == 'Other'] = 2
    y_test1 = to_categorical(y_test1, 3)

    X_val = X_test1
    y_val = y_test1

    vgg = VGG16()
    # re-structure the model
    vgg.layers.pop()
    vgg.layers.pop()
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-1].output)
    # testing on the labelled Test1 video
    predictions = get_actual_labels(predict_aux(X_val, vgg_model, vgg))

    print('normal test accuracy=', vgg_model.evaluate(get_image_features(X_val, vgg), y_val))
    print('non_neutral test accuracy=', non_neutral_accuracy(get_actual_labels(y_val), predictions))


def predict_labels():
    vgg_model = VGG16()
    # re-structure the model
    vgg_model.layers.pop()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-1].output)
    # testing on the labelled Test1 video
    vgg_model = final_model.Model().model
    # vgg_model.load_weights('weights-improvement-28-0.88.hdf5')
    vgg_model.load_weights('weights-improvement-06-0.88.hdf5')

    cap = cv2.VideoCapture('C:\\Users\\Rani\\Desktop\\AI project\\data (video)\\Test1.mp4')
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    i = 0
    labels = []
    while i < totalFrames:
        frames = []
        for j in range(i, i+100):
            ret, frame = cap.read()
            if ret:
                frames += [cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)]
            else:
                print('video read error (or finished reading)')
        predictions = get_actual_labels(predict_aux(frames, vgg_model, vgg))
        labels += list(predictions)
        print(i)
        i += 100
        if (i > 0) and (i % 1000) == 0:
            np.save('Test1_predicted_labels_new.npy', np.array(labels))

    np.save('Test1_predicted_labels_new.npy', np.array(labels))


if __name__ == '__main__':
    calculate_test1_accuracy()
    exit()
    # predict_labels()
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 600)
    fontScale = 2
    fontColor = (255, 255, 255)
    lineType = 2
    predicted = np.load('Test1_predicted_labels_new.npy')
    cap = cv2.VideoCapture('C:\\Users\\Rani\\Desktop\\AI project\\data (video)\\Test1.mp4')
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    j = 0
    while j < totalFrames:
        ret, frame = cap.read()
        cv2.putText(frame, predicted[j],
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.imshow('window', frame)
        cv2.waitKey(10)
        j += 1
