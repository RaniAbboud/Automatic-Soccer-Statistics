from keras.applications import VGG16
from tensorflow.python.estimator import keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from accuracy import get_actual_labels, non_neutral_accuracy
# from vgg_features_model import model as final_model
from vgg_features_model import multiple_data_model_loadingdata as final_model
import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input
from vgg_features_model.backup.model_smaller2 import Model as model77


def get_features(images, model, flip=False):
    features = []
    i = 0
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.rectangle(img, (40, 16), (70, 21), (255, 255, 255), 10)
        img = cv2.resize(img, (512, 288)) #############################
        if flip:
            img = cv2.flip(img, 1)
        clean_img = preprocess_input(img)
        # cv2.imshow('win', clean_img)
        # cv2.waitKey(0)

        # if i == 500 or i == 700:
        #     cv2.imshow('im', clean_img)
        #     cv2.waitKey(0)
        clean_img = np.expand_dims(clean_img, axis=0)
        # features += [model.predict(clean_img).flatten()]
        features += [model.predict(clean_img)]
        # print(i)
        i += 1
    return np.squeeze(np.array(features))


# def get_image_features(images, vgg):
#     # load the VGG base model
#     images = np.array([preprocess_input(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (512, 288))) for image in images])
#     # images = np.array([cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (512, 288)) for image in images])
#
#     # vgg = VGG16(include_top=False, input_shape=(288, 512, 3))
#
#     features = vgg.predict(images, batch_size=1)
#     features = [x.flatten() for x in features]
#
#     return np.array(features)


def predict_aux(images, model, vgg):
    features = get_features(images, vgg)
    # features = get_image_features(images, vgg)

    # return np.array(model.predict(features, batch_size=16))
    return np.array(model.predict(features, batch_size=256))


def calculate_test1_accuracy():
    # testing on the labelled Test1 video
    # vgg_model = final_model.Model().model
    vgg_model = model77().model
    # vgg_model.load_weights('weights-improvement-16-0.87.hdf5')
    # vgg_model.load_weights('weights-improvement-555-0.87.hdf5')
    vgg_model.load_weights(r'C:\Users\Rani\Desktop\AI Pycharm Project\vgg_features_model\backup\smaller 77\\'+'weights-improvement-607-0.88.hdf5')

    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\test\\'
    test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\test2\\'
    X_test1 = np.load(test_folder + 'image_array.npy')
    y_test1 = np.load(test_folder + 'labels_array.npy')

    y_test1[y_test1 == 'Real'] = 0
    y_test1[y_test1 == 'Barca'] = 1
    y_test1[y_test1 == 'Other'] = 2
    y_test1 = to_categorical(y_test1, 3)

    # #####################
    # # for layer in vgg_model.layers[:-3]:
    # #     print(layer)
    # #     layer.trainable = False
    #
    # X_train = X_test1[:200]
    # y_train = y_test1[:200]
    # X_val = X_test1[200:250]
    # y_val = y_test1[200:250]
    # vgg = VGG16(include_top=False, input_shape=(288, 512, 3))
    # vgg_model.fit(get_features(X_train, vgg, flip=True), y_train, epochs=200, validation_data=(get_features(X_val, vgg, flip=True), y_val), verbose=2)
    #
    # X_test1 = X_test1[250:]
    # y_test1 = y_test1[250:]
    # #####################

    X_val = X_test1
    y_val = y_test1

    vgg = VGG16(include_top=False, input_shape=(288, 512, 3))
    predictions = get_actual_labels(predict_aux(X_val, vgg_model, vgg))

    # print('normal test accuracy=', vgg_model.evaluate(get_image_features(X_val, vgg), y_val, batch_size=4))
    print('normal test accuracy=', vgg_model.evaluate(get_features(X_val, vgg), y_val, batch_size=4))
    print('non_neutral test accuracy=', non_neutral_accuracy(get_actual_labels(y_val), predictions))


def predict_labels():
    vgg = VGG16(include_top=False, input_shape=(288, 512, 3))
    # testing on the labelled Test1 video
    # vgg_model = final_model.Model().model
    vgg_model = model77().model
    # vgg_model.load_weights('weights-improvement-28-0.88.hdf5')
    # vgg_model.load_weights('weights-improvement-555-0.87.hdf5')
    vgg_model.load_weights(r'C:\Users\Rani\Desktop\AI Pycharm Project\vgg_features_model\backup\smaller 77\\'+'weights-improvement-607-0.88.hdf5')

    cap = cv2.VideoCapture("C:\\Users\\Rani\\Desktop\\AI project\\data (video)\\Test2.mp4")
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    i = 0
    labels = []
    while i < totalFrames:
        frames = []
        for j in range(i, i+100):
            ret, frame = cap.read()
            if ret:
                frames += [frame]
            else:
                print('video read error (or finished reading)')
        predictions = get_actual_labels(predict_aux(frames, vgg_model, vgg))
        labels += list(predictions)
        print(i)
        i += 100
        if (i > 0) and (i % 1000) == 0:
            np.save('Test2_predicted_labels_new77.npy', np.array(labels))

    np.save('Test2_predicted_labels_new77.npy', np.array(labels))


if __name__ == '__main__':
    calculate_test1_accuracy()
    exit()
    # predict_labels()
    # exit()
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 600)
    fontScale = 2
    fontColor = (255, 255, 255)
    lineType = 2
    predicted = np.load('Test2_predicted_labels_new77.npy')
    cap = cv2.VideoCapture('C:\\Users\\Rani\\Desktop\\AI project\\data (video)\\Test2.mp4')
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
        cv2.waitKey(20)
        j += 1
