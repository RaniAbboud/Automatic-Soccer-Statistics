import numpy as np
from keras.applications.vgg16 import VGG16
import cv2
from keras.applications.vgg16 import preprocess_input


def get_features(images, model, flip=False):
    features = []
    i = 0
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.rectangle(img, (40, 16), (70, 21), (255, 255, 255), 10)
        if flip:
            img = cv2.flip(img, 1)
        clean_img = preprocess_input(img)
        # if i == 500 or i == 700:
        #     cv2.imshow('im', clean_img)
        #     cv2.waitKey(0)
        clean_img = np.expand_dims(clean_img, axis=0)
        features += [model.predict(clean_img).flatten()]
        print(i)
        i += 1
    return np.squeeze(np.array(features))


if __name__ == '__main__':
    # load original data
    data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\'

    vgg = VGG16(include_top=False, input_shape=(288, 512, 3))

    images1 = np.load(data_folder+'image_array1.npy')
    labels1 = np.load(data_folder+'labels_array1.npy')
    features1 = get_features(images1, vgg, flip=False)

    print(features1.shape)
    images2 = np.load(data_folder + 'image_array2.npy')
    labels2 = np.load(data_folder + 'labels_array2.npy')
    features2 = get_features(images2, vgg, flip=True)

    print(features2.shape)
    features = np.concatenate((features1, features2), axis=0)
    print('concatenated features shape= ', features.shape)
    labels = np.concatenate((labels1, labels2), axis=0)

    ######################
    # for img in images[500:700]:
    #     cv2.imshow('im', img)
    #     cv2.waitKey(0)
    #     cv2.imshow('im', preprocess_input(img))
    #     cv2.waitKey(0)
    # exit()
    ######################
    # load the model
    # i=1
    # features = []
    # for img in images:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     clean_img = preprocess_input(cv2.rectangle(img, (40, 16), (70, 21), (255, 255, 255), 10))
    #     # if i==200 or i==250:
    #     #     cv2.imshow('img', clean_img)
    #     #     cv2.waitKey(0)
    #     #     #exit()
    #     clean_img = np.expand_dims(clean_img, axis=0)
    #     features += [model.predict(clean_img).flatten()]
    #     print(i)
    #     i += 1
    # features = np.squeeze(np.array(features))
    np.save('vgg_features1.npy', features)
    np.save('labels1.npy', labels)
