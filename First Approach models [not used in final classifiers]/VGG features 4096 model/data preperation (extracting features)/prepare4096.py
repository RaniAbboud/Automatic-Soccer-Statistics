import cv2
import numpy as np
from keras import Model
from keras.applications.vgg16 import VGG16, preprocess_input


def preprocess_img(im, desired_size=224):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    # im = cv2.flip(im, 1)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
    return preprocess_input(new_im)


def get_features(images, model, flip=False):
    features = []
    i = 0
    for img in images:
        cv2.rectangle(img, (40, 16), (70, 21), (255, 255, 255), 10)
        if flip:
            img = cv2.flip(img, 1)
        clean_img = preprocess_img(img)
        # if i == 500 or i == 700:
        #     cv2.imshow('im', clean_img)
        #     cv2.waitKey(0)
        clean_img = np.expand_dims(clean_img, axis=0)
        features += [model.predict(clean_img)]
        print(i)
        i += 1
    return np.array(features)


if __name__ == '__main__':
    # load original data
    data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\'

    images1 = np.load(data_folder+'image_array1.npy')
    labels1 = np.load(data_folder+'labels_array1.npy')

    print(images1.shape)
    images2 = np.load(data_folder + 'image_array2.npy')
    labels2 = np.load(data_folder + 'labels_array2.npy')

    print(images2.shape)
    images = np.concatenate((images1, images2), axis=0)
    print('concatenated images shape= ', images.shape)
    labels = np.concatenate((labels1, labels2), axis=0)

    # load the model
    vgg = VGG16()
    # re-structure the model
    # print(vgg.summary())
    # exit()
    vgg.layers.pop()
    vgg.layers.pop()
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-1].output)

    features1 = get_features(images1, vgg, flip=False)
    print('features1.shape=', features1.shape)
    features2 = get_features(images2, vgg, flip=True)
    print('features2.shape=', features2.shape)
    features = np.concatenate([features1, features2], axis=0)
    print('features.shape=', features.shape)
    np.save('vgg_features1_new4096.npy', features)
    np.save('labels1.npy', labels)
