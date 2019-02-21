import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# load original data:
data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\'

train_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\frame_type_data\\Train\\'
val_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\frame_type_data\\Validation\\'

images1 = np.load(data_folder+'image_array1.npy')
images2 = np.load(data_folder+'image_array2.npy')
images = np.concatenate([images1, images2], axis=0)

labels1 = np.load(data_folder+'labels_array1.npy')
labels2 = np.load(data_folder+'labels_array2.npy')
labels = np.concatenate([labels1, labels2], axis=0)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=42)

i = 0
for image in X_train:
    cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
    label = {'Barca': 'On', 'Real': 'On', 'Other': 'Off'}[y_train[i]]
    cv2.imwrite(train_folder + label + '\\' + 'image_' + str(i) + '.jpg', image)
    i += 1

i = 0
for image in X_val:
    cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
    label = {'Barca': 'On', 'Real': 'On', 'Other': 'Off'}[y_val[i]]
    cv2.imwrite(val_folder + label + '\\' + 'image_' + str(i) + '.jpg', image)
    i += 1
