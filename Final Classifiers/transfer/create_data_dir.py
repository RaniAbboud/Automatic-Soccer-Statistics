import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# load original data:
data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data 2\\'

train_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data 2\\data\\Train\\'
val_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data 2\\data\\Validation\\'

images1 = np.load(data_folder+'image_array1.npy')
labels1 = np.load(data_folder+'labels_array1.npy')

train_time = 30

images = images1[:int(images1.shape[0]*(train_time/45))]
labels = labels1[:int(images1.shape[0]*(train_time/45))]

test_images = images1[int(images1.shape[0]*(train_time/45)):]
test_labels = labels1[int(images1.shape[0]*(train_time/45)):]

test_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data 2\\1st half test\\'
np.save(test_dir+'test_images.npy', test_images)
np.save(test_dir+'test_labels.npy', test_labels)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.25, random_state=42, stratify=labels)

i = 0
for image in X_train:
    cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
    if y_train[i] == 'Other':
        i += 1
        continue
    cv2.imwrite(train_folder + y_train[i] + '\\' + 'image_' + str(i) + '.jpg', image)
    i += 1

i = 0
for image in X_val:
    cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
    if y_train[i] == 'Other':
        i += 1
        continue
    cv2.imwrite(val_folder + y_val[i] + '\\' + 'image_' + str(i) + '.jpg', image)
    i += 1
