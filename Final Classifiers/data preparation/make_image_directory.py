import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# load original data:
data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\'
test_data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\test\\'
output_folder = data_folder+'image_files\\'
train_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\data\\Train\\'
val_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\data\\Validation\\'
test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\data\\Test\\'

images1 = np.load(data_folder+'image_array1.npy')
images2 = np.load(data_folder+'image_array2.npy')
images = np.concatenate([images1, images2], axis=0)

labels1 = np.load(data_folder+'labels_array1.npy')
labels2 = np.load(data_folder+'labels_array2.npy')
labels = np.concatenate([labels1, labels2], axis=0)

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=42)
X_test = np.load(test_data_folder+'image_array.npy')
y_test = np.load(test_data_folder+'labels_array.npy')

# i = 0
# for image in X_train:
#     cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
#     cv2.imwrite(train_folder + y_train[i] + '\\' + 'image_' + str(i) + '.jpg', image)
#     i += 1
#
# i = 0
# for image in X_val:
#     cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
#     # image = cv2.flip(image, 1)
#     cv2.imwrite(val_folder + y_val[i] + '\\' + 'image_' + str(i) + '.jpg', image)
#     i += 1

i = 0
for image in X_test:
    cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
    # image = cv2.flip(image, 1)
    cv2.imwrite(test_folder + y_test[i] + '\\' + 'image_' + str(i) + '.jpg', image)
    i += 1
