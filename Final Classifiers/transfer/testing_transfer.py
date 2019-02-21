import keras
import cv2
import numpy as np
from keras_applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score

from fine_tuning_approach.Team_Classifier import TeamClf


def get_labels(output):
    predicted_class_indices = np.argmax(output, axis=1)
    predicted_classes = ['Real' if x == 0 else 'Barca' for x in predicted_class_indices]
    return predicted_classes


test_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data 2\\1st half test\\'
images = np.load(test_dir+'test_images.npy')
labels = np.load(test_dir+'test_labels.npy')

X_test = []
for image in images:
    cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
    X_test += [preprocess_input(image)]
X_test = np.array(X_test)

X_test = X_test[labels != 'Other']
labels = labels[labels != 'Other']

real_images = X_test[labels == 'Real']
barca_images = X_test[labels == 'Barca']
print('real_images.shape=', real_images.shape)

model = TeamClf()
model = model.model
for layer in model.layers[0].layers:
    layer.trainable = False
saved_weights_path = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\fine_tuning_approach\\transfer\\weights\\'
model.load_weights(saved_weights_path+'weights-improvement-21-0.85.hdf5')

preds = get_labels(model.predict(X_test, batch_size=4))
print('sklearn acc calculation: ', accuracy_score(labels, preds))

real_preds = get_labels(model.predict(real_images, batch_size=4))
print('sklearn real acc calculation: ', accuracy_score(['Real']*len(real_preds), real_preds))

barca_preds = get_labels(model.predict(barca_images, batch_size=4))
print('sklearn barca acc calculation: ', accuracy_score(['Barca']*len(barca_preds), barca_preds))
