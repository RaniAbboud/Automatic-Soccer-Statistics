import keras
from keras.callbacks import ModelCheckpoint
from keras_applications.vgg16 import preprocess_input
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from fine_tuning_approach.Team_Classifier import TeamClf
import cv2
import numpy as np

clf = TeamClf()
clf.load()

model = clf.model
for layer in model.layers[0].layers:
    layer.trainable = False

train_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data 2\\data\\Train\\'
val_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data 2\\data\\Validation\\'

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    zoom_range=[1, 1.25],
    preprocessing_function=preprocess_input)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

# Compile the model
model.compile(loss='categorical_crossentropy',
              # optimizer=optimizers.RMSprop(lr=1e-4),
              optimizer=keras.optimizers.Adam(lr=0.00001),
              metrics=['accuracy'])

# code for saving weights
saved_weights_path = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\fine_tuning_approach\\transfer\\weights\\'
weights_file_name = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(saved_weights_path + weights_file_name, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.summary()
# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=2, callbacks=callbacks_list)


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
