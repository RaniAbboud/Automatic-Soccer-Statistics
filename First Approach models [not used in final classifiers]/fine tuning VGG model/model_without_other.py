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

# refrence: https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
from accuracy import non_neutral_accuracy, get_actual_labels

img_shape = (288, 512, 3)

if __name__ == '__main__':
    # load original data:
    data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\'

    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=img_shape)

    # Freeze the layers except the last 2 layers
    for layer in vgg_conv.layers[:-2]:
    # for layer in vgg_conv.layers:
        layer.trainable = False

    # Create the model
    model = Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_conv)

    # Add new layers
    model.add(Flatten())
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    train_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\data\\Train\\'
    val_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\data\\Validation\\'
    # test_dir = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\data\\Test\\'

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        zoom_range=[1, 1.25],
        preprocessing_function=preprocess_input)

    validation_datagen = ImageDataGenerator(horizontal_flip=True,
                                            zoom_range=[1, 1.1],
                                            preprocessing_function=preprocess_input)

    # test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_batchsize = 8
    val_batchsize = 8

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
                  optimizer=keras.optimizers.Adam(lr=0.00002),
                  metrics=['accuracy'])

    # code for saving weights
    saved_weights_path = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\fine_tuning_approach\\weights\\'
    weights_file_name = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(saved_weights_path + weights_file_name, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    #########################
    # model.load_weights(saved_weights_path + 'weights-improvement-01-0.84.hdf5')
    # # Train the model
    # history = model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=train_generator.samples / train_generator.batch_size,
    #     epochs=15,
    #     validation_data=validation_generator,
    #     validation_steps=validation_generator.samples / validation_generator.batch_size,
    #     verbose=1, callbacks=callbacks_list)
    #
    # # Save the model
    # #model.save('last.h5')
    #
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs = range(len(acc))
    #
    # plt.plot(epochs, acc, 'b', label='Training acc')
    # plt.plot(epochs, val_acc, 'r', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()
    #
    # plt.figure()
    #
    # plt.plot(epochs, loss, 'b', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    #
    # plt.show()
    # cv2.waitKey(0)
    # cv2.waitKey(0)



    # model.load_weights(saved_weights_path + 'weights-improvement-02-0.83.hdf5')
    # test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\test\\'
    # test_generator = test_datagen.flow_from_directory(
    #     test_dir,
    #     target_size=(img_shape[0], img_shape[1]),
    #     batch_size=1,
    #     classes=['Real', 'Barca', 'Other'],
    #     class_mode='categorical',
    #     shuffle=False)
    # X_test = np.array([preprocess_input(x) for x in np.load(test_folder + 'image_array.npy')])

    model.load_weights(saved_weights_path + 'weights-improvement-11-0.83_gives_69.hdf5')
    test_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\test\\'
    X_test = np.load(test_folder + 'image_array.npy')
    X_test_new = []
    for image in X_test:
        cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)
        X_test_new += [preprocess_input(image)]
    X_test = np.array(X_test_new)

    y_test = np.load(test_folder + 'labels_array.npy')
    X_test = X_test[y_test != 'Other'] ##################### dropping Other frames
    y_test = y_test[y_test != 'Other'] ##################### dropping Other frames

    pred = model.predict(X_test, batch_size=8)

    # test_generator.reset()
    # print('evaluating acc with generator: ', model.evaluate_generator(generator=test_generator))
    # pred[pred >= 0.5] = 1
    # pred[pred < 0.5] = 0
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    # predictions = np.array([labels[k] for k in pred])
    predictions = np.array([labels[k] for k in predicted_class_indices])

    print('manual acc calculation: ', sum(predictions == y_test) / y_test.shape[0])
    print('sklearn acc calculation: ', accuracy_score(y_test, predictions))

    # VISUALIZE
    X_test = np.load(test_folder + 'image_array.npy')
    y_test = np.load(test_folder + 'labels_array.npy')
    X_test = X_test[y_test != 'Other']  ##################### dropping Other frames
    j=0
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5, 300)
    fontScale = 2
    fontColor = (255, 255, 255)
    lineType = 2
    for image in X_test:
        cv2.putText(image, predictions[j],
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.imshow('window', image)
        cv2.waitKey(0)
        j += 1


    exit()





    y_test[y_test == 'Real'] = 0
    y_test[y_test == 'Barca'] = 1
    y_test[y_test == 'Other'] = 2
    y_test = keras.utils.to_categorical(y_test, 3)
    print('model.evaluate: ', model.evaluate(X_test, y_test, batch_size=8))
    print('non_neutral test accuracy=', non_neutral_accuracy(get_actual_labels(y_test), predictions))
    # print('non_neutral test accuracy=', non_neutral_accuracy(get_actual_labels(y_test), get_actual_labels(model.predict(X_test, batch_size=8))))
    exit()





    X_test = np.array([preprocess_input(x) for x in np.load(test_folder+'image_array.npy')])

    for image in X_test:
        cv2.rectangle(image, (40, 16), (70, 21), (255, 255, 255), 10)

    y_test = np.load(test_folder+'labels_array.npy')
    y_test[y_test == 'Real'] = 0
    y_test[y_test == 'Barca'] = 1
    y_test[y_test == 'Other'] = 2
    y_test = keras.utils.to_categorical(y_test, 3)

    labels = train_generator.class_indices
    print(labels)

    print('normal test accuracy=', model.evaluate(X_test, y_test, batch_size=8))
    print('non_neutral test accuracy=', non_neutral_accuracy(get_actual_labels(y_test), get_actual_labels(model.predict(X_test, batch_size=8))))

    exit()







    saved_weights_path = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\vgg_features_model\\weights\\'
    weights_file_name = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(saved_weights_path + weights_file_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # model.load_weights('weights-improvement-100-0.81.hdf5')
    # model.fit(X_train, y_train, batch_size=256, epochs=1000, callbacks=callbacks_list,
    #                 validation_data=(X_val, y_val), verbose=2)
    # y_val1 = np.load(data_folder+'y_val1.npy')
    # y_val2 = np.load(data_folder+'y_val2.npy')
    # print(np.concatenate((y_val1, y_val2), axis=0).shape)
    # exit()

    # batch_size = 512
    # num_train_samples = 9985
    # num_val_samples = 1111
    # steps_per_epoch = num_train_samples / batch_size
    # validation_steps = num_val_samples / batch_size
    # # model.load_weights('weights-improvement-40-0.85.hdf5')
    # model.fit_generator(generator=generate_batches([data_folder+'X_train1.npy', data_folder+'X_train2.npy'], [data_folder+'y_train1.npy', data_folder+'y_train2.npy'], batch_size=batch_size),
    #                     callbacks=callbacks_list,validation_data=generate_batches([data_folder+'X_val1.npy', data_folder+'X_val2.npy'], [data_folder+'y_val1.npy', data_folder+'y_val2.npy'], batch_size=batch_size),
    #                     verbose=2, epochs=1000, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
    X_train = np.concatenate([np.load(data_folder+'X_train1.npy'), np.load(data_folder+'X_train2.npy')], axis=0)
    y_train = np.concatenate([np.load(data_folder+'y_train1.npy'), np.load(data_folder+'y_train2.npy')], axis=0)

    X_val = np.concatenate([np.load(data_folder + 'X_val1.npy'), np.load(data_folder + 'X_val2.npy')], axis=0)
    y_val = np.concatenate([np.load(data_folder+'y_val1.npy'), np.load(data_folder+'y_val2.npy')], axis=0)

    # model.load_weights('weights-improvement-39-0.87.hdf5')

    model.fit(X_train, y_train, batch_size=256, epochs=1000, callbacks=callbacks_list,
              validation_data=(X_val, y_val), verbose=2)
