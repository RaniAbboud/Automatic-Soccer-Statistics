import numpy as np
import keras
from sklearn.model_selection import train_test_split


def prepare_labels(raw_labels):
    labels = raw_labels
    labels[labels == 'Real'] = 0
    labels[labels == 'Barca'] = 1
    labels[labels == 'Other'] = 2
    return keras.utils.to_categorical(labels, 3)


if __name__ == '__main__':
    output_folder = 'C:\\Users\\Rani\Desktop\\AI Pycharm Project\\vgg_features_model\\prepared_data\\'
    # load original labels:
    data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\vgg_features_model\\'
    labels1 = prepare_labels(np.load(data_folder + 'labels1.npy'))
    labels2 = prepare_labels(np.load(data_folder + 'labels2.npy'))

    # load vgg features:
    features1 = np.squeeze(np.load(data_folder + 'vgg_features1.npy'))
    features2 = np.squeeze(np.load(data_folder + 'vgg_features2.npy'))

    X_train1, X_val1, y_train1, y_val1 = train_test_split(features1, labels1, test_size=0.1, random_state=42)
    np.save(output_folder+'X_train1.npy', X_train1)
    np.save(output_folder+'y_train1.npy', y_train1)
    np.save(output_folder+'X_val1.npy', X_val1)
    np.save(output_folder+'y_val1.npy', y_val1)

    X_train2, X_val2, y_train2, y_val2 = train_test_split(features2, labels2, test_size=0.1, random_state=42)
    np.save(output_folder+'X_train2.npy', X_train2)
    np.save(output_folder+'y_train2.npy', y_train2)
    np.save(output_folder+'X_val2.npy', X_val2)
    np.save(output_folder+'y_val2.npy', y_val2)



