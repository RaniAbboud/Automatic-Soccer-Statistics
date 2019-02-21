import numpy as np
import cv2


def get_team(i : int):
    if i == 0:
        return 'Real'
    if i == 1:
        return 'Barca'
    if i == 2:
        return 'Other'


def get_actual_labels(labels):
    actual_labels = np.argmax(labels, axis=1)
    actual_labels = [get_team(x) for x in actual_labels]
    return np.array(actual_labels)


def non_neutral_accuracy(actual, predicted):  # gets 2 arrays
    actual = np.array(actual)
    predicted = np.array(predicted)

    if actual.shape != predicted.shape:
        print('non_neutral accuracy function - ERROR: ', actual.shape, predicted.shape)

    non_neutral = (actual != 'Other')
    neutral = (actual == 'Other')

    actual_copy = actual.copy()
    predicted_copy = predicted.copy()
    predicted_copy[neutral] = 'X'
    actual_copy[neutral] = 'Y'

    same = (actual_copy == predicted_copy)
    return sum(same)/sum(non_neutral)
