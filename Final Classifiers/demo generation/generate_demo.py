import numpy as np
import cv2
from fine_tuning_approach.Team_Classifier import TeamClf
from fine_tuning_approach.Frame_Type_Classifier import FrameTypeClf


def generate_demo(teamLabels, frameTypeLabels, videoPath, create=False):
    bottomLeftCornerOfText = (8, 695)
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 3
    fontColor = (255, 255, 255)
    lineType = 4
    # predicted = np.load('Test'+str(num)+'_predicted_labels.npy')
    # cap = cv2.VideoCapture('C:\\Users\\Rani\\Desktop\\AI project\\data (video)\\Test'+str(num)+'.mp4')
    cap = cv2.VideoCapture(videoPath)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if create:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('new_demo.avi', fourcc, 30.0, (1280, 720))

    j = 0
    while j < totalFrames:
        ret, frame = cap.read()
        if frameTypeLabels[j] == 'On':
            cv2.putText(frame, '['+teamLabels[j]+']',
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
        if create:
            out.write(frame)
        else:
            cv2.imshow('window', frame)
            cv2.waitKey(20)
        j += 1
    cap.release()
    if create:
        out.release()


if __name__ == '__main__':
    # load models
    teamClf = TeamClf()
    teamClf.load()
    frameClf = FrameTypeClf()
    frameClf.load()

    # predict labels
    path = 'C:\\Users\\Rani\\Desktop\\AI project\\clasico2\\test720.mp4'
    teamLabels = teamClf.predict_labels(videoPath=path)
    frameTypeLabels = frameClf.predict_labels(videoPath=path)

    # # load labels
    # teamLabels = np.load()
    # frameTypeLabels = np.load()

    generate_demo(teamLabels, frameTypeLabels, videoPath=path, create=True)


