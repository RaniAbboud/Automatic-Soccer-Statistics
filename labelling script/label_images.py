import cv2
import os
import tkinter
import numpy as np

OUTPUT_DIR = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\images\\'
team = None
size_reduction = 0.4
save_after_n_labelings = 100  # because saving frequently slows the procedure
mouseX, mouseY = 0, 0
cur_img = None
FPS = 30

# data_folder = 'C:\\Users\\Rani\\Desktop\\AI Pycharm Project\\labelled data\\'
#
# images = np.load(data_folder+'image_array1.npy')
# labels = np.load(data_folder+'labels_array1.npy')
# ball_coords = np.load(data_folder+'ball_array1.npy')
#
# for i in range(len(images)):
#     cv2.circle(images[i], tuple(ball_coords[i]), 10, (255, 0, 0), 5)
#     cv2.imshow('im', images[i])
#     cv2.waitKey(0)
# exit()
# ###############################################
def get_training_video_number():
    print('Please enter the desired training video number (1/2), then press enter:')
    n = int(input())
    while n not in [1, 2]:
        print('Try again')
        n = int(input())
    return n


def get_nth_frame(video, frame_num):  # gets a cv2.VideoCapture and a frame-number
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_num >= totalFrames:
        return None  # no more frames
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = video.read()
    if ret:
        return frame
    return None


def get_frame(video):
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # for i in range(5,totalFrames,30):  # start at frame#5, jump by 30 frames (by 1-second since video is 30FPS)
    for i in range(5, totalFrames, FPS):  # start at frame#5, jump by 25 frames (by 1-second since video is 25FPS)
        frame = get_nth_frame(video, i)
        if frame is None:
            print('Error at get_frame')
            return None
        yield frame


def label_images_wrapper():
    n = get_training_video_number()
    if n == 1:
        cap = cv2.VideoCapture(r'C:\Users\Rani\Desktop\AI project\data (video)\Train1.mp4')
        # cap = cv2.VideoCapture(r'C:\Users\Rani\Desktop\AI project\data 3 (video)\Test1.mp4')
        # cap = cv2.VideoCapture('C:\\Users\\Rani\\Desktop\\AI project\\\data (video)\\Test2.mp4')
        # cap = cv2.VideoCapture('C:\\Users\\Rani\\Desktop\\AI project\\clasico2\\test720.mp4')
    else:  # n == 2
        cap = cv2.VideoCapture(r'C:\Users\Rani\Desktop\AI project\data (video)\Train2.mp4')
    label_images(cap)


def get_click(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        circled_img = cur_img.copy()
        cv2.circle(circled_img,(x,y),10,(255,0,0),5)
        cv2.imshow('image', circled_img)
        mouseX, mouseY = x, y
        # get_possession()


def label_images(video: cv2.VideoCapture):
    global cur_img
    count = 0
    images = []
    labels = []
    ball_coords = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_click)
    for img in get_frame(video):
#        img = cv2.imread(OUTPUT_DIR+im)
        cur_img = img
        img, team, ball_location = label_image(img)
        images += [img]
        labels += [team]
        ball_coords += [ball_location]
        count += 1
        if (count % save_after_n_labelings) == 0:  # save after every X labellings
            np.save('image_array', images)
            np.save('labels_array', labels)
            np.save('ball_array', ball_coords)
            print('saved '+str(count)+' examples')

    # Done, save labellings
    np.save('image_array', images)
    np.save('labels_array', labels)
    np.save('ball_array', ball_coords)
    # print count
    print('images len=', len(images))
    print('labels len=', len(labels))
    print('ball_coords len=', len(ball_coords))


def label_image(image):
    cv2.imshow('image', image)
    # waiting for possession label

    k = cv2.waitKey(0) & 0xFF
    while k not in [ord(c) for c in ['b', 'm', 'n']] + [27]:
        k = cv2.waitKey(0) & 0xFF

    ball_location = (mouseX, mouseY)

    if k == 27:  # got esc button
        exit()
    elif k == ord('b'):
        team = 'Barca'
    elif k == ord('m'):
        team = 'Real'
    elif k == ord('n'):
        team = 'Other'
    print('labeled ' + '*' + team + '* ' + str(ball_location))
    image = cv2.resize(image, (0, 0), fx=size_reduction, fy=size_reduction)
    label = team
    ball_location = tuple([int(round(x * size_reduction)) for x in ball_location])
    return image, label, ball_location

if __name__ == '__main__':
    label_images_wrapper()
