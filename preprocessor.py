import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def FrameExtractor():
    """Packages dataset from file directory and sends them to a model.

    Returns:
        array: Array of 3 tuples. [(x_train, y_train), (x_validation, y_validation), (x_test, y_test)].
            x_XXX = [samples, frames, height, width]
            y_YYY = [samples, one_hot_encoded_labels]
    """

    test_data_dir = '..\\Datasets\\florence3d_actions\\Florence_3d_actions\\test'
    train_data_dir = '..\\Datasets\\florence3d_actions\\Florence_3d_actions\\train'
    validation_data_dir = '..\\Datasets\\florence3d_actions\\Florence_3d_actions\\validation'
    x, y = collector(train_data_dir)
    xv, yv = collector(validation_data_dir)
    xt, yt = collector(test_data_dir)

    # At this point the content of the x,xv,xt,y,yv,yt is sorted according to the file structure, which
    # makes similar category to be in a sequeence. But sending them as it is will harm the training process
    # as the models will overfit for the sample pathces. To avoid this the data needs to be shuffled before
    # sending it to the model.

    return shuffler(x, y), shuffler(xv, yv), shuffler(xt, yt)


def collector(root_dir):
    labels = os.listdir(root_dir)  # gives the categories as 1,2,3,..,9
    label_to_y_map = []
    x = []
    y = []
    for idx, label in enumerate(labels):  # enumerate(labels):
        label_to_y_map.append((label, idx))
        sample_dir = os.path.join(root_dir, label)
        samples = os.listdir(sample_dir)
        for sample in samples:
            filename = os.path.join(sample_dir, sample)
            x_frame, y_label = cv2Frame(filename, idx)
            x.append(x_frame)
            y.append(y_label)
    file = open('classmap.txt', 'a')
    file.write(root_dir)
    file.write("\n")
    file.write(str(label_to_y_map))
    file.write("\n")
    file.close()
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    return (x, one_hot_encode(y, len(labels)))


def cv2Frame(filename='E:\\Work\\Thesis\\Datasets\\florence3d_actions\\Florence_3d_actions\\test\\1\\GestureRecording_Id1actor2idAction1category1.avi', idx=0):
    """
        'E:\\Work\\Thesis\\Datasets\\florence3d_actions\\Florence_3d_actions\\test\\1\\GestureRecording_Id1actor2idAction1category1.avi'
    """
    #---------------------cv2.VideoCapture Info---------------#
    # CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
    # CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    # CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    # CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    #
    # Note: The Documentation of cv2 shows constans with CV_ prefix but in most cases CV_ prefixes are dropped.
    #       Which means the doc is outdated.
    #---------------------------------------------------------#

    vid = cv2.VideoCapture(filename)
    n_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    n_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_channel = 1
    np_array = np.zeros([n_frame, n_height, n_width, n_channel])

    success = True
    while success:
        success, frame = vid.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=-1)
        np_array[int(vid.get(cv2.CAP_PROP_POS_FRAMES)-1)] = gray

    y = np.zeros((n_frame), dtype=int) + idx
    return (np_array, y)


def one_hot_encode(y, labels):
    x = np.zeros([y.shape[0], labels])
    x[np.arange(x.shape[0]), y] = 1

    return x


def shuffler(x, y):
    shuffled_index = np.arange(y.shape[0])
    np.random.shuffle(shuffled_index)
    x_shuffled = x[shuffled_index]
    y_shuffled = y[shuffled_index]

    return (x_shuffled, y_shuffled)
