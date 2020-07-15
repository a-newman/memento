import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

import config as cfg


def show_captions_file(fpath, start=0, end=10):
    with open(fpath) as infile:
        caps = json.load(infile)

    data = [(k, v) for k, v in caps.items()]

    for i in range(start, end):
        fname, captions = data[i]
        vidpath = os.path.join(cfg.MEMENTO_ROOT, "videos", fname + ".mp4")
        frames = load_video_opencv(vidpath)
        plot_frames(frames, is_255image=True, frames_to_show=5, title=fname)
        print(captions)
        print()


def load_video_opencv(video_file, verbose=False, is_train=False):

    cap = cv2.VideoCapture(video_file)

    if verbose:
        print('video_file:', video_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(
            video_file.split('/')[-1],
            'loaded. \nwidth, height, n_frames, fps:', width, height, n_frames,
            fps)

    test_frames = []
    success = True
    # i=0

    while success:
        success, frame = cap.read()
        # i+=1

        if success:
            test_frames.append(frame)

    test_frames = np.array(test_frames)
    test_frames = convert_rgb(test_frames)

    if verbose:
        print('Frames appended in numpy array. Shape of test_frames:',
              test_frames.shape)

    return test_frames.astype(np.uint8)


def convert_rgb(frames):
    cvt_frames = []

    for frame in frames:
        cvt_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return np.array(cvt_frames)


def plot_frames(frames,
                title='',
                is_optical_flow=False,
                is_255image=False,
                is_01image=False,
                idxs=None,
                frames_to_show=3,
                suptitle_y=0.75):
    """ Plots frames indicated in idxs after rescaling them to 0-255.

    Args
    ----
    frames: numpy array
        Array of frames to show. Dims: (F,H,W,C)
    title: str
        title for the set of subplots
    is_optical_flow: Boolean
        if True, assumes that the input frames are 2-channel OF frames
        with values between -0.5 and 0.5
    is_255image: boolean
        if True, assumes that input is array of uint8 (values from 0 to 255)
    is_01image: Boolean
        if True, assumes that input is array of floats (decimal values between 0 and 1)
    idxs: list of ints
        list of frame indexes to plot. Overrides frames_to_show.
    frames_to_show:
        number of equidistant frames to show from video. Will always show first and last frame.
    """

    # print(np.max(frames))

    fig = plt.figure(figsize=[16, 6])
    c = 1

    if idxs is None:
        idxs = [
            i * len(frames) // (frames_to_show - 1)
            for i in range(frames_to_show - 1)
        ]
        idxs.append(-1)

    for idx in idxs:
        if not is_optical_flow:
            if not is_01image:
                if not is_255image:
                    fr1 = (frames[idx] + 1) / 2
                else:
                    fr1 = frames[idx] / 255
            else:
                fr1 = frames[idx]
        else:
            fr1 = np.zeros((frames[idx].shape[0], frames[idx].shape[1], 3))
            fr1[:, :, :2] = frames[idx]

            if not is_255image:
                fr1 += 0.5
            else:
                fr1 /= 255

        plt.subplot(1, len(idxs), c)
        plt.axis('off')
        plt.yticks([])
        plt.xticks([])
        plt.imshow(fr1)
        c += 1

    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    # plt.suptitle(title + ' min %.3f, max %.3f' % (np.min(fr1), np.max(fr1)),
    #              y=suptitle_y)
    plt.show()
