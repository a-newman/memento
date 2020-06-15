import json
import os

import cv2
import fire
from tqdm import tqdm


def gen_metadata(folder, savepath):
    vids = os.listdir(folder)

    data = []

    for vid in tqdm(vids):
        vidpath = os.path.abspath(os.path.join(folder, vid))
        cap = cv2.VideoCapture(vidpath)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        metadata = {
            'width': width,
            'height': height,
            'num_frames': n_frames,
            'fps': fps,
            'path': vidpath,
            'filename': os.path.splitext(vid)[0]
        }

        data.append(metadata)

    with open(savepath, "w") as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    fire.Fire(gen_metadata)
