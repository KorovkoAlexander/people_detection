import os
import torch
import torch.utils.data as data
import pandas as pd
import cv2
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split

CELEB_CLASSES = ('__background__', 'person')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128))


class CelebDetection(data.Dataset):
    def __init__(self, root, image_sets, preproc=None):
        self.root = root
        self.image_set = image_sets[0]
        self.preproc = preproc
        self.table = pd.read_csv(
            os.path.join(
                self.root,
                "Anno",
                "celeb_info.csv"
            )
        )

        train, val = train_test_split(self.table, test_size=0.20, random_state=42)
        if self.image_set == "train":
            self.table = train
        elif self.image_set == "test":
            self.table = val

    def __getitem__(self, index):
        bbox = self.pull_bbox(index)
        img = self.pull_image(index)

        if self.preproc is not None:
            img, bbox = self.preproc(img, bbox)

        return img, bbox

    def __len__(self):
        return self.table.shape[0]

    def pull_image(self, index):
        filename = self.table.iloc[index]["image_id"]
        img_path = os.path.join(
            self.root,
            'Img',
            "img_celeba",
            filename
        )
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_bbox(self, index):
        target = np.array(literal_eval(self.table.iloc[index].bbox))
        # add sintetic labels
        target = np.hstack([target, np.ones((target.shape[0], 1))])
        return target

    def pull_img_anno(self, index):
        return self.pull_image(index), self.pull_bbox(index)

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


    def show(self, index):
        img, target = self.__getitem__(index)
        for obj in target:
            obj = obj.astype(np.int)
            cv2.rectangle(
                img,
                (obj[0], obj[1]),
                (obj[2], obj[3]),
                (255, 0, 0),
                3
            )
        cv2.imwrite('./image.jpg', img)

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        pass
