import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from ast import literal_eval

FALCON_CLASSES = ('__background__', 'left', 'right')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class FalconDetection(data.Dataset):
    def __init__(self, root, image_sets, preproc=None, target_transform=None):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.table = pd.read_csv(os.path.join(self.root, "layout_v2.csv"))
        self.table = self.table[self.table.split.isin(image_sets)]

    def __getitem__(self, index):
        target = self.pull_anno(index)
        img = self.pull_image(index)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return img, target

    def __len__(self):
        return self.table.shape[0]

    def pull_image(self, index):
        filename = self.table.iloc[index]["name"]
        img_path = os.path.join(self.root, filename)
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        target = np.array(literal_eval(self.table.iloc[index].bboxes))
        target[:, -1] = target[:, -1] + 1
        return target

    def pull_img_anno(self, index):
        return self.pull_image(index), self.pull_anno(index)

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


    def show(self, index):
        img, target = self.__getitem__(index)
        for obj in target:
            obj = obj.astype(np.int)
            cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (255, 0, 0), 3)
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
