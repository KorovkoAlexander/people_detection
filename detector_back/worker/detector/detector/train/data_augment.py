import cv2
import torch
import numpy as np
import albumentations as albu
from albumentations.augmentations.bbox_utils import normalize_bboxes


def preproc_for_test(image, insize):
    image = cv2.resize(image, (insize[0], insize[1]))
    image = image.transpose(2, 0, 1) / 255.0
    image = image.astype(np.float32)
    return image


def preproc_for_train(image, targets, resize):
    height, width, _ = image.shape
    image = cv2.resize(image, (resize[0], resize[1]))
    image = image.transpose(2, 0, 1)/255.0
    image = image.astype(np.float32)

    boxes = targets[:, :-1]
    labels = targets[:, -1]

    boxes = np.array(normalize_bboxes(boxes, height, width))
    b_w = boxes[:, 2] - boxes[:, 0]
    b_h = boxes[:, 3] - boxes[:, 1]
    mask_b = np.minimum(b_w, b_h) > 0.001

    labels = np.expand_dims(labels[mask_b], 1)
    targets = np.hstack((boxes[mask_b], labels))
    return image, targets

class preproc():
    def __init__(self, resize, rgb_means, p, writer=None):
        self.means = rgb_means
        self.resize = resize
        self.p = p
        self.writer = writer # writer used for tensorboard visualization
        self.epoch = 0
        self.transforms = albu.Compose([
            albu.HorizontalFlip(),
            albu.ShiftScaleRotate(rotate_limit=10),
        ], bbox_params={'format': 'pascal_voc'})

    def __call__(self, image, targets=None):
        # some bugs
        if self.p == -2: # abs_test
            targets = np.zeros((1, 5))
            targets[0] = image.shape[0]
            targets[0] = image.shape[1]
            image = preproc_for_test(image, self.resize)
            return torch.from_numpy(image), targets

        boxes = targets[:,:-1].copy()
        labels = targets[:,-1].copy()
        if len(boxes) == 0:
            targets = np.zeros((1,5))
            image = preproc_for_test(image, self.resize) # some ground truth in coco do not have bounding box! weird!
            return torch.from_numpy(image), targets

        if self.p == -1: # eval
            height, width, _ = image.shape
            boxes = np.array(normalize_bboxes(boxes, height, width))
            labels = np.expand_dims(labels,1)
            targets = np.hstack((boxes,labels))
            image = preproc_for_test(image, self.resize)
            return torch.from_numpy(image), targets

        image_o = image.copy()
        targets_o = targets.copy()

        if self.transforms:
            h, w, _ = image_o.shape

            targets_o[:, 0] = np.clip(targets_o[:, 0], 0, w)
            targets_o[:, 1] = np.clip(targets_o[:, 1], 0, h)
            targets_o[:, 2] = np.clip(targets_o[:, 2], 0, w)
            targets_o[:, 3] = np.clip(targets_o[:, 3], 0, h)

            targets_o = targets_o[targets_o[:, 0] < targets_o[:, 2], :]
            targets_o = targets_o[targets_o[:, 1] < targets_o[:, 3], :]

            ann = {
                'image': image_o,
                'bboxes': targets_o,
            }
            augs = self.transforms(**ann)
            image_o = augs["image"]
            targets_o = np.array(augs["bboxes"])
            if len(targets_o) > 0:
                image, targets = image_o, targets_o

        image, targets = preproc_for_train(image, targets, self.resize)
        return torch.from_numpy(image), targets

    def add_writer(self, writer, epoch=None):
        self.writer = writer
        self.epoch = epoch if epoch is not None else self.epoch + 1

    def release_writer(self):
        self.writer = None

