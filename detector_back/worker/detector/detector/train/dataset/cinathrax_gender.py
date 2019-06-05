import os
from typing import List
import torch
import torch.utils.data as data
import cv2
import numpy as np
from ast import literal_eval
from .tools.labeled_photos_database import LabeledPhotosDatabase
from .tools.cinema_info import Head, Body, CinemaInfo
#import cinethrax_data_hall_coefficients
#import cinethrax_data_seats_info

CINETHRAX_CLASSES = ('__background__', 'child', 'female', 'male')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

class LocalizationBodiesBasic(object):
    box_height_scale: float = 1.5
    pad_coeff: float = 0.25

    def localize(self, heads: List[Head]) -> List[Body]:
        return [self.expand_head_to_body(head) for head in heads]

    def expand_head_to_body(self, head: Head) -> Body:
        delta_row = head.bottom - head.top
        top_pad = int(delta_row * self.pad_coeff)
        bottom_pad = int(delta_row * self.pad_coeff) + int(delta_row * self.box_height_scale)
        h_pad = int(delta_row * self.pad_coeff)

        min_row = head.top - top_pad
        min_col = head.left - h_pad
        max_row = head.bottom + bottom_pad
        max_col = head.right + h_pad

        return Body(
            top=min_row,
            left=min_col,
            bottom=max_row,
            right=max_col
        )


class LocalizationBodiesAdaptive(object):
    body_width_expansion_mul = 0.3
    body_bottom_expansion = 0.9
    body_top_expansion = 0.3

    def __init__(self, hall):
        self.hall = hall

    def localize(self, heads: List[Head]) -> List[Body]:
        return [self.expand_head_to_body(head) for head in heads]

    def expand_head_to_body(self, head: Head) -> Body:
        width = self.hall.seat_width(head.y)

        top = int(head.y - width * self.body_top_expansion)
        left = int(head.x - width * self.body_width_expansion_mul)
        bottom = int(head.y + width * self.body_bottom_expansion)
        right = int(head.x + width * self.body_width_expansion_mul)

        return Body(
            top=top,
            left=left,
            bottom=bottom,
            right=right,
        )

class CinethraxGender(data.Dataset):
    def __init__(self, root, image_set, preproc=None, target_transform=None):
        self.root = root
        self.preproc = preproc
        self.target_transform = target_transform

        self._image_set = image_set[0]
        self._data_path = root
        self._lpd_path = os.path.join(root, self._image_set)
        self._lpd = LabeledPhotosDatabase(self._lpd_path)

        self._lpd.table_.points = self._lpd.table_.points.apply(literal_eval)
        def check(points):
            if len(points) > 0:
                return True
            return False

        self._lpd.table_ = self._lpd.table_[self._lpd.table_.points.apply(check)]

        self._image_ext = ['.JPEG']

        self._name = 'cinethrax_gender'
        self._classes = ('__background__', 'child', 'female', 'male')
        self._image_index = list(range(self._lpd.images_info.shape[0]))

        self.cinema_info = CinemaInfo(
            coefficients_path=cinethrax_data_hall_coefficients.one(),
            seats_info_path=cinethrax_data_seats_info.one()
        )

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
        return self._lpd.images_info.shape[0]

    def pull_image(self, index):
        item = self._lpd.images_info.iloc[index]
        img_path = self._lpd.get_image_path(self._lpd_path, item.filename)
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        item = self._lpd.images_info.iloc[index]
        points = item.points
        img_size = literal_eval(item.img_size)
        if len(img_size) == 3:
            image_width, image_height, _ = img_size
        else:
            image_width, image_height = img_size
        num_objs = len(points)

        hall = self.cinema_info.get_hall(item.cinema, item.hall)
        if hall.has_coefficients():
            localization_bodies = LocalizationBodiesAdaptive(hall)
        else:
            localization_bodies = LocalizationBodiesBasic()

        boxes = np.zeros((num_objs, 5))
        cls_encoding = {
            'c': 1,
            'f': 2,
            'm': 3,
        }
        # Load object bounding boxes into a data frame.
        for ix, (x, y, target) in enumerate(points):
            bbox = (
                localization_bodies.expand_head_to_body(Head(x, y, 16))
                    .prune(image_height - 1, image_width - 1)
            )
            x1, y1, x2, y2 = bbox.right, bbox.top, bbox.left, bbox.bottom
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if target in cls_encoding:
                cls = cls_encoding[target]
            else:
                cls = np.random.choice([1, 2, 3])
            boxes[ix, :] = [x1, y1, x2, y2, cls]

        return boxes

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
