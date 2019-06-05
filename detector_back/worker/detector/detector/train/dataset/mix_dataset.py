import os
import torch
import torch.utils.data as data
import pandas as pd
import cv2
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
from functools import partial

MIX_CLASSES = ('__background__', 'person')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128))

class DatasetInfo:
    def __init__(self, folder, anno, images, take_classes, top_n=-1):
        self.folder = folder
        self.anno = anno
        self.images = images
        self.top_n = top_n
        self.classes = take_classes

    @staticmethod
    def filter_bboxes(bboxes, classes):
        boxes = list(filter(lambda x: x[4] in classes, bboxes))
        if len(boxes) == 0:
            return None
        return boxes

    def get_table(self, root):
        table = pd.read_csv(
                os.path.join(
                    root,
                    self.folder,
                    self.anno
                )
            )
        table.boxes = table.boxes.apply(literal_eval)
        table.boxes = table.boxes.apply(partial(self.filter_bboxes, classes=self.classes))
        table = table.dropna()

        if self.top_n:
            if len(table) > self.top_n:
                return table.iloc[
                    np.random.randint(
                        low=1,
                        high=len(table),
                        size=self.top_n)
                ]
        return table

    def get_images_folder(self, root):
        return os.path.join(
            root,
            self.folder,
            self.images
        )


datasets = [
    DatasetInfo(
        folder="COCOPerson",
        anno="annotations/layout_person.csv",
        images="train2017",
        take_classes=[1],
        top_n=15000
    ),
    DatasetInfo(
        folder="CrowdHuman",
        anno="Anno/layout.csv",
        images="CrowdHuman",
        take_classes=[1],
        top_n=0
    ),
    DatasetInfo(
        folder="Rambler_office",
        anno="Anno/rambler_office_train.csv",
        images="Images",
        take_classes=[1],
        top_n=0
    )
]


class MixDetection(data.Dataset):
    def __init__(self, root, image_sets, preproc=None):
        self.root = root
        self.image_set = image_sets[0]
        self.preproc = preproc

        tables = [x.get_table(root) for x in datasets]
        for i, table in enumerate(tables):
            tables[i]["dataset"] = tables[i].apply(lambda x: i, axis=1)

        table = pd.concat(tables, sort=False)
        self.img_folders = [x.get_images_folder(root) for x in datasets]

        train, val = train_test_split(table, test_size=0.10, random_state=42)
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
        dataset = self.table.iloc[index]["dataset"]
        img_folder = self.img_folders[dataset]
        filename = self.table.iloc[index]["image_id"]
        img_path = os.path.join(img_folder, filename)
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_bbox(self, index):
        target = np.array(self.table.iloc[index].boxes)
        return target

    def pull_img_anno(self, index):
        return self.pull_image(index), self.pull_bbox(index)

    def pull_tensor(self, index):
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def show(self, index):
        pass

    def evaluate_detections(self, all_boxes, output_dir=None):
        pass
