import json
import os
from collections import defaultdict
from ast import literal_eval
from multiprocessing import Pool

import cv2
import pandas as pd

from .utils import NonEmptyDirectoryException, greyscale_to_rgb, isnotebook, load_img_from_url

if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def url_to_name(url, i=0):
    splitted = url[0:-4].split("/")
    date = splitted[-4]
    cinema = splitted[-3]
    hall = splitted[-2]
    rest = splitted[-1]
    return "{}__{}__{}_{}_v{}.jpg".format(cinema, hall, date, rest, i)


def cinema_and_hall_from_url(url):
    splitted_url = url[0:-4].split("/")
    cinema, hall = splitted_url[-3], splitted_url[-2]

    return cinema, hall


class LabeledPhotosDatabase(object):
    """
    LabeledPhotosDataset folder description:
    folder images -> contains downloaded images
    file images_info.csv -> csv file that contains information about downloaded images
    file logs.json -> json file that contains logs
    """

    images_folder = "images"
    table_filename = "images_info.csv"
    logs_filename = "logs.json"

    def __init__(self, database_path):
        """
        Constructor for LabeledPhotosDatabase class
        Args:
            database_path: path to instantiate database from
        """

        self.database_path_ = database_path
        self.table_ = pd.read_csv(os.path.join(database_path, self.table_filename), index_col=0)
        with open(os.path.join(database_path, self.logs_filename), "r") as file_logs:
            self.logs_ = json.loads(file_logs.read())

            # TODO: Implement assertions that database loaded from disk is not corrupted.

    @classmethod
    def from_table(cls, table, database_path):
        """
        Args:
            table: pd.DataFrame table with "location" and "points" fields that describe labeled photos.
            database_path: path to instantiate database into.

        Returns:
            LabeledPhotosDatabase instance.
        """

        required_columns = ["location", "points"]

        if not isinstance(table, pd.DataFrame):
            raise ValueError("Table parameter expected to be pandas dataframe.")
        if not all([x in table.columns for x in required_columns]):
            raise ValueError("Table dataframe must contain {} columns.".format(",".join(required_columns)))
        if not isinstance(database_path, str):
            raise ValueError("dataset_path must be string.")

        if not os.path.exists(database_path):
            os.makedirs(database_path, exist_ok=True)
        else:
            if os.path.isdir(database_path):
                if os.listdir(database_path):
                    raise NonEmptyDirectoryException(f"Directory {database_path} is not empty.")
            else:
                raise ValueError(f"Parameter dataset_path {database_path} is not a directory.")

        table = table[["location", "points"]]

        os.makedirs(cls.get_image_path(database_path), exist_ok=True)

        stats = defaultdict(int)
        filenames = dict()
        image_sizes = dict()
        cinemas = dict()
        halls = dict()
        errors = []

        for index, row in tqdm(table.iterrows()):
            repeat_location = stats[row.location]
            filename = url_to_name(row.location, repeat_location)

            try:
                img_source = load_img_from_url(row.location)
                if len(img_source.shape) == 2:
                    img_source = greyscale_to_rgb(img_source)
                cv2.imwrite(cls.get_image_path(database_path, filename), img_source)
            except:
                errors.append((row.location, filename))
            else:
                filenames[index] = filename
                stats[row.location] += 1
                image_sizes[index] = str(img_source.shape)
                cinema, hall = cinema_and_hall_from_url(row.location)
                cinemas[index] = cinema
                halls[index] = hall

                # TODO: extract session_id from lumiere

        table["filename"] = pd.Series(filenames)
        table["img_size"] = pd.Series(image_sizes)
        table["cinema"] = pd.Series(cinemas)
        table["hall"] = pd.Series(halls)

        table = table.loc[list(filenames.keys())].reset_index(drop=True)

        table.to_csv(os.path.join(database_path, cls.table_filename))

        with open(os.path.join(database_path, cls.logs_filename), "w") as file_logs:
            file_logs.write(json.dumps({"errors": errors}))

        return cls(database_path)

    @classmethod
    def process_photo(cls, photo_url, database_path, filename, points):
        filename_path = cls.get_image_path(database_path, filename)
        try:
            image = load_img_from_url(photo_url)
            if len(image.shape) == 2:
                image = greyscale_to_rgb(image)
            cv2.imwrite(filename_path, image)
        except:
            return False, None, filename
        else:
            try:
                points = literal_eval(points)
            except:
                return False, image.shape, filename
        return True, image.shape, filename

    @classmethod
    def from_lumiere_table(cls, table, database_path, num_workers=1, use_split=False):
        """
        Args:
            table: pd.DataFrame table with
                "location", "points", "cinema", "hall"
                and "session_id" fields that describe labeled photos.
            database_path: path to instantiate database into.

        Returns:
            LabeledPhotosDatabase instance.
        """

        required_columns = [
            "location", "points", "cinema", "hall", "session_id"
        ]
        if use_split:
            required_columns.extend(["split"])

        location_prefix = "http://formula-kino.adtech.rambler.ru"

        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                "Table parameter expected to be pandas dataframe."
            )
        if not all([x in table.columns for x in required_columns]):
            raise ValueError(
                "Table dataframe must contain {} columns."
                    .format(",".join(required_columns))
            )
        if not isinstance(database_path, str):
            raise ValueError("dataset_path must be string.")

        if not os.path.exists(database_path):
            os.makedirs(database_path, exist_ok=True)
        else:
            if os.path.isdir(database_path):
                if os.listdir(database_path):
                    raise NonEmptyDirectoryException(
                        f"Directory {database_path} is not empty."
                    )
            else:
                raise ValueError(
                    f"Parameter dataset_path {database_path} is not a directory."
                )

        table = table[required_columns].copy().reset_index(drop=True)
        table['duplicate_index'] = table.groupby('location').cumcount() + 1

        os.makedirs(cls.get_image_path(database_path), exist_ok=True)

        def form_task(index, row):
            filename = f"v{row.duplicate_index}_{row.location.replace('/', '_')}"
            task_info = (
                f"{location_prefix}/{row.location}",
                database_path,
                filename,
                row.points
            )
            return task_info

        task_generator = (
            form_task(index, row) for index, row in table.iterrows()
        )
        if num_workers <= 1:
            output = [
                cls.process_photo(*task_info) for task_info
                in tqdm(task_generator, total=table.shape[0])
            ]
        else:
            with Pool(num_workers) as mp_pool:
                output = mp_pool.starmap(cls.process_photo, task_generator)

        output = pd.DataFrame(output, columns=['status', 'img_size', 'filename'])
        table['img_size'] = output.img_size
        table['filename'] = output.filename

        errors = table[~output.status].to_dict()
        table = table[output.status].reset_index(drop=True)

        table.to_csv(os.path.join(database_path, cls.table_filename))

        with open(os.path.join(database_path, cls.logs_filename), "w") as file_logs:
            file_logs.write(json.dumps({"errors": errors}))

        return cls(database_path)

    @property
    def images_info(self):
        return self.table_

    @property
    def database_path(self):
        return self.database_path_

    @classmethod
    def get_image_path(cls, database_path, filename=None):
        if filename is None:
            return os.path.join(database_path, cls.images_folder)
        else:
            return os.path.join(database_path, cls.images_folder, filename)