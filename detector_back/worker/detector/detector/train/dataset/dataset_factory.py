import torch

from detector.train.data_augment import preproc
import torch.utils.data as data

from . import voc
from . import falcon_dataset
from . import cinathrax_gender
from . import celeb_dataset
from . import ntechlab_dataset
from . import coco_persons_dataset
from . import crowdhuman_dataset
from . import wider_face
from . import mix_dataset

dataset_map = {
                'voc': voc.VOCDetection,
                'falcon': falcon_dataset.FalconDetection,
                'celeb': celeb_dataset.CelebDetection,
                'ntechlab': ntechlab_dataset.NTechLabDetection,
                'cocopersons': coco_persons_dataset.COCOPersonsDetection,
                'cinethrax_gender': cinathrax_gender.CinethraxGender,
                'crowdhuman': crowdhuman_dataset.CrowdHumanDetection,
                'wider': wider_face.WiderFaceDetection,
                'mix': mix_dataset.MixDetection
            }

def gen_dataset_fn(name):
    """Returns a dataset func.

    Args:
    name: The name of the dataset.

    Returns:
    func: dataset_fn

    Raises:
    ValueError: If network `name` is not recognized.
    """
    if name not in dataset_map:
        raise ValueError('The dataset unknown %s' % name)
    func = dataset_map[name]
    return func


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations and list of landmarks

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])

        annos = torch.from_numpy(sample[1]).float()
        targets.append(annos)

    return (torch.stack(imgs, 0), targets)


def load_data(cfg, phase):
    if phase == 'train':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TRAIN_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, cfg.PROB))
        data_loader = data.DataLoader(dataset, cfg.TRAIN_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
        return data_loader
    if phase == 'eval':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, -1))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
        return data_loader
    if phase == 'test':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, -1))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
        return data_loader
    if phase == 'visualize':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, 1))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
        return data_loader
    raise AttributeError("Phase should be one of [train, test, eval, visualize]")
