import os
import time
from tqdm import trange

import cv2
import click

import torch.cuda

from detector import ObjectDetector
from detector.lib.utils.config_parse import get_config

NUM_WARM = 32
NUM_RUNS = 512

models = [
    {
        'model_cfg': 'm2det_resnet50_train_crowdhuman_300',
        'checkpoint': 'm2det_resnet50_crowdhuman_300.pth'
    },
    {
        'model_cfg': 'm2det_resnet50_train_crowdhuman_500',
        'checkpoint': 'm2det_resnet50_crowdhuman_500.pth'
    },
    {
        'model_cfg': 'ssd_resnet50_train_cocopersons_300',
        'checkpoint': 'ssd_resnet50_cocopersons_300.pth'
    },
    {
        'model_cfg': 'ssd_resnet50_train_cocopersons_500',
        'checkpoint': 'ssd_resnet50_train_cocopersons_500.pth'
    },
]


@click.command()
@click.option('--model_number', default=2)
def main(model_number):
    model = models[model_number]

    cfg = get_config(model['model_cfg'])

    detector = ObjectDetector(
        cfg,
        os.path.join(
            'resource',
            model['checkpoint']
        ),
        device_id=0
    )

    image = cv2.imread('image.jpg')

    for _ in trange(NUM_WARM, desc='Warm-up'):
        detector.predict(image, threshold=0.3)

    start = time.time()
    for _ in trange(NUM_RUNS, desc='Benches'):
        detector.predict(image, threshold=0.3)
    torch.cuda.synchronize()
    total = time.time() - start

    print(f'Performance: {NUM_RUNS / total:0.2f} rps')


if __name__ == '__main__':
    main()
