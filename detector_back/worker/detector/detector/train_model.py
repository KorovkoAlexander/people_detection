from __future__ import print_function

from .common.utils.config_parse import get_config
from .train.ssds_train import train_model
import click

@click.command()
@click.option("--cfg", default="ssd_resnet50_train_cocopersons_500")
def main(cfg):
    cfg = get_config(cfg)
    train_model(cfg)

if __name__ == '__main__':
    main()
