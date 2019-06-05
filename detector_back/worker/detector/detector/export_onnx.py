import click
import torch

from .common.models.model_builder import create_model
from .common.utils.config_parse import get_config


def load_model(cfg, checkpoint_path):
    model, priorbox = create_model(cfg.MODEL)

    # Utilize GPUs for computation
    model = model.cuda()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model


@click.command()
@click.option("--cfg", default="ssd_resnet50_train_cocopersons_500")
@click.option("--checkpoint_path")
def main(cfg, checkpoint_path):
    config = get_config(cfg)
    model = load_model(config, checkpoint_path)
    dummy_input = torch.randn(1, 3, config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])
    dummy_input = dummy_input.cuda()

    torch.onnx.export(
        model,
        dummy_input,
        f"{cfg}.onnx",
        verbose=True
    )


if __name__ == '__main__':
    main()
