import requests
import urllib.parse

import numpy as np
from PIL import Image
from skimage.io._plugins.pil_plugin import pil_to_ndarray


class NonEmptyDirectoryException(ValueError):
    pass


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def load_img_from_url(photo_url):
    if not isinstance(photo_url, str):
        raise ValueError("Photo_url parameter expected to be string.")

    if photo_url.startswith("http://"):
        photo_url = "http://" + urllib.parse.quote(photo_url)[9:]
    elif photo_url.startswith("https://"):
        photo_url = "https://" + urllib.parse.quote(photo_url)[10:]
    else:
        raise ValueError("Unknown url protocol!")

    response = requests.get(photo_url, stream=True)
    response.raw.decode_content = True
    image = Image.open(response.raw)
    npimage = pil_to_ndarray(image)

    return npimage


def greyscale_to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = img
    ret[:, :, 1] = ret[:, :, 2] = ret[:, :, 0]
    return ret
