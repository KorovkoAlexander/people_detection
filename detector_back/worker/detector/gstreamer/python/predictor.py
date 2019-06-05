import gi

gi.require_version("GstBase", "1.0")

import os

import numpy as np
import msgpack
import struct
from gi.repository import Gst, GObject, GstBase, GLib

from detector.inference.tensort import TRTDetector, load_cfg
from detector.inference.tensort.common import init_pycuda

Gst.init(None)

# TODO move these to the element's properties
WIDTH = 800
HEIGHT = 800


ICAPS = Gst.Caps(
    Gst.Structure(
        "video/x-raw",
        format="BGR",
        width=WIDTH,
        height=HEIGHT,
    )
)

OCAPS = Gst.Caps(
    Gst.Structure(
        "application/msgpack-predicts",
    )
)


DEFAULT_MODEL_CFG = "ssd_resnet18_train_mix_800"
DEFAULT_MODEL_CHECKPOINT = "ssd_resnet18_train_min_int.trt"
DEFAULT_GPU_DEVICE_ID = 0


class Predictor(GstBase.BaseTransform):
    __gstmetadata__ = ("Predictor", "Transform", "Predictor", "UM")

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, OCAPS
        ),
        Gst.PadTemplate.new(
            "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, ICAPS
        ),
    )

    predict_threshold = GObject.Property(
        type=float,
        nick="Predict Threshold",
        blurb="Minimum score a predict should have to be kept.",
        minimum=0.0,
        maximum=1.0,
        default=0.3795,
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_cfg = DEFAULT_MODEL_CFG
        self._model_checkpoint = DEFAULT_MODEL_CHECKPOINT
        self._gpu_device_id = DEFAULT_GPU_DEVICE_ID

        init_pycuda(self._gpu_device_id)

        self._detector = None

    @GObject.Property(
        type=str,
        nick="Model cfg",
        default=DEFAULT_MODEL_CFG,
        flags=GObject.ParamFlags.READWRITE,
    )
    def model_cfg(self):
        """Model config file."""
        return self._model_cfg

    @model_cfg.setter
    def model_cfg(self, model_cfg):
        if model_cfg != self._model_cfg:
            self._model_cfg = model_cfg
            self._detector = None

    @GObject.Property(
        type=str,
        nick="Model checkpoint",
        default=DEFAULT_MODEL_CHECKPOINT,
        flags=GObject.ParamFlags.READWRITE,
    )
    def model_checkpoint(self):
        """Model checkpoint file"""
        return self._model_checkpoint

    @model_checkpoint.setter
    def model_checkpoint(self, model_checkpoint):
        if model_checkpoint != self._model_checkpoint:
            self._model_checkpoint = model_checkpoint
            self._detector = None

    @GObject.Property(
        type=int,
        nick="GPU device id",
        minimum=0,
        maximum=100,  # TODO maybe detect the number of gpu devices instead?
        default=DEFAULT_GPU_DEVICE_ID,
        flags=GObject.ParamFlags.READWRITE,
    )
    def gpu_device_id(self):
        """GPU device id."""
        return self._gpu_device_id

    @gpu_device_id.setter
    def gpu_device_id(self, gpu_device_id):
        if gpu_device_id != self._gpu_device_id:
            self._gpu_device_id = gpu_device_id
            self._detector = None

    @property
    def detector(self):
        if self._detector is None:
            # Note that detector creation is slow: it takes abt. 2 seconds.
            cfg = load_cfg(self.model_cfg)
            self._detector = TRTDetector(
                cfg,
                engine_file_path=os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    "resource",
                    self.model_checkpoint,
                ),
                use_preproc=False
            )

        return self._detector

    def do_transform_caps(self, direction, caps, filter_):
        # (Kostya): without this override the element negotiates
        # the same formats on both caps, which is not what we need here:
        # our `src` cap must return predicts instead of a video stream.
        #
        # Copied from: https://github.com/GStreamer/gst-python/blob/30cc0fc83de306626bfb30b679156be30fbf1be9/examples/plugins/python/audioplot.py#L164

        if direction == Gst.PadDirection.SRC:
            res = ICAPS
        else:
            res = OCAPS

        if filter_:
            res = res.intersect(filter_)

        # Warmup the detector before the pipeline switches state to PLAYING.
        assert self.detector is not None

        return res

    def do_transform(self, inbuf, outbuf):
        is_success, info = inbuf.map(Gst.MapFlags.READ)
        assert is_success is True
        raw_data = np.frombuffer(info.data, dtype=np.uint8)
        inbuf.unmap(info)

        shape = (int(WIDTH), int(HEIGHT))
        image = raw_data.reshape((WIDTH, HEIGHT, 3))
        image_normalized = image.astype(np.float32) / 255.0
        image_normalized = np.transpose(image_normalized, [2, 0, 1])[np.newaxis, :, :, :]
        image_normalized = np.array(image_normalized, order='C')

        _labels, _scores, _coords = self.detector.predict(
            image_normalized, threshold=self.predict_threshold
        )
        _tracks = None

        # (Kostya): apparently the size of the output buffer matches
        # the size of the input one. This should be enough for us.
        #
        # Refer to `default_prepare_output_buffer`
        # from `gstreamer/libs/gst/base/gstbasetransform.c`
        # for details.

        payload = msgpack.packb([shape, _labels, _scores, _coords, _tracks], use_bin_type=True)
        # Gst.debug("sent %s" % len(_labels))

        magic_header = 0xfa91ffff
        full_data = struct.pack('!LL', magic_header, len(payload)) + payload

        assert len(full_data) <= outbuf.get_size(), (
            f"Too much data {len(full_data)} for buffer {outbuf.get_size()}"
        )

        outbuf.fill(0, full_data)

        # Gst.debug("written %s" % full_data[:30].hex())

        return Gst.FlowReturn.OK


GObject.type_register(Predictor)
__gstelementfactory__ = ("predictor_py", Gst.Rank.NONE, Predictor)
