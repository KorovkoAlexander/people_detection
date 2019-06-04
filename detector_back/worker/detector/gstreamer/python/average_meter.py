import gi

gi.require_version("GstBase", "1.0")

import numpy as np
import msgpack
import struct
from gi.repository import Gst, GObject, GstBase, GLib

Gst.init(None)

ICAPS = Gst.Caps(
    Gst.Structure(
        "application/msgpack-predicts"
    )
)

OCAPS = Gst.Caps(
    Gst.Structure(
        "application/meter",
    )
)


class AverageMeter(GstBase.BaseTransform):
    __gstmetadata__ = ("AverageMeter", "Transform", "Meter", "UM")

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, OCAPS
        ),
        Gst.PadTemplate.new(
            "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, ICAPS
        ),
    )

    max_age = GObject.Property(
        type=int,
        nick="Max Age",
        blurb="Amount of frames, infuencing on current count.",
        minimum=0,
        maximum=100,
        default=5,
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = []

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
        return res

    def do_transform(self, inbuf, outbuf):
        magic_header = 0xfa91ffff

        is_success, info = inbuf.map(Gst.MapFlags.READ)
        assert is_success is True

        s = '!LL'
        magic, payload_length, = struct.unpack_from(s, info.data)
        struct_size = struct.calcsize(s)
        assert magic == magic_header, f"unexpected payload! {magic:x}"
        shape, _labels, _scores, _coords, _tracks = msgpack.unpackb(
            info.data[struct_size: payload_length + struct_size],
            raw=False
        )

        inbuf.unmap(info)

        self.buffer.append(len(_labels))
        self.buffer = self.buffer[-self.max_age:]

        mean_count = int(sum(self.buffer)/len(self.buffer))
        # (Kostya): apparently the size of the output buffer matches
        # the size of the input one. This should be enough for us.
        #
        # Refer to `default_prepare_output_buffer`
        # from `gstreamer/libs/gst/base/gstbasetransform.c`
        # for details.

        #payload = msgpack.packb([mean_count], use_bin_type=True)
        # Gst.debug("sent %s" % len(_labels))
        meter_magic_header = 0xfa91f534
        full_data = struct.pack('!LLL', meter_magic_header, mean_count, 0xffffffff)

        assert len(full_data) <= outbuf.get_size(), (
            f"Too much data {len(full_data)} for buffer {outbuf.get_size()}"
        )

        outbuf.fill(0, full_data)

        return Gst.FlowReturn.OK


GObject.type_register(AverageMeter)
__gstelementfactory__ = ("average_meter_py", Gst.Rank.NONE, AverageMeter)
