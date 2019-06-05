import gi

gi.require_version("GstBase", "1.0")

import os
import cv2
import numpy as np
import msgpack
import struct
from gi.repository import Gst, GObject, GstBase, GLib
from typing import List

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


def read_mask():
    mask = cv2.imread(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "resource",
        "mask.bmp",
    ))
    return mask


class PeopleCounter(GstBase.BaseTransform):
    __gstmetadata__ = ("PeopleCounter", "Transform", "Counter", "UM")

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
        default=1,
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer: List[int] = []

        self.last_tracks: List[int] = []
        self.last_coords: List[List[float]] = []
        self.last_mean_count: int = None
        self.room_people_counter: int = 0
        self.original_mask = read_mask()
        self._mask = None

    def get_mask(self, shape):
        shape = tuple(shape)
        if self._mask is not None and self._mask.shape == shape:
            return self._mask
        mask = cv2.resize(self.original_mask, shape)
        self._mask = np.clip(mask[:, :, 2], 0, 1)
        return self._mask

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

        mask = self.get_mask(shape)

        # check new tracks
        appeared = 0
        left = 0

        for idx, track in enumerate(_tracks):
            if track not in self.last_tracks:
                coords = _coords[idx]
                center_x = int((coords[0] + coords[2]) / 2)
                center_y = int((coords[1] + coords[3]) / 2)
                if mask[center_y, center_x]:
                    appeared += 1

        for idx, track in enumerate(self.last_tracks):
            if track not in _tracks:
                coords = self.last_coords[idx]
                center_x = int((coords[0] + coords[2]) / 2)
                center_y = int((coords[1] + coords[3]) / 2)
                if mask[center_y, center_x]:
                    left += 1

        self.last_tracks = _tracks
        self.last_coords = _coords

        room_count = 0xffffffff
        if self.last_mean_count is not None:
            delta = self.last_mean_count - mean_count
            self.room_people_counter += delta - left + appeared
            self.room_people_counter = max(0, self.room_people_counter)  # prevent subzero counts
            room_count = self.room_people_counter

        self.last_mean_count = mean_count

        meter_magic_header = 0xfa91f534
        full_data = struct.pack('!LLL', meter_magic_header, mean_count, room_count)

        assert len(full_data) <= outbuf.get_size(), (
            f"Too much data {len(full_data)} for buffer {outbuf.get_size()}"
        )

        outbuf.fill(0, full_data)

        return Gst.FlowReturn.OK


GObject.type_register(PeopleCounter)
__gstelementfactory__ = ("people_counter_py", Gst.Rank.NONE, PeopleCounter)
