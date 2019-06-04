import gi

gi.require_version("GstBase", "1.0")

import numpy as np
import msgpack
import struct
from gi.repository import Gst, GObject, GstBase

from detector.common.models.tracker.poor_tracker import IOUTracker


Gst.init(None)

ICAPS = Gst.Caps(
    Gst.Structure(
        "application/msgpack-predicts"
    )
)

OCAPS = Gst.Caps(
    Gst.Structure(
        "application/msgpack-predicts",
    )
)


DEFAULT_MAX_AGE = 1
DEFAULT_MIN_HITS = 5


class Tracker(GstBase.BaseTransform):
    __gstmetadata__ = ("Tracker", "Transform", "Predictor", "UM")

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, OCAPS
        ),
        Gst.PadTemplate.new(
            "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, ICAPS
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_age = DEFAULT_MAX_AGE
        self._min_hits = DEFAULT_MIN_HITS
        self.trackers = []

    @GObject.Property(
        type=int,
        nick="Max Age",
        minimum=0,
        maximum=100,
        default=DEFAULT_MAX_AGE,
        flags=GObject.ParamFlags.READWRITE,
    )
    def max_age(self):
        """After that number of missing frames the track will be removed.
        """
        return self._max_age

    @max_age.setter
    def max_age(self, max_age):
        self._max_age = max_age
        for tracker in self.trackers:
            tracker.max_age = max_age

    @GObject.Property(
        type=int,
        nick="Min Hits",
        minimum=0,
        maximum=100,
        default=DEFAULT_MIN_HITS,
        flags=GObject.ParamFlags.READWRITE,
    )
    def min_hits(self):
        """Number of frames required to start a new track.
        """
        return self._min_hits

    @min_hits.setter
    def min_hits(self, min_hits):
        self._min_hits = min_hits
        for tracker in self.trackers:
            tracker.min_hits = min_hits

    def maybe_extend_trackers(self, trackers_len):
        for _ in range(trackers_len - len(self.trackers)):
            self.trackers.append(
                IOUTracker(max_age=self.max_age)
            )

    def calc_tracks(self, tracker, coords):
        track_ids = tracker.update(coords.tolist())
        return coords, track_ids

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

        if _labels:
            self.maybe_extend_trackers(max(_labels) + 1)

            _labels = np.array(_labels)
            _coords = np.array(_coords)
            _scores = np.array(_scores)

            labels_o, coords_o, tracks_o = [], [], []
            for label in np.unique(_labels):
                tracker = self.trackers[label]

                coords = _coords[_labels == label, :]
                scores = _scores[_labels == label]
                coords, track_ids = self.calc_tracks(
                    tracker,
                    coords
                )
                assert len(coords) == len(track_ids)
                labels = np.full_like(track_ids, fill_value=label)

                labels_o.append(labels)
                tracks_o.append(track_ids)
                coords_o.append(coords)

            labels_o = np.hstack(labels_o).tolist()
            tracks_o = np.hstack(tracks_o).tolist()
            coords_o = np.vstack(coords_o).tolist()
        else:
            # TODO maybe perform `update` on all trackers even when there're
            # no detections for that label currently?
            labels_o = []
            tracks_o = []
            coords_o = []

        # (Kostya): apparently the size of the output buffer matches
        # the size of the input one. This should be enough for us.
        #
        # Refer to `default_prepare_output_buffer`
        # from `gstreamer/libs/gst/base/gstbasetransform.c`
        # for details.

        payload = msgpack.packb([shape, labels_o, None, coords_o, tracks_o], use_bin_type=True)
        # Gst.debug("sent %s" % len(_labels))

        full_data = struct.pack('!LL', magic_header, len(payload)) + payload

        assert len(full_data) <= outbuf.get_size(), (
            f"Too much data {len(full_data)} for buffer {outbuf.get_size()}"
        )

        outbuf.fill(0, full_data)

        return Gst.FlowReturn.OK


GObject.type_register(Tracker)
__gstelementfactory__ = ("tracker_py", Gst.Rank.NONE, Tracker)
