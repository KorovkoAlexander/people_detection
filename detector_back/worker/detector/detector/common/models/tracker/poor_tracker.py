import numpy as np
from numba import njit


@njit
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def match_iou(boxes_list_1, boxes_list_2, iou_thresh=0.35):
    """
    Matches bboxes by IOU

    :param boxes_list_1 -> List[List[int]] represent Tracks:
    :param boxes_list_2-> List[List[int]] represent Predictions:
    :param iou_thresh -> int:
    :return: {
    0: None, -> Pred bbox with idx=0 is unmatched
    1: 2, -> Pred bbox with idx=1 matched Track #2
    3: 12
    }
    """
    IOU_matrix = [[compute_iou(bbox_1, bbox_2) for bbox_1 in boxes_list_1] for bbox_2 in boxes_list_2]
    IOU_matrix = np.array(IOU_matrix)
    xv, yv = np.meshgrid(np.arange(IOU_matrix.shape[0]), np.arange(IOU_matrix.shape[1]))
    mesh = np.stack([xv, yv], axis=2)
    mesh = mesh.reshape((-1, 2))

    indx = np.argsort(IOU_matrix.reshape((-1)), axis=0)
    matches = dict.fromkeys(range(len(boxes_list_2)), None)
    for ind in indx:
        coords = mesh[ind]
        IoU = IOU_matrix[coords[0], coords[1]]
        if IoU > iou_thresh:
            matches[coords[0]] = coords[1]
            IOU_matrix[coords[0], :] = iou_thresh -1
            IOU_matrix[:, coords[1]] = iou_thresh -1
    return matches


class Track:
    preds = []

    def __init__(self, track_id, track_size, lifetime):
        self.track_id = track_id
        self.track_size = track_size
        self.lifetime = lifetime

    def update(self, bbox):
        self.preds.append(bbox)
        self.preds = self.preds[-self.track_size:]


class Tracker:
    max_tracks = 1000

    def __init__(self, match_func, max_age):
        self.max_age = max_age
        self.match = match_func
        self.tracks = []
        self.track_ids = set()

    def update(self, bbox_list):
        assert isinstance(bbox_list, list)
        if len(self.tracks) == 0:
            matches = dict.fromkeys(range(len(bbox_list)), None)
        else:
            matches = self.match([x.preds[-1] for x in self.tracks], bbox_list)

        output_ids = [None] * len(bbox_list)
        # update tracks
        updated_tracks = []
        for bbox_id, track_id in matches.items():
            if track_id is None:  # new track
                available_track_ids = set(range(self.max_tracks)).difference(self.track_ids)
                available_track_ids = list(available_track_ids)

                new_track_id = np.random.choice(available_track_ids)
                self.track_ids.add(new_track_id)

                track = Track(new_track_id, 5, self.max_age)
                track.update(bbox_list[bbox_id])
                updated_tracks.append(track)
            else:
                track = self.tracks[track_id]
                track.update(bbox_list[bbox_id])
                updated_tracks.append(track)

            output_ids[bbox_id] = track.track_id

        # unmchached tracks
        for track_id, track in enumerate(self.tracks):
            if track_id not in matches.values():
                track.lifetime -= 1
                if track.lifetime <= 0:
                    self.track_ids.remove(track.track_id)
                else:
                    updated_tracks.append(track)

        self.tracks = updated_tracks
        return output_ids


class IOUTracker(Tracker):
    def __init__(self, max_age):
        super(IOUTracker, self).__init__(match_iou, max_age)


