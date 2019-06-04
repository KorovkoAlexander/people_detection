import numpy as np
from math import sqrt
from itertools import product


def decode(loc, priors, variances):
    boxes = np.hstack([
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = []
    if len(scores) == 0:
        return keep, 0
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = np.multiply(x2 - x1, y2 - y1)
    idx = np.argsort(scores)
    idx = idx[-top_k:]

    count = 0
    while len(idx) > 0:
        i = idx[-1]
        keep.append(i)
        count += 1
        if len(idx) == 1:
            break
        idx = idx[:-1]
        xx1 = np.take(x1, idx, axis=0)
        yy1 = np.take(y1, idx, axis=0)
        xx2 = np.take(x2, idx, axis=0)
        yy2 = np.take(y2, idx, axis=0)
        xx1 = np.clip(xx1, a_min=x1[i], a_max=None)
        yy1 = np.clip(yy1, a_min=y1[i], a_max=None)
        
        xx2 = np.clip(xx2, a_max=x2[i], a_min=None)
        yy2 = np.clip(yy2, a_max=y2[i], a_min=None)
        w = xx2 - xx1
        h = yy2 - yy1
        w = np.clip(w, a_min=0.0, a_max=None)
        h = np.clip(h, a_min=0.0, a_max=None)
        inter = w*h
        rem_areas = np.take(area, idx, axis=0)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union
        idx = idx[IoU <= overlap]
    return keep, count


def restore_bboxes(detections, threshold, img_shape):
    scale = np.array([*img_shape, *img_shape])
    labels, scores, coords = [list() for _ in range(3)]
    batch = 0
    for classes in range(detections.shape[1]):
        num = 0
        while detections[batch, classes, num, 0] >= threshold:
            score = detections[batch, classes, num, 0]
            coord = np.multiply(detections[batch, classes, num, 1:5], scale)
            scores.append(score.tolist())
            labels.append(classes - 1)
            coords.append(coord.tolist())
            num += 1
            if num >= detections.shape[2]:
                break
    return labels, scores, coords


class PriorBox:
    def __init__(
            self,
            image_size,
            feature_maps,
            aspect_ratios,
            scale,
            archor_stride=None,
            archor_offest=None,
            clip=True
    ):
        super(PriorBox, self).__init__()
        self.image_size = image_size
        self.feature_maps = feature_maps
        self.aspect_ratios = aspect_ratios
        self.num_priors = len(aspect_ratios)
        self.clip = clip
        if isinstance(scale[0], list):
            self.scales = [min(s[0] / self.image_size[0], s[1] / self.image_size[1]) for s in scale]
        elif isinstance(scale[0], float) and len(scale) == 2:
            num_layers = len(feature_maps)
            min_scale, max_scale = scale
            self.scales = [
                              min_scale + (max_scale - min_scale) * i / (num_layers - 1)
                              for i in range(num_layers)
                          ] + [1.0]
        
        if archor_stride:
            self.steps = [
                (steps[0] / self.image_size[0], steps[1] / self.image_size[1])
                for steps in archor_stride
            ]
        else:
            self.steps = [(1/f_h, 1/f_w) for f_h, f_w in feature_maps]

        if archor_offest:
            self.offset = [
                [offset[0] / self.image_size[0], offset[1] * self.image_size[1]]
                for offset in archor_offest
            ]
        else:
            self.offset = [[steps[0] * 0.5, steps[1] * 0.5] for steps in self.steps] 

    def __call__(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                cx = j * self.steps[k][1] + self.offset[k][1]
                cy = i * self.steps[k][0] + self.offset[k][0]
                s_k = self.scales[k]

                for ar in self.aspect_ratios[k]:
                    if isinstance(ar, int):
                        if ar == 1:
                            mean += [cx, cy, s_k, s_k]
                            s_k_prime = sqrt(s_k * self.scales[k+1])
                            mean += [cx, cy, s_k_prime, s_k_prime]
                        else:
                            ar_sqrt = sqrt(ar)
                            mean += [cx, cy, s_k*ar_sqrt, s_k/ar_sqrt]
                            mean += [cx, cy, s_k/ar_sqrt, s_k*ar_sqrt]
                    elif isinstance(ar, list):
                        mean += [cx, cy, s_k*ar[0], s_k*ar[1]]
        output = np.array(mean).reshape(-1, 4)
        if self.clip:
            output = np.clip(output, 0, 1)
        return output


class PostProcessor:
    def __init__(self, cfg, priors):
        self.num_classes = cfg['NUM_CLASSES']
        self.background_label = cfg['BACKGROUND_LABEL']
        self.conf_thresh = cfg['SCORE_THRESHOLD']
        self.nms_thresh = cfg['IOU_THRESHOLD']
        self.top_k = cfg['MAX_DETECTIONS']
        self.variance = cfg['VARIANCE']
        self.priors = priors

    def __call__(self, predictions):
        loc, conf = predictions
        num = loc.shape[0]
        conf_preds = conf.transpose([0, 2, 1])
        output = np.zeros(shape=(num, self.num_classes, self.top_k, 5))  # 15 = 1 label + 4 bbox + 10 landmarks

        for i in range(num):
            decoded_boxes = decode(loc[i], self.priors, self.variance)
            conf_scores = conf_preds[i]
            for cl in range(1, self.num_classes):
                c_mask = (conf_scores[cl] > self.conf_thresh)
                if np.all(c_mask == False):
                    continue
                scores = conf_scores[cl][c_mask]
                if len(scores) == 0:
                    continue
                boxes = decoded_boxes[c_mask, :]
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                if count > 0:
                    output[i, cl, :count] = \
                        np.hstack((scores[ids[:count]][:, np.newaxis],
                                   boxes[ids[:count]],
                                   ))
        return output
