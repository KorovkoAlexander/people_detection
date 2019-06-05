from __future__ import print_function
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from detector.common.models.model_builder import create_model
from detector.common.functions.detection import Detect


def preproc_for_test(image, insize):
    image = cv2.resize(image, (insize[0], insize[1]))
    image = image.transpose(2, 0, 1) / 255.0
    image = image.astype(np.float32)
    return image


class ObjectDetector:
    def __init__(self, cfg, checkpoint_path, device_id):
        self.cfg = cfg
        self.device_id = device_id
        self.model, self.priorbox = create_model(cfg.MODEL)
        self.priors = self.priorbox.forward()

        # Utilize GPUs for computation
        self.use_gpu = True
        self.half = False
        if self.use_gpu:
            self.model = self.model.cuda(device_id)
            self.priors = self.priors.cuda(device_id)
            cudnn.benchmark = True
            #self.half = cfg.MODEL.HALF_PRECISION
            if self.half:
                self.model = self.model.half()
                self.priors = self.priors.half()
                self.model = self.model.cuda(device_id)

        # Build preprocessor and detector
        self.detector = Detect(cfg.POST_PROCESS, self.priors)

        # Load weight:
        checkpoint = torch.load(checkpoint_path)
        # checkpoint = torch.load(cfg.RESUME_CHECKPOINT, map_location='gpu' if self.use_gpu else 'cpu')
        self.model.load_state_dict(checkpoint)

        # test only
        self.model.eval()

    def restore_bboxes(self, detections, threshold, scale):
        # output
        labels, scores, coords = [list() for _ in range(3)]
        batch = 0
        for classes in range(detections.size(1)):
            num = 0
            while detections[batch, classes, num, 0] >= threshold:
                score = detections[batch, classes, num, 0]
                coord = detections[batch, classes, num, 1:5] * scale

                if self.use_gpu:
                    score = score.cpu()
                    coord = coord.cpu()

                scores.append(score.numpy().tolist())
                labels.append(classes - 1)
                coords.append(coord.numpy().tolist())
                num += 1
                if num >= detections.size(2):
                    break
        return labels, scores, coords

    def predict(self, img, threshold=0.3):
        # make sure the input channel is 3 
        assert img.shape[2] == 3
        scale = torch.Tensor([img.shape[1::-1], img.shape[1::-1]]).view(-1)
        x = torch.from_numpy(preproc_for_test(img, self.cfg.DATASET.IMAGE_SIZE)).unsqueeze(0)
        if self.use_gpu:
            x = x.cuda(self.device_id)
        if self.half:
            x = x.half()

        # forward
        with torch.no_grad():
            out = self.model(x, phase='eval')  # forward pass

            # detect
            detections = self.detector.forward(out)

        # output
        labels, scores, coords = self.restore_bboxes(detections, threshold, scale)
        return labels, scores, coords


class BatchObjectDetector(ObjectDetector):
    def predict(self, imgs, scales, threshold=0.6):
        images = torch.from_numpy(imgs)
        if self.use_gpu:
            images = images.cuda(self.device_id)
        if self.half:
            images = images.half()

        # forward
        detections = []
        with torch.no_grad():
            out = self.model(images)  # forward pass

            # detect
            for x in out:
                x = x.unsqueeze(0)
                detections.append(
                    self.detector.forward(x)
                )

        # output
        labels_list, scores_list, coords_list = (list() for _ in range(3))
        for i, detections_per_image in enumerate(detections):
            labels, scores, coords = self.restore_bboxes(
                detections_per_image,
                threshold,
                scales[i]
            )
            labels_list.append(
                labels
            )
            scores_list.append(
                scores
            )
            coords_list.append(
                coords
            )

        return labels_list, scores_list, coords_list
