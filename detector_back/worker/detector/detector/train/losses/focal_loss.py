# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from detector.common.utils.box_utils import match


class FocalLoss(nn.Module):
    def __init__(self, cfg, priors, use_gpu=True):
        super(FocalLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.negpos_ratio = cfg.NEGPOS_RATIO
        self.threshold = cfg.MATCHED_THRESHOLD
        self.unmatched_threshold = cfg.UNMATCHED_THRESHOLD
        self.variance = cfg.VARIANCE
        self.priors = priors

        self.alpha = torch.ones(self.num_classes, 1) * cfg.ALPHA
        self.gamma = cfg.GAMMA

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        num = loc_data.size(0)
        priors = self.priors
        # priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(
                self.threshold,
                truths, defaults,
                self.variance,
                labels,
                loc_t,
                conf_t,
                idx
            )
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        pos = conf_t > 0
        num_pos = pos.sum()

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        loss_l /= num_pos.data.sum()

        # Confidence Loss (Focal loss)
        # Shape: [batch,num_priors,1]
        loss_c = self.reduced_focal_loss(
            conf_data.view(-1, self.num_classes),
            conf_t.view(-1, 1)
        )

        loss_c /= num_pos.data.sum()
        return loss_l, loss_c

    def focal_loss(self, inputs, targets):
        '''Focal loss.
        mean of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        '''
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        loss = batch_loss.mean()
        return loss

    def reduced_focal_loss(self, inputs, targets, thresh=0.5):
        '''Focal loss.
        mean of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        '''
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (class_mask * P).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = torch.pow((1 - probs + thresh), self.gamma)
        batch_loss[probs < thresh] = 1
        batch_loss = - alpha * batch_loss * log_p

        loss = batch_loss.sum()
        return loss