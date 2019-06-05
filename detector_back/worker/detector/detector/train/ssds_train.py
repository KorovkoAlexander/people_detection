from __future__ import print_function
import os
import sys

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init

from tensorboardX import SummaryWriter

from detector.common.utils.timer import Timer
from detector.common.functions.detection import Detect
from detector.common.models.model_builder import create_model
from detector.train.dataset.dataset_factory import load_data
from detector.train.losses import FocalLoss


class Solver(object):
    """
    A wrapper class for the training process
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load data
        print('===> Loading data')
        self.train_loader = load_data(cfg.DATASET, 'train') if 'train' in cfg.PHASE else None
        self.eval_loader = load_data(cfg.DATASET, 'eval') if 'eval' in cfg.PHASE else None
        self.test_loader = load_data(cfg.DATASET, 'test') if 'test' in cfg.PHASE else None
        self.visualize_loader = load_data(cfg.DATASET, 'visualize') if 'visualize' in cfg.PHASE else None

        # Build model
        print('===> Building model')
        self.model, self.priorbox = create_model(cfg.MODEL)
        self.priors = self.priorbox.forward()
        self.detector = Detect(cfg.POST_PROCESS, self.priors)

        # Print the model architecture and parameters
        #print('Model architectures:\n{}\n'.format(self.model))
        #print('Trainable scope: {}'.format(cfg.TRAIN.TRAINABLE_SCOPE))
        print(cfg)
        trainable_param = self.model.parameters() # self.trainable_param(cfg.TRAIN.TRAINABLE_SCOPE)

        # Utilize GPUs for computation
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print('Utilize GPUs for computation')
            print('Number of GPU available', self.use_gpu)
            self.priors.cuda()
            self.model = torch.nn.DataParallel(self.model.cuda(), device_ids=list(range(cfg.NGPU)))
            #self.model = self.model.cuda()

        self.optimizer = self.configure_optimizer(trainable_param)
        self.exp_lr_scheduler = self.configure_lr_scheduler(self.optimizer)
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS

        # metric
        self.criterion = FocalLoss(cfg.MATCHER, self.priors, self.use_gpu)

        # Set the logger
        self.writer = SummaryWriter(log_dir=cfg.LOG_DIR)
        self.test_writer = SummaryWriter(log_dir=os.path.join(cfg.LOG_DIR, "test"))
        self.output_dir = cfg.EXP_DIR
        self.checkpoint = cfg.RESUME_CHECKPOINT
        self.checkpoint_prefix = cfg.CHECKPOINTS_PREFIX

    def save_checkpoints(self, epochs, iters=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if iters:
            filename = self.checkpoint_prefix + '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
        else:
            filename = self.checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        torch.save(self.model.module.state_dict(), filename)
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'a') as f:
            f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
        print('Wrote snapshot to: {:s}'.format(filename))

        # TODO: write relative cfg under the same page

    def resume_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint)

        # remove the module in the parrallel model
        if 'module.' in list(checkpoint.items())[0][0]:
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            checkpoint = pretrained_dict

        resume_scope = self.cfg.TRAIN.RESUME_SCOPE
        # extract the weights based on the resume scope
        if resume_scope != '':
            pretrained_dict = {}
            for k, v in list(checkpoint.items()):
                for resume_key in resume_scope.split(','):
                    if resume_key in k:
                        pretrained_dict[k] = v
                        break
            checkpoint = pretrained_dict

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.model.state_dict()}

        checkpoint = self.model.state_dict()

        unresume_dict = set(checkpoint)-set(pretrained_dict)
        if len(unresume_dict) != 0:
            print("=> UNResume weigths:")
            print(unresume_dict)

        checkpoint.update(pretrained_dict)

        return self.model.load_state_dict(checkpoint)

    def find_previous(self):
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
            checkpoint = line[line.find(':') + 2:-1]
            epoches.append(epoch)
            resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints

    def weights_init(self, m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    def initialize(self):
        if self.checkpoint:
            print('Loading initial model weights from {:s}'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)

        start_epoch = 0
        return start_epoch

    def trainable_param(self, trainable_scope):
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_param = []
        for module in trainable_scope.split(','):
            if hasattr(self.model, module):
                # print(getattr(self.model, module))
                for param in getattr(self.model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(self.model, module).parameters())

        return trainable_param

    def train_model(self):
        previous = self.find_previous()
        if previous:
            start_epoch = previous[0][-1]
            self.resume_checkpoint(previous[1][-1])
        else:
            start_epoch = self.initialize()

        # export graph for the model, onnx always not works
        # self.export_graph()

        # warm_up epoch
        warm_up = self.cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
        for epoch in iter(range(start_epoch+1, self.max_epochs+1)):
            #learning rate
            sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.max_epochs))
            if epoch > warm_up:
                self.exp_lr_scheduler.step(epoch-warm_up)
            if 'train' in self.cfg.PHASE:
                self.train_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.writer, epoch, self.use_gpu)
            if 'test' in self.cfg.PHASE:
                self.test_epoch(self.model, self.test_loader, self.criterion, self.test_writer, epoch, self.use_gpu)

            if epoch % self.cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
                self.save_checkpoints(epoch)


    def train_epoch(self, model, data_loader, optimizer, criterion, writer, epoch, use_gpu):
        model.train()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        for iteration in iter(range((epoch_size))):
            images, targets = next(batch_iterator)
            if use_gpu:
                images = images.cuda()
                targets = [anno.cuda() for anno in targets]
            _t.tic()
            # forward
            out = model(images, phase='train')

            # backprop
            optimizer.zero_grad()
            try:
                loss_l, loss_c = criterion(out, targets)
            except RuntimeError:
                continue

            # some bugs in coco train2017. maybe the annonation bug.
            if loss_l.item() == float("Inf"):
                continue

            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            time = _t.toc()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            # log per iter
            log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.item(), cls_loss=loss_c.item())

            sys.stdout.write(log)
            sys.stdout.flush()

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        lr = optimizer.param_groups[0]['lr']
        log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\n'.format(lr=lr,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Train/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Train/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Train/lr', lr, epoch)


    def test_epoch(self, model, data_loader, criterion, writer, epoch, use_gpu):
        model.eval()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        with torch.no_grad():
            for iteration in iter(range((epoch_size))):
                images, targets = next(batch_iterator)
                if use_gpu:
                    images = images.cuda()
                    targets = [anno.cuda() for anno in targets]
                _t.tic()
                # forward
                out = model(images, phase='train')

                try:
                    loss_l, loss_c = criterion(out, targets)
                except RuntimeError:
                    continue

                # some bugs in coco train2017. maybe the annonation bug.
                if loss_l.item() == float("Inf"):
                    continue

                time = _t.toc()
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()

                # log per iter
                log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#' * int(round(10 * iteration / epoch_size)) + '-' * int(
                        round(10 * (1 - iteration / epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.item(), cls_loss=loss_c.item())

                sys.stdout.write(log)
                sys.stdout.flush()

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f}\n'.format(
            time=_t.total_time, loc_loss=loc_loss / epoch_size, conf_loss=conf_loss / epoch_size)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Train/loc_loss', loc_loss / epoch_size, epoch)
        writer.add_scalar('Train/conf_loss', conf_loss / epoch_size, epoch)


    def configure_optimizer(self, trainable_param):
        cfg = self.cfg.TRAIN.OPTIMIZER
        if cfg.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(trainable_param,
                                  lr=cfg.LEARNING_RATE,
                                  momentum=cfg.MOMENTUM,
                                  weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'rmsprop':
            optimizer = optim.RMSprop(trainable_param,
                                      lr=cfg.LEARNING_RATE,
                                      momentum=cfg.MOMENTUM,
                                      alpha=cfg.MOMENTUM_2,
                                      eps=cfg.EPS,
                                      weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'adam':
            optimizer = optim.Adam(trainable_param,
                                   lr=cfg.LEARNING_RATE,
                                   betas=(cfg.MOMENTUM, cfg.MOMENTUM_2),
                                   eps=cfg.EPS,
                                   weight_decay=cfg.WEIGHT_DECAY)
        else:
            AssertionError('optimizer can not be recognized.')
        return optimizer


    def configure_lr_scheduler(self, optimizer):
        cfg = self.cfg.TRAIN.LR_SCHEDULER
        if cfg.SCHEDULER == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.STEPS[0], gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'multi_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.STEPS, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'exponential':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'SGDR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MAX_EPOCHS)
        else:
            AssertionError('scheduler can not be recognized.')
        return scheduler


def train_model(cfg):
    s = Solver(cfg)
    s.train_model()
    return True
