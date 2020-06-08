import os
import sys
import time
import glob
import numpy as np
import torch
import json
import codecs
import yaml
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, '../RobustDARTS')

from src import utils
from src.utils import Genotype
from src.evaluation.model import Network
from src.evaluation.args import Helper

TORCH_VERSION = torch.__version__


helper = Helper()
args = helper.config

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save,
                                      'log_{}_{}.txt'.format(args.search_task_id, args.task_id)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# add here torch version >= 1.0
if TORCH_VERSION.startswith('1'):
    device = torch.device('cuda:{}'.format(args.gpu))

class NetworkExtension(nn.Module):

  def __init__(self, orig_num_classes, num_classes, auxiliary):
    super(NetworkExtension, self).__init__()
    self._auxiliary = auxiliary
    self.classifier = nn.Linear(orig_num_classes ,num_classes)

  def forward(self, logits_logits_aux):
    logits = logits_logits_aux[0]
    logits_aux = logits_logits_aux[1]
    if self._auxiliary and self.training:
      logits_aux = torch.sigmoid(self.classifier(logits_aux))
    logits = torch.sigmoid(self.classifier(logits))
    return logits, logits_aux

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  if TORCH_VERSION.startswith('1'):
    torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # load search configuration file holding the found architectures
  if args.dataset == 'dr-detection':
    configuration = '_'.join([args.space, 'malaria']) #transer learning from cifar10
  else:
    configuration = '_'.join([args.space, args.dataset])
  settings\
    = '_'.join([str(args.search_dp), str(args.search_wd)])
  with open(args.archs_config_file, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)
    arch = dict(cfg)[configuration][settings][args.search_task_id]

  print(arch)
  genotype = eval(arch)
  if args.dataset == 'malaria':
    darts_model = Network(args.init_channels, 1000, args.layers, args.auxiliary, genotype)
    extension_model = NetworkExtension(1000, 2, args.auxiliary)
    model = nn.Sequential(darts_model,extension_model)
  else :
    model = Network(args.init_channels, args.n_classes, args.layers, args.auxiliary, genotype)

  if TORCH_VERSION.startswith('1'):
    model = model.to(device)
  else:
    model = model.cuda()

  if args.model_path is not None:
    utils.load(model, args.model_path, genotype)


  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  if TORCH_VERSION.startswith('1'):
    criterion = criterion.to(device)
  else:
    criterion = criterion.cuda()

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  scheduler = CosineAnnealingLR(
      optimizer, float(args.epochs))

  train_queue, valid_queue, _, _ = helper.get_train_val_loaders()

  errors_dict = {'train_acc': [], 'train_loss': [], 'valid_acc': [],
                 'valid_loss': []}

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    darts_model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    # training
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    # evaluation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    # update the errors dictionary
    errors_dict['train_acc'].append(100 - train_acc)
    errors_dict['train_loss'].append(train_obj)
    errors_dict['valid_acc'].append(100 - valid_acc)
    errors_dict['valid_loss'].append(valid_obj)

  if args.dataset == 'malaria':
    torch.save(darts_model.state_dict(), args.save)
  else:
    torch.save(model.state_dict(), args.save)
  with codecs.open(os.path.join(args.save,
                                'errors_{}_{}.json'.format(args.search_task_id, args.task_id)),
                   'w', encoding='utf-8') as file:
    json.dump(errors_dict, file, separators=(',', ':'))

  utils.write_yaml_results_eval(args, args.results_test, 100-valid_acc)

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, input_target in enumerate(train_queue):

    if args.dataset == 'dr-detection':
        input = input_target['image']
        target = input_target['label']
    else:
        input = input_target[0]
        target = input_target[1]

    if TORCH_VERSION in ['1.0.1', '1.1.0']:
      input = input.to(device)
      target = target.to(device)
    else:
      input = Variable(input).cuda()
      target = Variable(target).cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()

    if TORCH_VERSION.startswith('1'):
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    else:
      nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    if args.dataset == 'malaria' or args.dataset == 'dr-detection':
      prec1 = utils.accuracy(logits, target)
      prec1 = prec1[0]
      n = input.size(0)
      if TORCH_VERSION.startswith('1'):
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
      else:
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)

      if step % args.report_freq == 0:
        logging.info('train %03d %e %f', step, objs.avg, top1.avg)
        if args.debug:
          break
    else:
      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      if TORCH_VERSION.startswith('1'):
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
      else:
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

      if step % args.report_freq == 0:
        logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        if args.debug:
          break

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  if TORCH_VERSION.startswith('1'):
    with torch.no_grad():
      for step, input_target in enumerate(valid_queue):

        if args.dataset == 'dr-detection':
          input = input_target['image']
          target = input_target['label']
        else:
          input = input_target[0]
          target = input_target[1]

        if TORCH_VERSION in ['1.0.1', '1.1.0']:
          input = input.to(device)
          target = target.to(device)
        else:
          input = Variable(input).cuda()
          target = Variable(target).cuda()

        logits, _ = model(input)
        loss = criterion(logits, target)
        if args.dataset == 'malaria' or args.dataset == 'dr-detection':
          prec1 = utils.accuracy(logits, target)
          prec1 = prec1[0]
          n = input.size(0)
          objs.update(loss.item(), n)
          top1.update(prec1.item(), n)

          if step % args.report_freq == 0:
            logging.info('valid %03d %e %f', step, objs.avg, top1.avg)
            if args.debug:
              break
        else:
          prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
          n = input.size(0)
          objs.update(loss.item(), n)
          top1.update(prec1.item(), n)
          top5.update(prec5.item(), n)

          if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if args.debug:
              break
  else:
    for step, input_target in enumerate(valid_queue):
      if args.dataset == 'dr-detection':
        input = Variable(input_target['image'], requires_grad=False).cuda()
        target = Variable(input_target['label'], requires_grad=False).cuda()
      else:
        input = Variable(input_target[0], requires_grad=False).cuda()
        target = Variable(input_target[1], requires_grad=False).cuda()

      logits, _ = model(input)
      loss = criterion(logits, target)

      if args.dataset == 'malaria' or args.dataset == 'dr-detection':
        prec1 = utils.accuracy(logits, target)
        prec1 = prec1[0]
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f', step, objs.avg, top1.avg)
          if args.debug:
            break
      else:
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
          if args.debug:
            break

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

