import os
import yaml
import argparse
import numpy as np
import torch.utils
import torchvision.datasets as dset
import sys
import random
FILE_ABSOLUTE_PATH = os.path.abspath(__file__)
search_folder_path = os.path.dirname(FILE_ABSOLUTE_PATH)
src_path = os.path.dirname(search_folder_path)
robustdarts_path = os.path.dirname(src_path)
project_path = os.path.dirname(robustdarts_path)
dr_detection_dataset_path = os.path.join(project_path, 'DR_Detection', 'dataset')
dr_detection_data_path = os.path.join(project_path, 'DR_Detection', 'data')
malaria_dataset_path = os.path.join(project_path, 'cell_images')
sys.path.append(dr_detection_dataset_path)
sys.path.append(malaria_dataset_path)
print(sys.path)
from dataset import ImageLabelDataset, loadImageToTensor
from malaria_dataset import MalariaImageLabelDataset

from copy import copy
from src import utils


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser("DARTS evaluation")

        # general options
        parser.add_argument('--data',                    type=str,            default='./data',       help='location of the data corpus')
        parser.add_argument('--space',                   type=str,            default='s1',           help='space index')
        parser.add_argument('--dataset',                 type=str,            default='cifar10',      help='dataset')
        parser.add_argument('--search_wd',               type=float,          default=3e-4,           help='weight decay used during search')
        parser.add_argument('--search_dp',               type=float,          default=0.2,            help='drop path probability used during search')
        parser.add_argument('--gpu',                     type=int,            default=0,              help='gpu device id')
        parser.add_argument('--model_path',              type=str,            default=None,           help='path to save the model')
        parser.add_argument('--seed',                    type=int,            default=2,              help='random seed')
        parser.add_argument('--resume',                  action='store_true', default=False,          help='resume search')
        parser.add_argument('--debug',                   action='store_true', default=False,          help='use one-step unrolled validation loss')
        parser.add_argument('--job_id',                  type=int,            default=1,              help='SLURM_ARRAY_JOB_ID number')
        parser.add_argument('--task_id',                 type=int,            default=1,              help='SLURM_ARRAY_TASK_ID number')
        parser.add_argument('--search_task_id',          type=int,            default=1,              help='SLURM_ARRAY_TASK_ID number during search')

        # training options
        parser.add_argument('--epochs',                  type=int,            default=600,            help='num of training epochs')
        parser.add_argument('--batch_size',              type=int,            default=96,             help='batch size')
        parser.add_argument('--learning_rate',           type=float,          default=0.025,          help='init learning rate')
        parser.add_argument('--momentum',                type=float,          default=0.9,            help='momentum')
        parser.add_argument('--weight_decay',            type=float,          default=3e-4,           help='weight decay')
        parser.add_argument('--grad_clip',               type=float,          default=5,              help='gradient clipping')

        # one-shot model options
        parser.add_argument('--init_channels',           type=int,            default=16,             help='num of init channels')
        parser.add_argument('--layers',                  type=int,            default=8,              help='total number of layers')
        parser.add_argument('--auxiliary',               action='store_true', default=False,          help='use auxiliary tower')
        parser.add_argument('--auxiliary_weight',        type=float,          default=0.4,            help='weight for auxiliary loss')

        # augmentation options
        parser.add_argument('--cutout',                  action='store_true', default=False,          help='use cutout')
        parser.add_argument('--cutout_length',           type=int,            default=16,             help='cutout length')
        parser.add_argument('--cutout_prob',             type=float,          default=1.0,            help='cutout probability')
        parser.add_argument('--drop_path_prob',          type=float,          default=0.2,            help='drop path probability')

        # logging options
        parser.add_argument('--save',                    type=str,            default='experiments/eval_logs',    help='log directory name')
        parser.add_argument('--archs_config_file',       type=str,            default='./experiments/search_logs/results_arch.yaml', help='search logs directory')
        parser.add_argument('--results_test',            type=str,            default='results_perf', help='filename where to write test errors')
        parser.add_argument('--report_freq',             type=float,          default=50,             help='report frequency')

        # medical data file paths
        parser.add_argument('--valid_files', type=str, default=dr_detection_data_path + '/test_public_df.csv',    help='path to the csv file that contain validation images')
        parser.add_argument('--train_files', type=str, default=dr_detection_data_path + '/train_all_df.csv',      help='path to the csv file that contain train images')

        self.args = parser.parse_args()
        utils.print_args(self.args)


class Helper(Parser):
    def __init__(self):
        super(Helper, self).__init__()

        self.args._save = copy(self.args.save)
        self.args.save = '{}/{}/{}/{}_{}-{}'.format(self.args.save,
                                                    self.args.space,
                                                    self.args.dataset,
                                                    self.args.search_dp,
                                                    self.args.search_wd,
                                                    self.args.job_id)

        utils.create_exp_dir(self.args.save)

        config_filename = os.path.join(self.args._save, 'config.yaml')
        if not os.path.exists(config_filename):
            with open(config_filename, 'w') as f:
                yaml.dump(self.args_to_log, f, default_flow_style=False)

        if self.args.dataset == 'cifar100':
            self.args.n_classes = 100
        elif self.args.dataset == 'dr-detection':
            self.args.n_classes = 5
        elif self.args.dataset == 'malaria':
            self.args.n_classes = 2
        else:
            self.args.n_classes = 10


    @property
    def config(self):
        return self.args

    @property
    def args_to_log(self):
        list_of_args = [
            "epochs",
            "batch_size",
            "learning_rate",
            "momentum",
            "grad_clip",
            "init_channels",
            "layers",
            "cutout_length",
            "auxiliary",
            "auxiliary_weight",
            "archs_config_file",
        ]

        args_to_log = dict(filter(lambda x: x[0] in list_of_args,
                                  self.args.__dict__.items()))
        return args_to_log

    def get_train_val_loaders(self):
        if self.args.dataset == 'cifar10':
            train_transform, valid_transform = utils._data_transforms_cifar10(self.args)
            train_data = dset.CIFAR10(
                root=self.args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR10(
                root=self.args.data, train=False, download=True, transform=valid_transform)
        elif self.args.dataset == 'cifar100':
            train_transform, valid_transform = utils._data_transforms_cifar100(self.args)
            train_data = dset.CIFAR100(
                root=self.args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR100(
                root=self.args.data, train=False, download=True, transform=valid_transform)
        elif self.args.dataset == 'svhn':
            train_transform, valid_transform = utils._data_transforms_svhn(self.args)
            train_data = dset.SVHN(
                root=self.args.data, split='train', download=True, transform=train_transform)
            valid_data = dset.SVHN(
                root=self.args.data, split='test', download=True, transform=valid_transform)
        elif self.args.dataset == 'dr-detection':
            train_transform, valid_transform = utils._data_transforms_dr_detection(self.args)
            labels = {0: 'No DR', 1: 'Mild DR', 2: 'Moderate DR', 3: 'Severe DR', 4: 'Poliferative DR'}
            train_data = ImageLabelDataset(csv=self.args.train_files,
                                                  transform=train_transform,
                                                  label_names=labels)
            valid_data = ImageLabelDataset(csv=self.args.valid_files,
                                                  transform=valid_transform,
                                                  label_names=labels)
        elif self.args.dataset == 'malaria':
            train_transform, valid_transform = utils._data_transforms_malaria(self.args)
            train_data = MalariaImageLabelDataset(transform=train_transform, shuffle=True)
            valid_data = train_data
            train_portion = 0.9
            num_train = len(train_data)
            indices = random.sample(range(num_train), num_train)
            split = int(np.floor(train_portion * num_train))

        if self.args.dataset == 'dr-detection':
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size,
                sampler=torch.utils.data.sampler.RandomSampler(train_data), pin_memory=True, num_workers=2)

            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size,
                sampler=torch.utils.data.sampler.RandomSampler(valid_data), pin_memory=True, num_workers=2)
        elif self.args.dataset == 'malaria':
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                pin_memory=True, num_workers=2)

            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                pin_memory=True, num_workers=2)
        else:
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size,
                shuffle=True, pin_memory=True, num_workers=2)

            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size,
                shuffle=False, pin_memory=True, num_workers=2)

        return train_queue, valid_queue, train_transform, valid_transform
