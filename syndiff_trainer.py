import argparse
import torch
import numpy as np

import os

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from dataset import CreateDatasetSynthesis

from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr


from backbones.discriminator import Discriminator_small, Discriminator_large
from backbones.ncsnpp_generator_adagn import NCSNpp
import backbones.generator_resnet 
from utils.EMA import EMA


class SynDiffTrainer:
    def __init__(self, config):
        self.config = config

        torch.manual_seed(self.config.seed + rank)
        torch.cuda.manual_seed(self.config.seed + rank)
        torch.cuda.manual_seed_all(self.config.seed + rank)

        # Create dataset
        self.train_dataset = CreateDatasetSynthesis(config, mode='train')
        self.val_dataset = CreateDatasetSynthesis(config, mode='val')

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )