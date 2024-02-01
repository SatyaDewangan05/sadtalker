import os
from argparse import ArgumentParser
# import 3DMM Extractor
from src.face3d.extract_kp_videos import KeypointExtractor
# import ExpNet
from src.audio2exp_models.audio2exp import Audio2Exp
from src.audio2exp_models.networks import SimpleWrapperV2
from src.audio2exp_models.networks import LightningMyModel

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningModule, Trainer

import safetensors.torch
from src.utils.init_path import init_path
from src.utils.safetensor_helper import load_x_from_safetensor
from yacs.config import CfgNode as CN
# import poseVAE
from src.audio2pose_models.audio2pose import Audio2Pose

def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def load_pretrained_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    # Optionally, load other components such as optimizer and scheduler if they are saved in the checkpoint.
    # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])



# netG = SimpleWrapperV2()
# lightning_model = LightningMyModel(netG)

# ckpt_path = "/home/ubuntu/sadtalker/lightning_logs/version_1/checkpoints/epoch=9-step=10.ckpt"
# checkpoint = torch.load(ckpt_path, map_location=torch.device('cuda'))

# model = lightning_model.model.load_state_dict(ckpt_path)

# # disable randomness, dropout, etc...
# model.eval()


# Instantiate your LightningMyModel
model = LightningMyModel()

# Load pretrained weights
print('loading model...')
pretrained_checkpoint_path = "/home/ubuntu/sadtalker/lightning_logs/version_1/checkpoints/epoch=9-step=10.ckpt"
model.load_pretrained_checkpoint(pretrained_checkpoint_path)
print('model loaded!!!')

audiox = torch.randn((10, 1, 80, 16), requires_grad=True)
ref = torch.randn((1, 10, 64), requires_grad=True)
ratio = torch.randn((1, 10, 1), requires_grad=True)

print(model(audiox, ref, ratio).shape)