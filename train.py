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
# import faceRenderer

device = 'cuda'
train = False

def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def main(args):
    # SadTalker_paths
    checkpoint_dir = './checkpoints'
    current_root_path = os.getcwd()
    size = 512
    old_version = False
    preprocess = 'crop'
    sadtalker_path = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)

    # Load All configs
    fcfg_pose = open(sadtalker_path['audio2pose_yaml_path'])
    cfg_pose = CN.load_cfg(fcfg_pose)
    cfg_pose.freeze()
    fcfg_exp = open(sadtalker_path['audio2exp_yaml_path'])
    cfg_exp = CN.load_cfg(fcfg_exp)
    cfg_exp.freeze()
    netG = SimpleWrapperV2()
    netG = netG.to(device)
    for param in netG.parameters():
        netG.requires_grad = False
    netG.eval()
    try:
        if sadtalker_path['use_safetensor']:
            checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
            netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
        else:
            load_cpk(sadtalker_path['audio2exp_checkpoint'], model=netG, device=device)
    except:
        raise Exception("Failed in loading audio2exp_checkpoint")

    # Load Data in batches
    batch = 'load batch'

    # Init Models

    # 3DMM Extraction: image [c, w, h] -> keypoints(array)
    # image = ''
    # extract_3dmm = KeypointExtractor(device=device)
    # keypoints = extract_3dmm.extract_keypoint(image)

    # ExpNet
    audio2exp = Audio2Exp(netG, cfg_exp, device=device, prepare_training_loss=False)
    audio2exp = audio2exp.to(device)
    # for param in audio2exp.parameters():
    #     param.requires_grad = False
    # audio2exp.eval()
    audio_batch = 'batch of audio'
    # exp_pred = audio2exp.generate(audio_batch)

    # PoseVAE
    audio2pose = Audio2Pose(cfg_pose, None, device=device)
    audio2pose = audio2pose.to(device)
    audio2pose.eval()
    for param in audio2pose.parameters():
        param.requires_grad = False
    # pose_pre = audio2pose.generate

    # Start Training
    if args.train:
        cfg = ''
        audiox = torch.randn((1, 10, 1, 80, 16), requires_grad=True)
        ref = torch.randn((1, 1, 10, 64), requires_grad=True)
        ratio = torch.randn((1, 1, 10, 1), requires_grad=True)
        dataset = [[audiox], [ref], [ratio]]
        expNet = Audio2Exp(netG, cfg_exp, device=device)
        expNet.new_train(dataset)


if __name__ == '__main__':
    
    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 
    parser.add_argument("--train", type=bool, default=True, help="start training" ) 


    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)