import os
from tqdm import tqdm
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
# importing poseVAE libraries
from src.audio2pose_models.audio2pose import Audio2Pose
# importing faceRenderer libraries

# data_loader libraries
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize
from moviepy.editor import VideoFileClip
# data proprocess
from PIL import Image
import cv2
from src.utils.preprocess import CropAndExtract
from src.generate_batch import get_data

device = 'cuda'
train = False

def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def create_batch(video_path):
        # Preprocess the input data
        # input_video = args.video_path

        # Extractingg Audio from input video
        if False and not os.path.exists(args.audio_path):
            video_clip = VideoFileClip(input_video)
            extracted_audio = video_clip.audio
            extracted_audio.write_audiofile('input/male1.mp3')
            args.audio_path = 'input/male1.mp3'

        if False and not os.path.exists(args.first_frame_path):
            first_frame = []
            video_stream = cv2.VideoCapture(input_video)
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break 
                first_frame = frame
                if len(first_frame)>=1:
                    video_stream.release()
                    break
            cv2.imwrite('input/first_frame.png', first_frame)
            args.first_frame_path = 'input/first_frame.png'
            

        sadtalker_paths = init_path('checkpoints', os.path.join(os.getcwd(), 'src/config'), 512, False, 'crop')
        preprocess_model = CropAndExtract(sadtalker_paths, args.device)
        first_coeff_path, _, _ =  preprocess_model.generate(video_path, 'results', 'crop', source_image_flag=True, pic_size=512)
        print('first_coeff_path: ', first_coeff_path)
        all_coeff_path, _, _ =  preprocess_model.generate(video_path, 'results', 'crop', source_image_flag=False, pic_size=512)
        print('all_coeff_path: ', all_coeff_path)
        batch = get_data(first_coeff_path, video_path, args.device, ref_eyeblink_coeff_path=None)
        label_batch = get_data(all_coeff_path, video_path, args.device, ref_eyeblink_coeff_path=None)
        print('batch shape: ', batch['ref'].shape)
        print('label_batch shape: ', label_batch['ref'].shape)

        tensor = []
        
        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]
        exp_coeff_pred = []
        for i in tqdm(range(0, T, 10),'audio2exp:'): # every 10 frames
            current_mel_input = mel_input[:,i:i+10]
            #ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['ref'][:, :, :64][:, i:i+10]
            label_ref = label_batch['ref'][:, :, :64][:, i:i+10]
            ratio = batch['ratio_gt'][:, i:i+10]                               #bs T
            audiox = current_mel_input.view(-1, 1, 80, 16)                  # bs*T 1 80 16
            if len(audiox) != 10:
                continue
            tensor.append([audiox, ref, ratio, label_ref])
            
        return tensor

def main(args):
    # SadTalker_paths
    checkpoint_dir = './checkpoints'
    current_root_path = os.getcwd()
    size = 256
    old_version = False
    preprocess = 'crop'
    sadtalker_path = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)

    # Load All configs
    if True:
        fcfg_pose = open(sadtalker_path['audio2pose_yaml_path'])
        cfg_pose = CN.load_cfg(fcfg_pose)
        cfg_pose.freeze()
        fcfg_exp = open(sadtalker_path['audio2exp_yaml_path'])
        cfg_exp = CN.load_cfg(fcfg_exp)
        cfg_exp.freeze()
        netG = SimpleWrapperV2()
        netG = netG.to(device)
        for param in netG.parameters():
            netG.requires_grad = True
        netG.eval()
        try:
            if sadtalker_path['use_safetensor']:
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
            else:
                load_cpk(sadtalker_path['audio2exp_checkpoint'], model=netG, device=device)
        except:
            raise Exception("Failed in loading audio2exp_checkpoint")
    else:
        cfg_exp = ''
        netG = SimpleWrapperV2()

    # Load Data in batches
    batch = 'load batch'

    # Init Models

    # 3DMM Extraction: image [c, w, h] -> keypoints(array)
    # image = ''
    # extract_3dmm = KeypointExtractor(device=device)
    # keypoints = extract_3dmm.extract_keypoint(image)

    # ExpNet
    if True:
        audio2exp = Audio2Exp(netG, cfg_exp, device=device, prepare_training_loss=False)
        audio2exp = audio2exp.to(device)
        # for param in audio2exp.parameters():
        #     param.requires_grad = False
        # audio2exp.eval()
        audio_batch = 'batch of audio'
        # exp_pred = audio2exp.generate(audio_batch)

    # PoseVAE
    if False:
        audio2pose = Audio2Pose(cfg_pose, None, device=device)
        audio2pose = audio2pose.to(device)
        audio2pose.eval()
        for param in audio2pose.parameters():
            param.requires_grad = False
        # pose_pre = audio2pose.generate


    # Load data_loader
    if args.random_dataset:
        print("Training with the Random Dataset...")
        ## sample: [['features'], ['labels']]*(dataset_size) = [[(10, 1, 80, 16), (1, 10, 64), (1, 10, 1)], [(1, 10, 64)]]*(dataset_size)
        ## dataset = transforms.Compose(transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))]))
        dataset = []
        dataset_len = 1
        print('generating dataset...')
        for _ in tqdm(range(dataset_len), 'Dataset Progress'):
            audiox = torch.randn((10, 1, 80, 16), requires_grad=True)
            ref = torch.randn((1, 10, 64), requires_grad=True)
            ratio = torch.randn((1, 10, 1), requires_grad=True)
            dataset.append([audiox, ref, ratio])
        # train_size = int(0.8 * len(dataset))
        # train_data, _ = random_split(dataset, [train_size, len(dataset) - train_size])
        # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    else:
        print("Training with the Actual Dataset...")
        dataset = []
        for idx, item in tqdm(enumerate(os.listdir(args.train_data)), "Loading Dataset: "):
            item_path = os.path.join(args.train_data, item)
            # print("item path:", item_path)
            dataset.append(create_batch(item_path))
            if idx >=8:
                break
        dataset = dataset*32
        # dataset = create_batch(args.video_path)*32
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print('dataset is generated and loaded successfully')

    # Start Training
    if args.train:
        cfg = ''
        # audiox = torch.randn((1, 10, 1, 80, 16), requires_grad=True)
        # ref = torch.randn((1, 1, 10, 64), requires_grad=True)
        # ratio = torch.randn((1, 1, 10, 1), requires_grad=True)
        # dataset = [[audiox], [ref], [ratio]]
        expNet = Audio2Exp(netG, cfg_exp, device=device)
        expNet.new_train(train_loader)


## Need to Configure
if __name__ == '__main__':
    parser = ArgumentParser()  

    parser.add_argument("--train", type=bool, default=True, help="start training") 
    parser.add_argument("--first_frame_path", type=str, default='input/first_frame.mp4', help="training video's first frame path") 
    parser.add_argument("--train_data", type=str, default='dataset/vox/train', help="training dataset video path") 
    parser.add_argument("--video_path", type=str, default='input/male1.mp4', help="training video path") 
    parser.add_argument("--audio_path", type=str, default='input/male1.mp3', help="training audio path") 
    parser.add_argument("--epoch", type=int, default=10, help="number of epoch") 
    parser.add_argument("--batch_size", type=int, default=8, help="batch size") 
    parser.add_argument("--random_dataset", type=bool, default=False, help="train with random tensors") 

    args = parser.parse_args()
    print('args: ', args)

    # python3 train.py --random_dataset False

    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)