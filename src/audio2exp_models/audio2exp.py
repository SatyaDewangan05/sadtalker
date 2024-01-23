from torch.nn.modules.module import T
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Utils
import os
import numpy as np
from time import  strftime
from src.utils.init_path import init_path
from src.utils.preprocess import CropAndExtract
from src.generate_batch import get_data
from src.audio2exp_models.networks import LightningMyModel


class Audio2Exp(nn.Module):
    def __init__(self, netG, cfg, device, prepare_training_loss=False):
        super(Audio2Exp, self).__init__()
        self.cfg = cfg
        self.device = device
        self.netG = netG.to(device)

    def get_samples(self, args):
        sample_args = ['checkpoint_dir', 'size', 'old_version', 'preprocess', 'result_dir', 'pic_path', 'first_frame_dir', 'driven_audio', 'ref_eyeblink', 'ref_eyeblink_frame_dir']
        sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(os.getcwd(), 'src/config'), args.size, args.old_version, args.preprocess)
        preprocess_model = CropAndExtract(sadtalker_paths, self.device)
        save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"), 'first_frame_dir')
        first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(args.pic_path, args.first_frame_dir, args.preprocess,\
                                                                             source_image_flag=True, pic_size=args.size)
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(args.ref_eyeblink, args.ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
        return first_coeff_path, args.driven_audio. ref_eyeblink_coeff_path
        
    def get_batch(self, batch, split=10):
        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        exp_coeff_pred = []
        new_batch = []
        for i in tqdm(range(0, T, split),'audio2exp:'):         # every 'split=10' frames
            
            current_mel_input = mel_input[:,i:i+split]
            ref = batch['ref'][:, :, :64][:, i:i+split]
            ratio = batch['ratio_gt'][:, i:i+split]             # bs T

            audiox = current_mel_input.view(-1, 1, 80, 16)      # bs*T 1 80 16
            break
            new_batch.append((audiox, ref, ratio))
        # new_batch = np.array(new_batch.cpu())
        # return new_batch
        print('at_get_batch')
        print('audiox: ', audiox.shape)
        print('ref: ', ref.shape)
        print('ratio: ', ratio.shape)
        return [[audiox.unsqueeze(0)], [ref.unsqueeze(0)], [ratio.unsqueeze(0)]]

    def new_train(self, batch, args=None, dataset=None):
        print('Starting Training...')
        # args.max_epochs = 50
        # args.devices = 5
        # args.accelerator = "gpu"
        # args.strategy = "ddp"
        # args.check_val_every_n_epoch = 10

        max_epochs = 50
        devices = 1
        accelerator = "gpu"
        # strategy = "ddp"
        check_val_every_n_epoch = 10

        lightning_model = LightningMyModel(self.netG)

        # # Define your dataset and DataLoader
        # # (1)
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # # (2)
        # dataset = transforms.Compose(transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))]))
        # train_size = int(0.8 * len(dataset))
        # train_data, _ = random_split(dataset, [train_size, len(dataset) - train_size])
        # train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

        # Data Extraction
        # sample_args = ['checkpoint_dir', 'size', 'old_version', 'preprocess', 'result_dir', 'pic_path', 'first_frame_dir', 'driven_audio', 'ref_eyeblink', 'ref_eyeblink_frame_dir']
        # first_coeff_path, audio_path, ref_eyeblink_coeff_path = self.get_samples(args)
        # batch = get_data(first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path, still=args.still)
        # train_loader = self.get_batch(batch)
        # train_loader = [batch]
        print('at train_loader')
        train_loader = [batch]*128
        print('audiox: ', train_loader[0][0][0].shape)
        print('ref: ', train_loader[0][1][0].shape)
        print('ratio: ', train_loader[0][2][0].shape)

        # default used by the Trainer
        # sampler = torch.utils.data.DistributedSampler(train_loader, shuffle=True)
        # dataloader = DataLoader(train_loader, batch_size=32, shuffle=True)

        # train_loader = [train_loader]*64
        # print('train_loader size', train_loader[0].size())

        # python inference.py --driven_audio input/input_audio.wav --source_image input/male1.jpeg
        # python train.py --driven_audio input/input_audio.wav --source_image input/male1.jpeg

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='simplewrapperV2-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min'
        )

        # Instantiate the Lightning Trainer and train the model
        trainer = Trainer(max_epochs=max_epochs,
                devices=devices,
                accelerator=accelerator,
                enable_progress_bar=True,
                check_val_every_n_epoch=check_val_every_n_epoch,
                callbacks=[checkpoint_callback])
        
        trainer.fit(lightning_model, train_loader)

    def test(self, batch):

        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10),'audio2exp:'): # every 10 frames
            
            current_mel_input = mel_input[:,i:i+10]

            #ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['ref'][:, :, :64][:, i:i+10]
            ratio = batch['ratio_gt'][:, i:i+10]                               #bs T

            audiox = current_mel_input.view(-1, 1, 80, 16)                  # bs*T 1 80 16

            curr_exp_coeff_pred  = self.netG(audiox, ref, ratio)         # bs T 64 

            exp_coeff_pred += [curr_exp_coeff_pred]

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': torch.cat(exp_coeff_pred, axis=1)
            }
        return results_dict


