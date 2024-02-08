import torch
from torch import nn
from pytorch_lightning import LightningModule
from src.utils.loss_fn import l_distil, l_lks, l_read
from tqdm import tqdm

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, use_act = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual
        self.use_act = use_act

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        
        if self.use_act:
            return self.act(out)
        else:
            return out

class SimpleWrapperV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )

        #### load the pre-trained audio_encoder 
        #self.audio_encoder = self.audio_encoder.to(device)  
        '''
        wav2lip_state_dict = torch.load('/apdcephfs_cq2/share_1290939/wenxuazhang/checkpoints/wav2lip.pth')['state_dict']
        state_dict = self.audio_encoder.state_dict()

        for k,v in wav2lip_state_dict.items():
            if 'audio_encoder' in k:
                print('init:', k)
                state_dict[k.replace('module.audio_encoder.', '')] = v
        self.audio_encoder.load_state_dict(state_dict)
        '''

        self.mapping1 = nn.Linear(512+64+1, 64)
        #self.mapping2 = nn.Linear(30, 64)
        #nn.init.constant_(self.mapping1.weight, 0.)
        nn.init.constant_(self.mapping1.bias, 0.)

    def forward(self, x, ref, ratio):
        # print('x : ', x.shape)
        # print('ref: ', ref.shape)
        # print('ratio: ', ratio.shape)
        x = self.audio_encoder(x).view(x.size(0), -1)
        ref_reshape = ref.reshape(x.size(0), -1)
        ratio = ratio.reshape(x.size(0), -1)
        
        y = self.mapping1(torch.cat([x, ref_reshape, ratio], dim=1)) 
        out = y.reshape(ref.shape[0], ref.shape[1], -1) #+ ref # resudial
        return out


# Define a LightningModule that extends PyTorch Lightning's LightningModule
class LightningMyModel(LightningModule):
    def __init__(self, model=None):
        super(LightningMyModel, self).__init__()
        self.model = SimpleWrapperV2() if model is None else model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, audiox, ref, ratio):
        # print('audiox : ', audiox.shape)
        # print('ref: ', ref.shape)
        # print('ratio: ', ratio.shape)
        # audiox, ref, ratio = x
        return self.model(audiox, ref, ratio)

    def training_step(self, data_batch, batch_idx=None):
        # labels = 'input/male1.mp4'
        # labels = torch.randn((1, 10, 64), requires_grad=True).to('cuda:0')
        # print(f'data_batch len: [{len(data_batch)}, {len(data_batch[0])}]')
        # print(data_batch)

        # Individual Features
        audiox = data_batch[0][0]   # processed audio
        ref = data_batch[1][0]      # 3dmm face landmark
        ratio = data_batch[2][0]    # z_blink
        label = data_batch[1][0]    # Label
        # print('audiox : ', audiox.shape)
        # print('ref: ', ref.shape)
        # print('ratio: ', ratio.shape)

        # Hyper-parameter
        lambda_distill = 2
        lambda_eye = 200
        lambda_lks = 0.01
        lambda_read = 0.01

        if False:
            for batch in data_batch:
                audiox = batch[0][0]
                ref = batch[1][0]
                ratio = batch[2][0]
                # print('audiox : ', audiox.shape)
                # print('ref: ', ref.shape)
                # print('ratio: ', ratio.shape)
                outputs = self.model(audiox, ref, ratio)
                # print('outputs shape: ', outputs.shape)
                loss = self.criterion(outputs, labels)
                # print('loss: ', loss)

        output = self.model(audiox, ref, ratio)
        # print('output shape: ', output.shape)

        loss_distil = l_distil(output[0], label[0])
        loss_lks = l_lks(ref[0], output[0], ratio[0], lambda_eye=lambda_eye)
        loss_read = l_read(output[0], label[0])
        loss = lambda_distill*loss_distil + lambda_lks*loss_lks + lambda_read*loss_read
        # print('loss: ', loss)
        # Log training metrics
        self.log("loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    def load_pretrained_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print("Keys in the checkpoint file:", checkpoint.keys())
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)