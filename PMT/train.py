# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''
import os
from matplotlib import pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from PMT_Net import Encode, Decode, Common

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kornia
from data_loader.dataset import H5Dataset

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


model_str = 'double_VI2IR'

# . Set the hyper-parameters for training
num_epochs = 120  # total epoch
epoch_gap = 40  # epoches of Phase I

lr = 1e-4
weight_decay = 0
batch_size = 8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1.  # alpha1
coeff_mse_loss_IF = 1.
coeff_tv = 2.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5
num = 1

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = Encode().to(device)
DIDF_Decoder = Decode().to(device)
Attention = Common().to(device)

def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()


# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    Attention.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')

# ckpt_path=r"/public/home/zhuyu/chl_train/paper/add_Res/VI2IR35.pth"
# DIDF_Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
# DIDF_Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
# Attention.load_state_dict(torch.load(ckpt_path)['Attention'])

# data loader
trainloader = DataLoader(H5Dataset(r"../Enocder_train/data/MSRS_train_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

''' `
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(0, 120):
    i = 0
    ''' train '''
    for i, (data_VIS, data_IR, data_VIS_C) in enumerate(loader['train']):
        i += 1
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        # data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        Attention.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        Attention.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        if epoch < epoch_gap:  # Phase I
            feature_V, feature_I = DIDF_Encoder(data_VIS, data_IR)

            feature_V_SAM, feature_I_SAM = Attention(feature_V, feature_I)

            data_VIS_hat, data_IR_hat = DIDF_Decoder(feature_V_SAM, feature_I_SAM, stage=True)

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))
        
            cc_loss = 1 - torch.abs(cc(feature_V_SAM, feature_I_SAM))

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_tv * Gradient_loss + cc_loss * 50

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                Attention.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
        else:  # Phase II
            feature_V, feature_I = DIDF_Encoder(data_VIS, data_IR)
            feature_V_SAM, feature_I_SAM = Attention(feature_V, feature_I)
            data_VIS_I, data_I_VIS  = DIDF_Decoder(feature_V_SAM, feature_I_SAM, stage=False)

            Gradient_loss_vi = L1Loss(kornia.filters.SpatialGradient()(data_IR),kornia.filters.SpatialGradient()(data_VIS_I))
            mse_loss_V =  Loss_ssim(data_IR, data_VIS_I) + MSELoss(data_IR, data_VIS_I) + 2 * Gradient_loss_vi
            
            Gradient_loss_ir = L1Loss(kornia.filters.SpatialGradient()(data_VIS),kornia.filters.SpatialGradient()(data_I_VIS))
            mse_loss_I = Loss_ssim(data_VIS, data_I_VIS) + MSELoss(data_VIS, data_I_VIS) + 2 * Gradient_loss_ir
            loss = mse_loss_V + mse_loss_I

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()

        # Determine approximate time left
        batches_done = epoch *len(trainloader) + i
        batches_left = num_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(trainloader),
                loss.item(),
                time_left,
            )
        )

    # adjust the learning rate
		
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    if not epoch < epoch_gap:
        scheduler1.step()
        scheduler2.step()


    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6

    if epoch % 5 == 0:
        checkpoint = {
            'DIDF_Encoder': DIDF_Encoder.state_dict(),
            'DIDF_Decoder': DIDF_Decoder.state_dict(),
            'Attention': Attention.state_dict(),
        }
        torch.save(checkpoint, os.path.join(r"./models", str(epoch) + '.pth'))

if True:
    checkpoint = {
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'Attention': Attention.state_dict(),
    }
    torch.save(checkpoint, os.path.join(r"./models", str(epoch) + '.pth'))


