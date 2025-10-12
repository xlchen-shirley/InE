import cv2
from net_Res_double import Encode, Decode, Common
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def transnet(data_vi, data_ir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Encoder_1 = Encode().to(device)
    Decoder_1 = Decode().to(device)
    Attention = Common().to(device)
    # Encoder_1 = nn.DataParallel(Encode()).to(device)
    # Decoder_1 = nn.DataParallel(Decode()).to(device)
    # Attention = nn.DataParallel(Common()).to(device)
    ckpt_path = r"./model/VI2IR119.pth"
    Encoder_1.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder_1.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    Attention.load_state_dict(torch.load(ckpt_path)['Attention'])
    Encoder_1.eval()
    Decoder_1.eval()
    Attention.eval()
    data_VIS, data_IR = data_vi, data_ir
    feature_V, feature_I = Encoder_1(data_VIS, data_IR)
    feature_V_SAM, feature_I_SAM = Attention(feature_V, feature_I)

    data_VIS_I, data_I_VIS = Decoder_1(feature_V_SAM, feature_I_SAM, stage=False)
    return data_VIS_I, data_I_VIS
