import time

import cv2

from PMT_Net import Encode, Decode, Common
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
total_time = 0
def bgr_to_ycrcb(path):
    one = cv2.imread(path,1)
    one = one.astype('float32')
    (B, G, R) = cv2.split(one)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    return Y, cv2.merge([Cr,Cb])

def ycrcb_to_bgr(one):
    one = one.astype('float32')
    Y, Cr, Cb = cv2.split(one)
    B = (Cb - 0.5) * 1. / 0.564 + Y
    R = (Cr - 0.5) * 1. / 0.713 + Y
    G = 1. / 0.587 * (Y - 0.299 * R - 0.114 * B)
    return cv2.merge([B, G, R])

for dataset_name in ["128"]:
    print("\n"*2+"="*80)
    model_name="VI2IR"
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name)
    test_out_folder=os.path.join('test_result',dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Encoder_1 = Encode().to(device)
    Decoder_1 = Decode().to(device)
    Attention = Common().to(device)
    ckpt_path = r"./models/1"
    Encoder_1.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder_1.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    Attention.load_state_dict(torch.load(ckpt_path)['Attention'])
    Encoder_1.eval()
    Decoder_1.eval()
    Attention.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):
            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = cv2.split(image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='YCrCb'))[0][np.newaxis, np.newaxis, ...] / 255.0
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))


            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
            feature_V, feature_I = Encoder_1(data_VIS, data_IR)
            feature_V_SAM, feature_I_SAM = Attention(feature_V, feature_I)

            data_VIS_I,_ = Decoder_1(feature_V_SAM, feature_I_SAM, stage=False)
            fi = np.squeeze((data_VIS_I * 255).cpu().numpy())
            # float32 to uint8
            fi = fi.astype(np.uint8)
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)
