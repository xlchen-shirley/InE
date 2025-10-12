import os
import torch.utils.data as Data
import h5py
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

to_tensor = transforms.Compose([transforms.ToTensor()])

class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        VIS_C = np.array(h5f['vis_c_patchs'][key])
        h5f.close()
        return torch.Tensor(VIS), torch.Tensor(IR), torch.Tensor(VIS_C).squeeze(0)

# class MSRS_loader(Data.Dataset):
#     def __init__(self, msrs_path, transform=to_tensor):
#         self.path_ir = os.path.join(msrs_path, 'ir')
#         self.path_vi = os.path.join(msrs_path, 'vi')
#         self.transform = transform
#         self.name_list = os.listdir(self.path_ir)
#
#     def __len__(self):
#         return len(self.name_list)
#
#     def __getitem__(self, index):
#         name = self.name_list[index]  # 获得当前图片的名称
#         inf_image = Image.open(os.path.join(self.path_ir, name)).convert('L')  # 获取红外图像
#         vis_image = Image.open(os.path.join(self.path_vi, name))
#         inf_image = self.transform(inf_image)
#         vis_image = self.transform(vis_image)
#         return vis_image, inf_image
