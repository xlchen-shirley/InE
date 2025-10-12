import os

from PIL import Image
from torch.utils import data
from torchvision import transforms

from common import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])


class data_load(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'vi':
                self.vis_path = temp_path  # 获得红外路径
            elif sub_dir == 'ir':
                self.if_path = temp_path

        self.name_list = os.listdir(self.vis_path)  # 获得子目录下的图片的名称
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称
        vis_image = Image.open(os.path.join(self.vis_path, name))
        if_image = Image.open(os.path.join(self.if_path, name)).convert('L')

        vis_image = self.transform(vis_image)
        if_image = self.transform(if_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_y_image,if_image, name

    def __len__(self):
        return len(self.name_list)
