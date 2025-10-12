import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from commend import RGB2YCrCb
from model import InE_enocder
from model_TII import BiSeNet
path_res_ir = r'./models/ir/encoder_ir.pth'
path_res_vi = r'./models/vi/encoder_vi.pth'
path_seg = r'../../Enocder_train/model/model_final.pth'

@torch.no_grad()
def loss(vis_y_image, f_y_image, vi2ir, ir_image, ir2vi):


    diff_loss = 0
    model_seg = BiSeNet(9).eval().cuda()
    net_res_ir = InE_enocder(1).eval().cuda()
    net_res_vi = InE_enocder(1).eval().cuda()

    feature_fuse_ir = []
    feature_ir = []
    feature_vi = []
    feature_fuse_vi = []


    net_res_ir.load_state_dict(torch.load(path_res_ir))
    net_res_vi.load_state_dict(torch.load(path_res_vi))
    model_seg.load_state_dict(torch.load(path_seg))
    ir_3 = torch.cat((ir_image, ir_image, ir_image), dim=1)
    vi_3 = torch.cat((vis_y_image, vis_y_image, vis_y_image), dim=1)
    # =====================================ir============================
    image_seg, _ = model_seg(ir_3)
    _, feature_img = net_res_ir(ir_image, image_seg)
    _, feature_pseudo = net_res_ir(vi2ir, image_seg)
    _, feature_fuse = net_res_ir(f_y_image, image_seg)
    diff_ir = torch.mean(torch.abs(feature_img - feature_pseudo), dim=(2, 3)).detach().cpu().numpy()
    diff_ir = np.mean(diff_ir, axis=0)

    indices = diff_ir.argsort()[::-1][:10]
    for i in indices:
        feature_img_selected = feature_img[:, i, :, :]
        feature_fuse_selected = feature_fuse[:, i, :, :]
        feature_fuse_ir.append(feature_fuse_selected)
        feature_ir.append(feature_img_selected)
    #=====================================vi============================
    image_seg, _ = model_seg(vi_3)
    _, feature_img = net_res_vi(vis_y_image, image_seg)
    _, feature_pseudo = net_res_vi(ir2vi, image_seg)
    _, feature_fuse = net_res_vi(f_y_image, image_seg)

    diff_vi = torch.mean(torch.abs(feature_img - feature_pseudo), dim=(2, 3)).detach().cpu().numpy()
    diff_vi = np.mean(diff_vi, axis=0)

    indices = diff_vi.argsort()[::-1][:10]
    for i in indices:
        feature_img_selected = feature_img[:, i, :, :]
        feature_fuse_selected = feature_fuse[:, i, :, :]
        feature_fuse_vi.append(feature_fuse_selected)
        feature_vi.append(feature_img_selected)

    for num in range(10):
        diff_f = torch.mean(torch.abs(feature_ir[num] - feature_fuse_ir[num]), dim=(0, 1, 2)) + \
                 torch.mean(torch.abs(feature_vi[num] - feature_fuse_vi[num]), dim=(0, 1, 2))
        diff_loss += (diff_f / 2)

    return diff_loss / 10


if __name__=='__main__':
    vis = Image.open(r'')
    ir = Image.open(r'').convert('L')
    f = Image.open(r'')
    vi2ir = Image.open(r'')
    ir2vi = Image.open(r'')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    vis = transform(vis)
    ir = transform(ir)
    vi2ir = transform(vi2ir)
    ir2vi = transform(ir2vi)
    f = transform(f)

    vis_y_image, _, _ = RGB2YCrCb(vis)
    f_y, _, _ = RGB2YCrCb(f)

    vis_y_image = vis_y_image.unsqueeze(0).cuda()
    ir = ir.unsqueeze(0).cuda()
    vi2ir = vi2ir.unsqueeze(0).cuda()
    f_y = f_y.unsqueeze(0).cuda()
    ir2vi = ir2vi.unsqueeze(0).cuda()

    a = loss(vis_y_image, f_y, vi2ir, ir, ir2vi)
