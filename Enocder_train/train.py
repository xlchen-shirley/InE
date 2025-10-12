import argparse
import os
from h5 import H5Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from model import InE_enocder
import torch
import torch.nn.functional as F
from model_TII import BiSeNet

def reconstruction_loss(decoded_ir, decoded_vis, original_ir, original_vis):
    return F.mse_loss(decoded_ir, original_ir) + F.mse_loss(decoded_vis, original_vis)

if __name__ == '__main__':
    image = 'vi'
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_path', default=f'./model/{image}')  # 模型存储路径
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    trainloader = DataLoader(H5Dataset(r"./data/MSRS_train_128_stride_200.h5"), batch_size=1, shuffle=True,
                             num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_res_vi = r'./model/model_final.pth'
    net = InE_enocder(1).to(device)
    seg_net = BiSeNet(9).cuda().eval()
    seg_net.load_state_dict(torch.load(path_res_vi))
    loss_total = 0
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(200):

        train_tqdm = tqdm(trainloader, total=len(trainloader))
        net.train()
        iter_num = 0
        for data_IR, data_VIS in train_tqdm:
            iter_num += 1
            optimizer.zero_grad()
            ir_image = data_IR.cuda()
            vis_image = data_VIS.cuda()
            vis_3 = torch.cat((vis_image, vis_image, vis_image), dim=1)
            seg_ir, _ = seg_net(vis_3)
            out_p, feature_1 = net(vis_image, seg_ir)
            out_i, feature_2 = net(ir_image, seg_ir)
            recon_loss = reconstruction_loss(out_i, out_p, ir_image, vis_image)
            loss = recon_loss
            loss_total += loss.item()
            train_tqdm.set_postfix(epoch=epoch, loss_total=loss.item(), recon_loss=recon_loss.item())
            loss.backward()
            optimizer.step()
        print({loss_total/iter_num})
        loss_total = 0
        torch.save(net.state_dict(), f'{args.save_path}/InE_encoder_{image}_epoch_{epoch}.pth')
