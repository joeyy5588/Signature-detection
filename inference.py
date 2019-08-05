import logging
from model import GGGAN
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import argparse
import utils
import os
from torchvision.utils import save_image
from utils import ensure_dir
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default='saved/checkpoint_5.pth', help='the path of checkpoint to be loaded')
parser.add_argument('-s', '--save_dir', type=str, default='saved/img/', help='the path to save result')
parser.add_argument('-i', '--img', type=str, default='saved/img/', help='the path inference image')
opt = parser.parse_args()
ensure_dir(opt.save_dir) 

if __name__ == '__main__':
    
    G = GGGAN.Generator()

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    
    g_state_dict = torch.load(opt.checkpoint)['gen_state_dict']
    G.load_state_dict(g_state_dict)

    G.to(device), G.eval()
    G = nn.DataParallel(G)

    with torch.no_grad():
        path = opt.img

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            img = np.array(img).astype(np.float32)
            img = 255.0 - img
            img = (img - 127.5) / 127.5
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            img = img.to(device)
        img = G(img)[0].squeeze().cpu().numpy()
        img = (img * 127.5 + 127.5)
        img = img.astype(int)
        img = 255 - img
        new_path = 'inference_' + path
        plt.imsave(new_path, img, cmap='gray')
