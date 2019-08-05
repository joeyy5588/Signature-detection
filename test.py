import logging
from model import GGGAN
import torch
import argparse
import utils
import os
from torchvision.utils import save_image
from utils import ensure_dir

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default='saved/checkpoint_100.pth', help='the path of checkpoint to be loaded')
parser.add_argument('-s', '--save_dir', type=str, default='saved/img/', help='the path to save result')
opt = parser.parse_args()
ensure_dir(opt.save_dir) 

if __name__ == '__main__':
    model = GGGAN
    test_dir = "test/original/"
    
    G = model.Generator()
    g_state_dict = torch.load(opt.checkpoint)['gen_state_dict']
    G.load_state_dict(g_state_dict)

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    
    G.to(device), G.eval()

    with torch.no_grad():
        testing_list = os.listdir(test_dir)

        for i in range(10):
            path = test_dir + testing_list[i]

            with open(path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('L')
                img = np.array(img).astype(np.float32)
                img = 255.0 - img
                img = (img - 127.5) / 127.5
                img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                img = img.to(self.device)
            img = G(img)[0].squeeze().cpu().numpy()
            img = (img * 127.5 + 127.5)
            img = img.astype(int)
            img = 255 - img
            new_path = 'saved/img/_rm_' + testing_list[i]
            plt.imsave(new_path, img, cmap='gray')
