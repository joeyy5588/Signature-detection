from loader import DataLoader
from model import GGGAN, GGGGAN, SNDIS, ResDense, AttnUNET
from trainer import GGGANTrainer, SINGLEGANTrainer, UNETTrainer
from utils import ensure_dir
import logging
import argparse

handlers = [logging.FileHandler('output.log', mode = 'w'), logging.StreamHandler()]
logging.basicConfig(handlers = handlers, level=logging.INFO, format='')
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='saved/')
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--decoder', type=int, help='num of decoder', default=1)
parser.add_argument('--mode', type=str, default='GAN')


opt = parser.parse_args()
ensure_dir(opt.save_dir)

if __name__ == '__main__':
    # Remember to set pretrain False when there is no pretrained lang model
    D = DataLoader(data_dir = 'data/', batch_size = opt.batch, shuffle = True, validation_split = 0.2)
    if opt.mode == "CNN":
        GEN = GGGAN.Generator()
        T = UNETTrainer(gen = GEN, dataloader = D, opt = opt)
        T.train()

    elif opt.mode == "GAN":
        if opt.decoder == 1:
            GEN = AttnUNET.Generator()
            DIS = AttnUNET.Discriminatorv2()
            T = SINGLEGANTrainer(gen = GEN, dis = DIS, dataloader = D, opt = opt)
            T.train()
        else:
            GEN = GGGGAN.Generator()
            DIS = GGGGAN.Discriminator()
            T = GGGANTrainer(gen = GEN, dis = DIS, dataloader = D, opt = opt)
            T.train()