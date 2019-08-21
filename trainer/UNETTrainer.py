import numpy as np
import torch
import torch.nn.functional as F
import logging
import os
import random
import torch.nn as nn
import math
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image

class UNETTrainer:    
    def __init__(self, gen, dataloader, opt):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt = opt
        self.n_epochs = opt.n_epochs
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.device = self._prepare_gpu()
        self.gen = gen
        self.gen_iter = 1
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr = 1e-3, betas=(0.5, 0.999))
        self.reconstruction_loss = nn.L1Loss()
        self.real_label = 1
        self.fake_label = 0
        self.test_dir = 'test/original/'
        self.begin_epoch = 0
        self.all_log = []
        self._resume_checkpoint(opt.checkpoint)

    def train(self):
        opt = self.opt
        all_log = self.all_log
        self.gen.to(self.device)
        self.gen = nn.DataParallel(self.gen)
        self.logger.info('[GEN_STRUCTURE]')
        self.logger.info(self.gen)

        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            log = self._train_epoch(i)
            merged_log = {**log}
            all_log.append(merged_log)
            if (i + 1)%5 == 0:
                checkpoint = {
                    'log': all_log,
                    'gen_state_dict': self.gen.module.state_dict(),
                    'gen_optimizer': self.gen_optimizer.state_dict(),
                }

                check_path = os.path.join(opt.save_dir, 'checkpoint_UNET_' + str(i+1) + '.pth')
                torch.save(checkpoint, check_path)
                print("SAVING CHECKPOINT:", check_path)

    def _train_epoch(self, epoch):
        self.gen.train()
        G_sum_loss = 0

        for batch_idx, (origin_img, hw_img, pt_img)  in enumerate(self.dataloader):

            #Train Discriminator
            #for p in self.dis.parameters():
            #    p.data.clamp_(-0.01, 0.01)

            origin_img = origin_img.to(self.device)
            hw_img = hw_img.to(self.device)
            pt_img = pt_img.to(self.device)

            #Train Generator
            loss_g = self._train_gen(origin_img, hw_img, pt_img)

            G_sum_loss += loss_g.item()
            print('[%d/%d] Loss_G: %.4f' % (batch_idx + 1, len(self.dataloader), loss_g.item()))
        
        log = {
            'epoch': epoch,
            'Gen_loss': G_sum_loss,
        }
        print("======================================================================================")
        print('FINISH EPOCH: [%d/%d] Loss_G: %.4f'% (epoch + 1, self.n_epochs, G_sum_loss))
        print("======================================================================================")
        if (epoch + 1)%5 == 0:
            self._inference_testing()

        return log

    def _train_gen(self, origin_img, hw_img, pt_img):
        self.gen_optimizer.zero_grad()

        fake_hw = self.gen(origin_img)

        hw_loss = self._RECONSTRUCT_loss(fake_hw, hw_img, pt_img)

        loss_g = hw_loss

        loss_g.backward()
        self.gen_optimizer.step()

        return loss_g

    def _RECONSTRUCT_loss(self, gen_img, gt_img, pt_img, loss_type="L1"):
        if loss_type != "L1":
            self.reconstruction_loss = nn.MSELoss()
        thres = gt_img > -0.5
        pt_thresh = ((pt_img > -0.5) & (gt_img < -0.5))
        bg_loss = self.reconstruction_loss(gen_img[~thres], gt_img[~thres])
        pt_loss = self.reconstruction_loss(gen_img[pt_thresh], gt_img[pt_thresh])
        fg_loss = self.reconstruction_loss(gen_img[thres], gt_img[thres])


        return (bg_loss + 4 * pt_loss + 10 * fg_loss)

    def _inference_testing(self):
        with torch.no_grad():
            testing_list = os.listdir(self.test_dir)

            for i in range(10):
                path = self.test_dir + testing_list[i]

                with open(path, 'rb') as f:
                    img = Image.open(f)
                    img = img.convert('L')
                    img = np.array(img).astype(np.float32)
                    img = 255.0 - img
                    img = (img - 127.5) / 127.5
                    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
                    img = img.to(self.device)
                img = self.gen(img)[0].squeeze().cpu().numpy()
                img = (img * 127.5 + 127.5)
                img = img.astype(int)
                img = 255 - img
                new_path = 'saved/img/_rm_' + testing_list[i]
                plt.imsave(new_path, img, cmap='gray')


    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

    def _resume_checkpoint(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            self.gen.load_state_dict(checkpoint['gen_state_dict'])
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            self.begin_epoch = checkpoint['log'][-1]['epoch'] + 1
            self.all_log = checkpoint['log']

            for state in self.gen_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        except:
            self.logger.error('[Resume] Cannot load from checkpoint')
