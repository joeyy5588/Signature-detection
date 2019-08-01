import numpy as np
import torch
import torch.nn.functional as F
import logging
import os
import random
import torch.nn as nn
import math
from torchvision.utils import save_image


class DCGANTrainer:    
    def __init__(self, gen, dis, dataloader, opt):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.opt = opt
        self.n_epochs = opt.n_epochs
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.device = self._prepare_gpu()
        self.gen = gen
        self.dis = dis
        self.gen_iter = 1
        self.dis_iter = 1
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr = 0.0002, betas=(0.5, 0.999))
        self.dis_optimizer = torch.optim.Adam(self.dis.parameters(), lr = 0.0002, betas=(0.5, 0.999))
        self.reconstruction_loss = nn.L1Loss()
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.real_label = 1
        self.fake_label = 0
        self.begin_epoch = 0
        self.all_log = []
        self._resume_checkpoint(opt.checkpoint)

    def train(self):
        opt = self.opt
        all_log = self.all_log
        self.gen.to(self.device)
        self.dis.to(self.device)
        self.logger.info('[GEN_STRUCTURE]')
        self.logger.info(self.gen)
        self.logger.info('[DIS_STRUCTURE]')
        self.logger.info(self.dis)

        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            log = self._train_epoch(i)
            merged_log = {**log}
            all_log.append(merged_log)
            checkpoint = {
                'log': all_log,
                'gen_state_dict': self.gen.state_dict(),
                'dis_state_dict': self.dis.state_dict(),
            }
            if (i + 1)%5 == 0:
                check_path = os.path.join(opt.save_dir, 'checkpoint_' + str(i+1) + '.pth')
                torch.save(checkpoint, check_path)

    def _train_epoch(self, epoch):
        self.gen.train()
        self.dis.train()
        G_sum_loss = 0
        D_sum_loss = 0

        for batch_idx, origin_img, hw_img, pt_img  in enumerate(self.dataloader):

            #Train Discriminator
            origin_img = origin_img.to(self.device)
            hw_img = hw_img.to(self.device)
            pt_img = pt_img.to(self.device)

            loss_d, D_x, D_G_z1 = self._train_dis(origin_img, hw_img, pt_img)

            #Train Generator
            loss_g, D_G_z2 = self._train_gen(origin_img, hw_img, pt_img)

            G_sum_loss += loss_g.item()
            D_sum_loss += loss_d.item()
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'%\
             (batch_idx + 1, len(self.dataloader), loss_d.item(), loss_g.item(), D_x, D_G_z1, D_G_z2))
        
        log = {
            'epoch': epoch,
            'Gen_loss': G_sum_loss,
            'Dis_loss': D_sum_loss
        }
        print("======================================================================================")
        print('FINISH EPOCH: [%d/%d] Loss_D: %.4f Loss_G: %.4f'% (epoch + 1, self.n_epochs, D_sum_loss, G_sum_loss))
        print("======================================================================================")
        if (epoch + 1)%5 == 0:
            with torch.no_grad():
                fixed_image = self.gen(self.fixed_noise)
                save_image(fixed_image.data[:25], "saved/images_%d.png" % (epoch + 1), nrow = 5, normalize = True)

        return log

    def _train_dis(self, origin_img, hw_img, pt_img):
        self.dis_optimizer.zero_grad()

        fake_hw, fake_pt = self.gen(origin_img)
        fake_hw = fake_hw.detach()
        fake_pt = fake_pt.detach()
        fake_origin = fake_hw + fake_pt

        fake_img = torch.cat((fake_hw, fake_pt, fake_origin))
        real_img = torch.cat((hw_img, pt_img, origin_img))
        print(fake_hw.size(), fake_img.size())

        real_predict = self.dis(real_img)
        fake_predict = self.dis(fake_img)

        real_loss = self._GAN_loss(real_predict, True)
        fake_loss = self._GAN_loss(fake_predict, False)
        D_x = real_predict.mean().item()
        D_G_z1 = fake_predict.mean().item()

        loss_d = (real_loss + fake_loss) / 2
        loss_d.backward()
        self.dis_optimizer.step()

        return loss_d, D_x, D_G_z1

    def _train_gen(self, origin_img, hw_img, pt_img):
        self.gen_optimizer.zero_grad()

        fake_hw, fake_pt = self.gen(origin_img)
        fake_hw = fake_hw
        fake_pt = fake_pt
        fake_origin = fake_hw + fake_pt

        fake_img = torch.cat((fake_hw, fake_pt, fake_origin))

        fake_predict = self.dis(fake_img)

        gen_loss = self._GAN_loss(fake_predict, True)
        hw_loss = self._RECONSTRUCT_loss(fake_hw, hw_img)
        pt_loss = self._RECONSTRUCT_loss(fake_pt, pt_img)
        print(gen_loss, hw_loss, pt_loss)

        loss_g = gen_loss + hw_loss + pt_loss

        D_G_z2 = fake_predict.mean().item()
        loss_g.backward()
        self.gen_optimizer.step()

        return loss_g, D_G_z2

    def _GAN_loss(self, pred, real, loss_type="BCE"):
        if loss_type != "BCE":
            self.gan_loss = self.loss = nn.MSELoss()
        if real:
            target = torch.ones(pred.size()).to(self.device)
        else:
            target = torch.zeros(pred.size()).to(self.device)

        return self.gan_loss(pred, target)

    def _RECONSTRUCT_loss(self, gen_img, gt_img, loss_type="L1"):
        if loss_type != "L1":
            self.reconstruction_loss = nn.MSELoss()

        return self.reconstruction_loss(gen_img, gt_img)

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device

    def _resume_checkpoint(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.begin_epoch = checkpoint['log'][-1]['epoch'] + 1
            self.all_log = checkpoint['log']
        except:
            self.logger.error('[Resume] Cannot load from checkpoint')