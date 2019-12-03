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

class UNETATTNTrainer:    
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
        self.test_dir = 'data/mixed/'
        self.begin_epoch = 0
        self.all_log = []
        self._resume_checkpoint(opt.checkpoint)

    def train(self):
        opt = self.opt
        all_log = self.all_log
        print('flag1')
        self.gen = nn.DataParallel(self.gen)
        self.dis = nn.DataParallel(self.dis)
        print('flag2')
        self.gen.to(self.device)
        self.dis.to(self.device)
        print('flag3')
        # self.logger.info('[GEN_STRUCTURE]')
        # self.logger.info(self.gen)
        # self.logger.info('[DIS_STRUCTURE]')
        # self.logger.info(self.dis)

        for i in range(self.begin_epoch, self.begin_epoch + self.n_epochs):
            log = self._train_epoch(i)
            merged_log = {**log}
            all_log.append(merged_log)
            if (i + 1)%5 == 0:
                checkpoint = {
                    'log': all_log,
                    'gen_state_dict': self.gen.module.state_dict(),
                    'dis_state_dict': self.dis.module.state_dict(),
                    'gen_optimizer': self.gen_optimizer.state_dict(),
                    'dis_optimizer': self.dis_optimizer.state_dict()
                }

                check_path = os.path.join(opt.save_dir, 'checkpoint_GAN_' + str(i+1) + '.pth')
                torch.save(checkpoint, check_path)
                print("SAVING CHECKPOINT:", check_path)

    def _train_epoch(self, epoch):
        self.gen.train()
        self.dis.train()
        G_sum_loss = 0
        D_sum_loss = 0

        for batch_idx, (origin_img, hw_img, pt_img)  in enumerate(self.dataloader):

            #Train Discriminator
            #for p in self.dis.parameters():
            #    p.data.clamp_(-0.01, 0.01)

            origin_img = origin_img.to(self.device)
            hw_img = hw_img.to(self.device)
            pt_img = pt_img.to(self.device)

            #Train Discriminator
            loss_d, D_x, D_G_z1 = self._train_dis(origin_img, hw_img, pt_img)

            #Train Generator
            loss_g, D_G_z2 = self._train_gen(origin_img, hw_img, pt_img)

            G_sum_loss += loss_g.item()
            D_sum_loss += loss_d.item()
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'%\
             (batch_idx + 1, len(self.dataloader), loss_d.item(), loss_g.item(), D_x, D_G_z1, D_G_z2))

        valid_loss = self.validation()
        
        log = {
            'epoch': epoch,
            'Gen_loss': G_sum_loss,
            'Dis_loss': D_sum_loss,
            'Val_loss': valid_loss
        }
        print("======================================================================================")
        print('FINISH EPOCH: [%d/%d] Loss_D: %.4f Loss_G: %.4f'% (epoch + 1, self.n_epochs, D_sum_loss, G_sum_loss))
        print("======================================================================================")
        if (epoch + 1)%5 == 0:
            self._inference_testing()

        return log

    def _train_dis(self, origin_img, hw_img, pt_img):
        self.dis_optimizer.zero_grad()

        fake_hw = self.gen(origin_img, hw_img)
        fake_hw = fake_hw.detach()

        fake_pt = self.gen(pt_img, hw_img)
        fake_pt = fake_pt.detach()
        white_img = torch.full(origin_img.size(), -1).to(self.device)

        fake_hw_2 = self.gen(hw_img, hw_img)
        fake_hw_2 = fake_hw_2.detach()

        real_predict = self.dis(hw_img, origin_img, hw_img)
        fake_predict = self.dis(fake_hw, origin_img, hw_img)
        real_predict_2 = self.dis(white_img, origin_img, hw_img)
        fake_predict_2 = self.dis(fake_pt, origin_img, hw_img)
        real_predict_3 = self.dis(hw_img, origin_img, hw_img)
        fake_predict_3 = self.dis(fake_hw_2, origin_img, hw_img)

        real_loss = self._GAN_loss(real_predict, True)
        fake_loss = self._GAN_loss(fake_predict, False)
        real_loss_2 = self._GAN_loss(real_predict_2, True)
        fake_loss_2 = self._GAN_loss(fake_predict_2, False)
        real_loss_3 = self._GAN_loss(real_predict_3, True)
        fake_loss_3 = self._GAN_loss(fake_predict_3, False)
        D_x = real_predict.mean().item()
        D_G_z1 = fake_predict.mean().item()

        loss_d = (real_loss + fake_loss + real_loss_2 + fake_loss_2 + real_loss_3 + fake_loss_3) / 6
        loss_d.backward()
        self.dis_optimizer.step()

        return loss_d, D_x, D_G_z1

    def _train_gen(self, origin_img, hw_img, pt_img):
        self.gen_optimizer.zero_grad()

        fake_hw = self.gen(origin_img, hw_img)
        fake_pt = self.gen(pt_img, hw_img)
        fake_hw_2 = self.gen(hw_img, hw_img)
        white_img = torch.full(origin_img.size(), -1).to(self.device)


        fake_img = torch.cat((origin_img, fake_hw), dim=1)
        fake_img_2 = torch.cat((pt_img, fake_pt), dim=1)
        fake_img_3 = torch.cat((hw_img, fake_hw_2), dim=1)
        fake_predict = self.dis(fake_hw, origin_img, hw_img)
        fake_predict_2 = self.dis(fake_pt, origin_img, hw_img)
        fake_predict_3 = self.dis(fake_hw_2, origin_img, hw_img)

        gen_loss = self._GAN_loss(fake_predict, True)
        # hw_loss = self._RECONSTRUCT_loss(fake_hw, hw_img, pt_img)
        gen_loss_2 = self._GAN_loss(fake_predict_2, True)
        hw_loss_2 = nn.L1Loss()(fake_pt, white_img)
        gen_loss_3 = self._GAN_loss(fake_predict_3, True)
        hw_loss_3 = self._RECONSTRUCT_loss(fake_hw_2, hw_img, pt_img)

        loss_g = (gen_loss + gen_loss_2 + hw_loss_2 + gen_loss_3 + hw_loss_3) / 5

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

    def _RECONSTRUCT_loss(self, gen_img, gt_img, pt_img, loss_type="L1"):
        if loss_type != "L1":
            self.reconstruction_loss = nn.MSELoss()
        thres = gt_img > -0.5
        pt_thresh = pt_img > -0.5
        bg_loss = self.reconstruction_loss(gen_img[pt_thresh], gt_img[pt_thresh])
        fg_loss = self.reconstruction_loss(gen_img[thres], gt_img[thres])


        return (bg_loss + fg_loss * 10) 

    def _inference_testing(self):
        with torch.no_grad():
            for batch_idx, (origin_img, hw_img, pt_img)  in enumerate(self.dataloader.split_validation()):
                if batch_idx < 10:
                    origin_img = origin_img.to(self.device)
                    hw_img = hw_img.to(self.device)
                    pt_img = pt_img.to(self.device)
                    fake_hw = self.gen(origin_img, hw_img)
                    print(fake_hw.shape)
                    img = fake_hw[0].squeeze().cpu().numpy()
                    img = (img * 127.5 + 127.5)
                    img = img.astype(int)
                    img = 255 - img
                    new_path = 'saved/img/_rm_' + str(batch_idx)
                    plt.imsave(new_path, img, cmap='gray')

    def validation(self):
        with torch.no_grad():
            total_hw_loss = 0
            for batch_idx, (origin_img, hw_img, pt_img)  in enumerate(self.dataloader.split_validation()):
                origin_img = origin_img.to(self.device)
                hw_img = hw_img.to(self.device)
                pt_img = pt_img.to(self.device)
                fake_hw = self.gen(origin_img, hw_img)
                hw_loss = self._RECONSTRUCT_loss(fake_hw, hw_img, pt_img)
                total_hw_loss += hw_loss.item()
            print("VALID_LOSS:", total_hw_loss)
            return total_hw_loss

    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda' if n_gpu > 0 else 'cpu')
        return device

    def _resume_checkpoint(self, path):
        if path == None: return
        try:
            checkpoint = torch.load(path)
            self.gen.load_state_dict(checkpoint['gen_state_dict'])
            self.dis.load_state_dict(checkpoint['dis_state_dict'])
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
            self.begin_epoch = checkpoint['log'][-1]['epoch'] + 1
            self.all_log = checkpoint['log']

            for state in self.gen_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            for state in self.dis_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        except:
            self.logger.error('[Resume] Cannot load from checkpoint')

    def _resume_unet_checkpoint(self, path):
        if path == None: return
        if True:
            checkpoint = torch.load(path)
            self.gen.load_state_dict(checkpoint['gen_state_dict'])
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])

            print('here')

            for state in self.gen_optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        else:
            self.logger.error('[Resume] Cannot load from checkpoint')

