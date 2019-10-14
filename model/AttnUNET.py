import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Generator, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.down5 = down(512, 512)
        self.up1_1 = up(1024, 256)
        self.up1_2 = up(512, 128)
        self.up1_3 = up(256, 64)
        self.up1_4 = up(128, 64)
        self.out1_c = outconv(64, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.down5(x5)
        x, p1 = self.up1_1(x5, x4)
        x, p2 = self.up1_2(x, x3)
        x, p3 = self.up1_3(x, x2)
        x, p4 = self.up1_4(x, x1)
        x = self.out1_c(x)
        x = torch.tanh(x)
        return x, p1, p2, p3, p4

class Discriminator(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(n_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attn1 = Self_Attn(256, 'relu')
        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attn2 = Self_Attn(512, 'relu')
        self.conv5 = nn.Conv2d(512, n_classes, kernel_size=4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            #elif isinstance(m, nn.utils.spectral_norm):
            #    m.weight.data.normal_(1.0, 0.02)
            #    if m.bias is not None:
            #        m.bias.data.zero_()

    def forward(self, img):
        out = self.model(img)
        out, p1 = self.attn1(out, out)
        out = self.conv4(out)
        out, p2 = self.attn2(out, out)
        out = self.conv5(out)

        return out.squeeze(), p1, p2

class Discriminatorv2(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Discriminatorv2, self).__init__()

        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(n_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attn1 = Self_Attn(512, 'relu')
        self.conv5 = nn.Conv2d(512, n_classes, kernel_size=4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            #elif isinstance(m, nn.utils.spectral_norm):
            #    m.weight.data.normal_(1.0, 0.02)
            #    if m.bias is not None:
            #        m.bias.data.zero_()

    def forward(self, syn_img, origin_img):
        syn_out = self.model(syn_img)
        origin_out = self.model(origin_img)
        out, p1 = self.attn1(syn_out, origin_out)
        out = self.conv5(out)
        print(out.size())

        return out.squeeze(), p1
        

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(y).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(y).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, down=False):
        super(double_conv, self).__init__()
        if not down:
            self.conv = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_CNN(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, down=False):
        super(double_conv, self).__init__()
        if not down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x



class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, True)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, use_attn=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)
        self.attn = Self_Attn(in_ch//2, 'relu')
        self.use_attn = use_attn

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        if self.use_attn:
            x2, p = self.attn(x1, x2)
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_attn:
            return x, p
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
