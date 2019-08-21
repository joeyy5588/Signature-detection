import torch.nn as nn
from .snconv import SNConv2d
from .snlinear import SNLinear
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, use_BN = False, downsample=False):
        super(ResBlock, self).__init__()
        #self.conv1 = SNConv2d(n_dim, n_out, kernel_size=3, stride=2)
        hidden_channels = in_channels
        self.downsample = downsample

        self.resblock = self.make_res_block(in_channels, out_channels, hidden_channels, use_BN, downsample)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)
    def make_res_block(self, in_channels, out_channels, hidden_channels, use_BN, downsample):
        model = []
        if use_BN:
            model += [nn.BatchNorm2d(in_channels)]

        model += [nn.ReLU()]
        model += [SNConv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(hidden_channels, out_channels, kernel_size=3, padding=1)]
        if downsample:
            model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        if self.downsample:
            model += [nn.AvgPool2d(2)]
            return nn.Sequential(*model)
        else:
            return nn.Sequential(*model)

    def forward(self, input):
        return self.resblock(input) + self.residual_connect(input)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OptimizedBlock, self).__init__()
        self.res_block = self.make_res_block(in_channels, out_channels)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)
    def make_res_block(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(out_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def forward(self, input):
        return self.res_block(input) + self.residual_connect(input)

class SNResDiscriminator(nn.Module):
    def __init__(self, ndf=64, ndlayers=4):
        super(SNResDiscriminator, self).__init__()
        self.res_d = self.make_model(ndf, ndlayers)
        self.fc = nn.Sequential(SNLinear(ndf*16, 1))
    def make_model(self, ndf, ndlayers):
        model = []
        model += [OptimizedBlock(2, ndf)]
        tndf = ndf
        for i in range(ndlayers):
            model += [ResBlock(tndf, tndf*2, downsample=True)]
            tndf *= 2
        model += [nn.ReLU()]
        return nn.Sequential(*model)
    def forward(self, input):
        out = self.res_d(input)
        out = F.avg_pool2d(out, out.size(3), stride=1)
        out = out.view(-1, 1024)
        return self.fc(out)

class SNDiscriminator(nn.Module):
    def __init__(self, nc=2, ndf=64):
        super(SNDiscriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            #SNConv2d()
            SNConv2d(nc, ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1 x 32
            SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf*2, ndf * 2, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SNConv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
        #self.snlinear = nn.Sequential(SNLinear(ndf * 4 * 4 * 4, 1),
        #                              nn.Sigmoid())

    def forward(self, input):
        output = self.main(input)
        #output = output.view(output.size(0), -1)
        #output = self.snlinear(output)
        return output
