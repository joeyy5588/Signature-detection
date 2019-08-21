import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1, bias=False),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1, bias=False)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self):
        super(RDN, self).__init__()
        ngf = 32
        kSize = 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (6, 6, 64)
        }['C']

        # In conv
        self.inconv = nn.Sequential(
            nn.Conv2d(1, ngf, kSize, padding=(kSize-1)//2, stride=1, bias=True),
            nn.Conv2d(ngf, ngf, kSize, padding=(kSize-1)//2, stride=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Down sample net
        G0 = ngf * 2
        self.downconv = nn.Sequential(    
            nn.Conv2d(ngf, G0, kSize, padding=(kSize-1)//2, stride=2),
            nn.InstanceNorm2d(G0),
            nn.ReLU(inplace=True),
        )

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])


        # Up sample net
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(G0, ngf, 2, stride=2),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

        # Out conv
        self.outconv = nn.Sequential(
            nn.Conv2d(ngf, 1, kernel_size=3, padding=1, bias=True),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.inconv(x)
        x = self.downconv(x)
        F_1 = x

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        
        ########################################

        ###  Pending: Global feature fusion  ###

        ########################################

        x = self.upconv(x)

        x = self.outconv(x)

        return x


class RDNDiscriminator(nn.Module):
    def __init__(self, n_channels=2, n_classes=1):
        super(RDNDiscriminator, self).__init__()

        use_ins_norm = True

        self.model = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_ins_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_ins_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=use_ins_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=use_ins_norm)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, n_classes, kernel_size=3, stride=1, padding=1),
        )

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
        validity = self.model(img)

        return validity
