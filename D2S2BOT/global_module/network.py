import torch
from torch import nn


import sys
sys.path.append('../global_module/')
from global_module.activation import mish, gelu, gelu_new
from global_module.TransformerBlock import BottleStack



up_kwargs = {'mode': 'bilinear', 'align_corners': True}

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class D2S2BoT(nn.Module):
    def __init__(self, band, classes,layers,patch):
        super(D2S2BoT, self).__init__()
        self.band = band
        self.classes = classes
        # spectral branch
        self.name = 'D2S2BOT'
        self.D2S2BoT = BottleStack(dim=120, fmap_size=11, dim_out=120, proj_factor=2, downsample=False, heads=2, dim_head=30, num_layers=layers, rel_pos_emb=False, activation=nn.ReLU())

        self.global_pooling = nn.AdaptiveAvgPool3d(1)


        self.conv2_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=24, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish(),
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=24, padding=(1, 1, 2),
                      kernel_size=(3, 3, 5), stride=(1, 1, 1)),
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=48, out_channels=24, padding=(0, 0, 3),
                      kernel_size=(1, 1, 7), stride=(1, 1, 2)),
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish(),
        )

        self.conv4_1 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=36, padding=(1, 1, 2),
                      kernel_size=(3, 3, 5), stride=(1, 1, 1)),
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish(),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=36, padding=(1, 1, 1),
                      kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=72, out_channels=60, padding=(0, 0, 3),
                      kernel_size=(1, 1, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish(),
        )


        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=90, out_channels=60, padding=(0, 0, 3),
                      kernel_size=(1, 1, 7), stride=(1, 1, 1)),
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv3d(in_channels=60, out_channels=60, padding=(0, 0, 0),
                      kernel_size=(1, 1, int(self.band/2)), stride=(1, 1, 1)),
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish(),
        )
        self.conv3D_res = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=30, padding=(0, 0, 3),
                      kernel_size=(1, 1, 7), stride=(1, 1, 2)),
            nn.BatchNorm3d(30, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish(),
        )

        self.spaconv1 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=24, padding=(0, 0),
                      kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )
        self.spaconv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.band, out_channels=24, padding=(2, 2),
                      kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )
        self.spaconv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.band, out_channels=24, padding=(1, 1),
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )
        self.spaconv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=36, padding=(2, 2),
                      kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(36, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )
        self.spaconv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=36, padding=(1, 1),
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(36, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )
        self.spaconv2 = nn.Sequential(
            nn.Conv2d(in_channels=72, out_channels=60, padding=(0, 0),
                      kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )
        self.spaconv3 = nn.Sequential(
            nn.Conv2d(in_channels=90, out_channels=60, padding=(0, 0),
                      kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )

        self.spaconv4 = nn.Sequential(
            nn.Conv2d(in_channels=120, out_channels=60, padding=(0, 0),
                      kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )
        self.spaconv_res = nn.Sequential(
            nn.Conv2d(in_channels=self.band, out_channels=30, padding=(3, 3),
                      kernel_size=(7, 7), stride=(1, 1)),
            nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            mish()
        )

        self.fc1 = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(120, classes)
            # nn.Softmax()
        )



    def forward(self, X):
        # Spectral Resblock

        x2 = torch.cat((self.conv2_1(X), self.conv2_2(X)), dim=1)
        x3 = self.conv3(x2)
        x4 = torch.cat((self.conv4_1(x3), self.conv4_2(x3)), dim=1)
        x5 = self.conv5(x4)
        x6 =torch.cat((x5, self.conv3D_res(X)), dim=1)
        x7 =self.conv6(x6)
        x8 =self.conv7(x7)

        # Spatial Resblock
        x_spa = (torch.squeeze(X, dim=1)).permute(0, 3, 1, 2)
        x_spa1 = torch.cat((self.spaconv1_1(x_spa), self.spaconv1_2(x_spa)), dim=1)
        x_spa2 = self.spaconv1(x_spa1)
        x_spa2 = torch.cat((self.spaconv2_1(x_spa2), self.spaconv2_2(x_spa2)), dim=1)
        x_spa3 =self.spaconv2(x_spa2)
        x_spares =self.spaconv_res(x_spa)
        x_spa4 = self.spaconv3(torch.cat((x_spa3, x_spares), dim=1))
        x_spa5 = (torch.unsqueeze(x_spa4, dim=4))


        # Fusion
        x_in1 = torch.cat((x8, x_spa5), dim=1)
        # Bottleneck Transformer
        x_fal = self.D2S2BoT(x_in1)

        # Liner Projection Classifier
        x_fal1 = (self.global_pooling(x_fal)).squeeze(-1).squeeze(-1).squeeze(-1)
        output = self.fc1(x_fal1)

        return output
