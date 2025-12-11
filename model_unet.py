# Original unet model implementation
import torch, torch.nn as nn, torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        # in_ch is channels after concatenation (skip + upsampled)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # if using transposed conv, change here accordingly
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, x_skip):
        x = self.up(x)
        dy = x_skip.size(2) - x.size(2)
        dx = x_skip.size(3) - x.size(3)
        x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        x = torch.cat([x_skip, x], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch): 
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # encoder
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)    # <- 1024 bottleneck

        # decoder (note in_ch = cat(skip, upsampled))
        self.up5   = Up(1024 + 512, 512)
        self.up4   = Up(512  + 256, 256)
        self.up3   = Up(256  + 128, 128)
        self.up2   = Up(128  + 64,   64)
        self.up1   = DoubleConv(64, 64)
        self.outc  = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)             # 1024 channels
        x  = self.up5(x5, x4)
        x  = self.up4(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up2(x,  x1)
        x  = self.up1(x)
        return self.outc(x)              # logits

