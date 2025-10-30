import torch
import torch.nn as nn
import torch.nn.functional as F

from . import blocks

class UNet(nn.Module):
    CHANNELS = [16, 32, 64, 128]
    TR_CHANNELS = [256, 128, 64, 32]

    def __init__(self, in_channels, out_channels, kernel_size=3, conv1_kernel_size=5):
        super().__init__()

        ch = self.CHANNELS
        tr = self.TR_CHANNELS

        # Input Convolution
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv3d(in_channels, ch[0], kernel_size=conv1_kernel_size,
                               padding=conv1_kernel_size // 2, bias=False)
        # Encoder path 16 -> 32 -> 64 -> 128
        self.enc_block1 = blocks.PreActResBlock(ch[0], kernel_size, 1)
        self.down1 = blocks.down_block(ch[0], ch[1], kernel_size)

        self.enc_block2 = blocks.PreActResBlock(ch[1], kernel_size, 2)
        self.down2 = blocks.down_block(ch[1], ch[2], kernel_size)

        self.enc_block3 = blocks.PreActResBlock(ch[2], kernel_size, 3)
        self.down3 = blocks.down_block(ch[2], ch[3], kernel_size)

        # Middle convolution 128
        self.mid_block = blocks.PreActResBlock(ch[3], kernel_size, 3)

        # Decoder path 256 -> 128 -> 64 -> 32
        self.up3 = blocks.up_block(tr[0], tr[1] // 2, kernel_size)
        self.dec_block3 = blocks.PreActResBlock(tr[1], kernel_size, 3)

        self.up2 = blocks.up_block(tr[1], tr[2] // 2, kernel_size)
        self.dec_block2 = blocks.PreActResBlock(tr[2], kernel_size, 2)

        self.up1 = blocks.up_block(tr[2], tr[3] // 2, kernel_size)
        self.dec_block1 = blocks.PreActResBlock(tr[3], kernel_size, 1)

        # Combine outputs from top 3 layers
        self.out3_conv = blocks.norm_nonl_conv(tr[1], out_channels, kernel_size=1, padding=0)
        self.out3_upsample = nn.Upsample(scale_factor=2 * 2, mode='trilinear', align_corners=True)
        self.out2_conv = blocks.norm_nonl_conv(tr[2], out_channels, kernel_size=1, padding=0)
        self.out2_upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.out1_conv = blocks.norm_nonl_conv(tr[3], out_channels, kernel_size=1, padding=0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.InstanceNorm3d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        enc0 = self.conv1(input)

        skip1 = self.enc_block1(enc0)
        enc1 = self.down1(skip1)

        skip2 = self.enc_block2(enc1)
        enc2 = self.down2(skip2)

        skip3 = self.enc_block3(enc2)
        enc3 = self.down3(skip3)

        mid = self.mid_block(enc3)

        dec3_lower = self.up3(mid)
        dec3 = torch.cat((dec3_lower, skip3), dim=1)
        out3 = self.dec_block3(dec3)

        dec2_lower = self.up2(out3)
        dec2 = torch.cat((dec2_lower, skip2), dim=1)
        out2 = self.dec_block2(dec2)

        dec1_lower = self.up1(out2)
        dec1 = torch.cat((dec1_lower, skip1), dim=1)
        out1 = self.dec_block1(dec1)

        out3_f = self.out3_conv(out3)
        out3_f = self.out3_upsample(out3_f)

        out2_f = self.out2_conv(out2)
        out2_f = self.out2_upsample(out2_f)

        out1_f = self.out1_conv(out1)
        
        out = torch.cat((out3_f, out2_f, out1_f), dim=1)
        return out
        
        # out = self.out3_conv(out3)
        # out = self.out2_conv(out2) + self.out3_upsample(out)
        # out = self.out1_conv(out1) + self.out2_upsample(out)
        # return out
        # return self.out1_conv(out1)