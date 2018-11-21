from torch.nn import Module, Sequential 
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d
from torch.nn import ReLU, Sigmoid

class UNet3D(Module):
    
    def __init__(self, num_channels=32, num_classes=2, feat_channels=[64, 128, 256, 512, 1024]):
        
        super(UNet3D, self).__init__()

        # Encoder downsamplers
        self.pool1 = MaxPool3d((1,2,2))
        self.pool2 = MaxPool3d((1,2,2))
        self.pool3 = MaxPool3d((1,2,2))
        self.pool4 = MaxPool3d((1,2,2))
        
        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0])
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1])
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2])
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3])
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4])

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2*feat_channels[3], feat_channels[3])
        self.dec_conv_blk3 = Conv3D_Block(2*feat_channels[2], feat_channels[2])
        self.dec_conv_blk2 = Conv3D_Block(2*feat_channels[1], feat_channels[1])
        self.dec_conv_blk1 = Conv3D_Block(2*feat_channels[0], feat_channels[0])

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])
        
        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], 1, kernel_size=1, stride=1, padding=1, bias=True)
        
        # Activation function
        self.sigmoid = Sigmoid()

    def forward(self, x):
        
        # Encoder part

        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)
        
        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)
        
        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)
        
        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part
        
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        seg = self.sigmoid(self.one_conv(d_high1))

        return seg

class Conv3D_Block(Module):
        
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1):
        
        super(Conv3D_Block, self).__init__()
        
        self.conv1 = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        Conv3d(out_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

    def forward(self, x):
        return self.conv2(self.conv1(x))

class Deconv3D_Block(Module):
    
    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        
        super(Deconv3D_Block, self).__init__()
        
        self.deconv = Sequential(
                        ConvTranspose3d(inp_feat, out_feat, kernel_size=(1,kernel,kernel), 
                                    stride=(1,stride,stride), padding=(0, padding, padding), output_padding=0, bias=True),
                        ReLU())
    
    def forward(self, x):
        
        return self.deconv(x)

if __name__=='__main__':
    
    net = UNet3D()

    import torch
    x = torch.ones(1, 32, 1, 128, 128)

    print (net.forward(x))
