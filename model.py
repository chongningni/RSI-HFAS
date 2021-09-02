from model import common
import time
import torch
import torch.nn as nn
import math
import torchvision.utils as SI

def make_model(args, parent=False):
    return metafpn1(args)


class Pos2Weight(nn.Module):
    def __init__(self, inC, kernel_size=3, outC=3):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.kernel_size * self.kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return out


class FPN(nn.Module):
    def __init__(self, G0, kSize=3):
        super(FPN, self).__init__()

        kSize1 = 1
        self.conv1 = RDB_Conv(G0, G0, kSize)
        self.conv2 = RDB_Conv(G0, G0, kSize)
        self.conv3 = RDB_Conv(G0, G0, kSize)
        self.conv4 = RDB_Conv(G0, G0, kSize)
        self.conv5 = RDB_Conv(G0, G0, kSize)
        self.conv6 = RDB_Conv(G0, G0, kSize)
        self.conv7 = RDB_Conv(G0, G0, kSize)
        self.conv8 = RDB_Conv(G0, G0, kSize)
        self.conv9 = RDB_Conv(G0, G0, kSize)
        self.conv10 = RDB_Conv(G0, G0, kSize)
        self.compress_in1 = nn.Conv2d(4 * G0, G0, kSize1, padding=(kSize1 - 1) // 2, stride=1)
        self.compress_in2 = nn.Conv2d(3 * G0, G0, kSize1, padding=(kSize1 - 1) // 2, stride=1)
        self.compress_in3 = nn.Conv2d(2 * G0, G0, kSize1, padding=(kSize1 - 1) // 2, stride=1)
        self.compress_in4 = nn.Conv2d(2 * G0, G0, kSize1, padding=(kSize1 - 1) // 2, stride=1)
        self.compress_out = nn.Conv2d(4 * G0, G0, kSize1, padding=(kSize1 - 1) // 2, stride=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x11 = x + x4
        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x5_res = self.compress_in1(x5)
        x5 = self.conv5(x5_res)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x12 = x5_res + x7
        x8 = torch.cat((x5, x6, x7), dim=1)
        x8_res = self.compress_in2(x8)
        x8 = self.conv8(x8_res)
        x9 = self.conv9(x8)
        x13 = x8_res + x9
        x10 = torch.cat((x8, x9), dim=1)
        x10_res = self.compress_in3(x10)
        x10 = self.conv10(x10_res)
        x14 = x10_res + x10
        output = torch.cat((x11, x12, x13, x14), dim=1)
        output = self.compress_out(output)
        output = output + x
        return output


class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        kSize = 3
        kSize1 = 1

        self.fpn1 = FPN(num_features)
        self.fpn2 = FPN(num_features)
        self.fpn3 = FPN(num_features)
        self.fpn4 = FPN(num_features)
        self.compress_in = nn.Conv2d(2 * num_features, num_features, kSize1, padding=(kSize1 - 1) // 2, stride=1)
        self.compress_out = nn.Conv2d(4 * num_features, num_features, kSize1, padding=(kSize1 - 1) // 2, stride=1)

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)  # tense拼接
        x = self.compress_in(x)

        fpn1 = self.fpn1(x)
        fpn2 = self.fpn2(fpn1)
        fpn3 = self.fpn3(fpn2)
        fpn4 = self.fpn4(fpn3)
        output = torch.cat((fpn1, fpn2, fpn3, fpn4), dim=1)
        output = self.compress_out(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True


class metafpn1(nn.Module):
    def __init__(self, args, act_type='prelu', norm_type=None):
        super(metafpn1,
              self).__init__()  # 第一句话，调用父类的构造函数，这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化。当然，如果初始化的逻辑与父类的不同，不使用父类的方法，自己重新初始化也是可以的。

        kernel_size = args.RDNkSize
        self.num_steps = 4
        self.num_features = args.G0
        self.scale_idx = 0
        self.scale = 1
        in_channels = args.n_colors
        num_groups = 6
        self.args = args

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # LR feature extraction block
        self.conv_in = common.ConvBlock(in_channels, 4 * self.num_features,
                                        # 3×3Conv      一个卷积核产生一个feature map就是num_features
                                        kernel_size=3,
                                        act_type=act_type, norm_type=norm_type)
        self.feat_in = common.ConvBlock(4 * self.num_features, self.num_features,
                                        kernel_size=1,
                                        act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = FeedbackBlock(self.num_features, num_groups, self.scale, act_type, norm_type)

        # reconstruction block
        # uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        # self.out = DeconvBlock(num_features, num_features,
        #                        kernel_size=kernel_size, stride=stride, padding=padding,
        #                        act_type='prelu', norm_type=norm_type)
        self.P2W = Pos2Weight(inC=self.num_features)

    def repeat_x(self, x):
        scale_int = math.ceil(self.scale)
        N, C, H, W = x.size()
        x = x.view(N, C, H, 1, W, 1)

        x = torch.cat([x] * scale_int, 3)
        x = torch.cat([x] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, pos_mat):
        self._reset_state()

        x = self.sub_mean(x)
        scale_int = math.ceil(self.scale)
        # uncomment for pytorch 0.4.0
        # inter_res = self.upsample(x)

        # comment for pytorch 0.4.0
        inter_res = nn.functional.interpolate(x, scale_factor=scale_int, mode='bilinear', align_corners=False)

        x = self.conv_in(x)
        x = self.feat_in(x)

        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)
            
            #output1 = h.clone()
           # for i in range(60):
             #   output2 = output1[:,i:i+3,:,:]
              #  SI.save_image(output2,"results/result"+str(i)+".png")
            
            # meta###########################################
            local_weight = self.P2W(
                pos_mat.view(pos_mat.size(1), -1))  ###   (outH*outW, outC*inC*kernel_size*kernel_size)
            up_x = self.repeat_x(h)  ### the output is (N*r*r,inC,inH,inW)

            # N*r^2 x [inC * kH * kW] x [inH * inW]
            cols = nn.functional.unfold(up_x, 3, padding=1)
            scale_int = math.ceil(self.scale)

            cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                          1).permute(0, 1, 3, 4, 2).contiguous()

            local_weight = local_weight.contiguous().view(x.size(2), scale_int, x.size(3), scale_int, -1, 3).permute(1,
                                                                                                                     3,
                                                                                                                     0,
                                                                                                                     2,
                                                                                                                     4,
                                                                                                                     5).contiguous()
            local_weight = local_weight.contiguous().view(scale_int ** 2, x.size(2) * x.size(3), -1, 3)

            out = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
            out = out.contiguous().view(x.size(0), scale_int, scale_int, 3, x.size(2), x.size(3)).permute(0, 3, 4, 1, 5,
                                                                                                          2)
            out = out.contiguous().view(x.size(0), 3, scale_int * x.size(2), scale_int * x.size(3))

            h = torch.add(inter_res, out)
            h = self.add_mean(h)
                 
            outs.append(h)

        return outs  # return output of every timesteps

    def _reset_state(self):
        self.block.reset_state()

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        self.scale = self.args.scale[scale_idx]
