import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import torch
import numpy as np
from torch.autograd import Variable
from pytorch_msssim import ssim, ms_ssim
from PIL import Image
import matplotlib.pyplot as plt
import datetime

def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std

#adain???
def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4

    # ??????????????output_last_feature??????????????
class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=False):
        out_list = []
        out_list.append(images)
        h1 = self.slice1(images)
        out_list.append(h1)
        h2 = self.slice2(h1)
        out_list.append(h2)
        h3 = self.slice3(h2)
        out_list.append(h3)
        h4 = self.slice4(h3)
        out_list.append(h4)
        if output_last_feature:
            return h4
        else:
            # print(h1.shape)
            # print(h2.shape)
            # print(h3.shape)
            # print(h4.shape)
            return out_list

class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h

class Extract(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv =nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
            #nn.Sigmoid()
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(0.0, 0.01)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self,input):
        x=self.conv(input)
        return x

# 这里打算把解码网络改一下，让它可以更好的适应残

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 这里原来是512 -> 256 被改成了1024 -> 256
        # 这里是将特征提取网络提取的特征放入512+512 -> 1024
        self.rc1 = RC(1024, 256, 3, 1)  #slice9
        # 256+128 -> 384
        self.rc2 = RC(512, 256, 3, 1)
        #self.rc2 = RC(384, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        # 128+64 -> 192
        #self.rc6 = RC(192, 128, 3, 1)
        self.rc6 = RC(256, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(128, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1)
        # 这一块是多加的(rc10)
        # 3+3 -> 4
        self.rc10 = RC(6, 3, 3, 1,False)

    def forward(self, features, extra_feature):
        print(extra_feature[0].shape)
        h = torch.cat((features, extra_feature[4]), 1)
        h = self.rc1(h)
        h = F.interpolate(h, scale_factor=2)
        h = torch.cat((h, extra_feature[3]), 1)
        # ???????????
        h = self.rc2(h)

        #h = torch.cat((h, extra_feature[3]), 1)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = torch.cat((h, extra_feature[2]), 1)
        h = self.rc6(h)
        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)
        h = torch.cat((h, extra_feature[1]), 1)
        h = self.rc8(h)
        h = self.rc9(h)
        h = torch.cat((h,extra_feature[0]),1)
        h = self.rc10(h)
        return h

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()
        self.sec_fea_ext = VGG()
        self.extract = Extract()

    def generate(self, content_images, style_images, secret_images,alpha=1.0):
        # alpha????????????
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        # t.shape 1x 512x 32x 32
        # 在此处加入了特征提取和连接的部分（连接了secret_images 和 adain的特征，相当于将三个图像的特征弄在了一起）
        sec_fea = self.sec_fea_ext(secret_images)
        # print(sec_fea[3].shape)
        # ttt=sec_fea[3].cpu()
        # #
        # ## 对中间层进行可视化,可视化64个特征映射
        # plt.figure(figsize=(16, 16))
        # for ii in range(ttt.shape[1]):
        #     ## 可视化每张手写体
        #     plt.subplot(16, 16, ii + 1)
        #     plt.imshow(ttt.data.numpy()[0, ii, :, :], cmap="jet")
        #     plt.axis("off")
        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        # mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S')  # 这里是引用时间
        # plt.savefig('see/'+str(mkfile_time)+'.png')
        # plt.show()
        #sec_fea =self.sec_fea_ext(secret_images )#,output_last_feature=False)     #尝试隐藏彩色图像
        out = self.decoder(t,sec_fea)
        return out

    def extractor_model(self, out_image):
        extraaa = self.extract(out_image)
        return extraaa


    @staticmethod
    def calc_content_loss(out_features, t):
        return F.mse_loss(out_features, t)

    @staticmethod
    def calc_style_loss(content_middle_features, style_middle_features):
        loss = 0
        for c, s in zip(content_middle_features, style_middle_features):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    # ???????????????????????????????????????

    def forward(self, content_images, style_images,secret_images, alpha=1.0, lam=10):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        sec_features = self.sec_fea_ext(secret_images)     #彩色图像


        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        # 此处也加入了秘密图像特征提取部分
        # 这里改了一下，把原来的特征直接加进去改为像U型那样添加
        #print('1111', t.shape1
        out = self.decoder(t,sec_features)
        secret_extract = self.extract(out)

        output_features = self.vgg_encoder(out, output_last_feature=True)
        output_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)

#################secret的损失#########使用
        # criterion = MS_SSIM_L1_LOSS().cuda()
        # loss_secret = criterion(secret_images,secret_extract)

        loss_mse = self.calc_content_loss(secret_extract, secret_images)
        loss_ssim=ssim(secret_extract, secret_images,size_average=True)
        loss_mssim=ms_ssim(secret_extract, secret_images,size_average=True)
        # #print(type(loss_mssim))
        loss_secret = 0.3*loss_mse+0.5*(1 - loss_ssim) + 0.5*(1-loss_mssim)
        loss_c = self.calc_content_loss(output_features, t)
        loss_s = self.calc_style_loss(output_middle_features, style_middle_features)
        loss =  loss_c + loss_s + 5*loss_secret
        return loss ,loss_secret

    #????????
