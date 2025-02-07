import torch
import torch.nn as nn
from torchvision.models import vgg19_bn
#定义卷积块
class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,is_max_pool=False,padding_mode="zeros"):
        super().__init__()
        self.is_max_pool=is_max_pool
        self.max_pool =nn.MaxPool2d(2,2)
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1,bias=False,padding_mode=padding_mode),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,bias=False,padding_mode=padding_mode),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1)
        )
    def forward(self,input):
        return self.conv(self.max_pool(input)if self.is_max_pool else input)

#定义Backbone网络
class Backbone4Unet(nn.Module):
    def __init__(self,class_numb=1000):
        super().__init__()
        #定义Unet特征层部分
        self.features = nn.Sequential(
            ConvBlock(3,64,padding_mode="reflect"),
            ConvBlock(64,128,is_max_pool=True,padding_mode="reflect"),
            ConvBlock(128,256,is_max_pool=True,padding_mode="reflect"),
            ConvBlock(256,512,is_max_pool=True,padding_mode="reflect"),
            ConvBlock(512,1024,is_max_pool=True,padding_mode="reflect"),
        )
        #定义ImageNet1000数据集分类层
        self.avg_pool = nn.Sequential(
            ConvBlock(1024,1024,is_max_pool=True,padding_mode="reflect"),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(1024*1*1,class_numb)
    def forward(self,input):
        out = self.features(input)
        out=self.avg_pool(out)
        b_s = out.size(0)
        out = out.view(b_s,-1)
        return self.classifier(out)
#定义Unet网络
class myUnet(nn.Module):
    def  __init__(self,C=2,is_pretrained=False,is_vgg19_bn=False):
        super().__init__()
        backbone = self.get_backbone(is_pretrained,is_vgg19_bn)

        if is_vgg19_bn:
            self.down_0 = backbone[:6]
            self.down_1 = backbone[6:13]
            self.down_2 = backbone[13:26]
            self.down_3 = backbone[26:39]
            self.down_4 = backbone[39:52]
            for n, parameter in enumerate(self.down_4[-3].parameters()):
                in_channel = parameter.size(0)
                if n==0:
                    break
        else:
            self.down_0 = backbone[0]
            self.down_1 = backbone[1]
            self.down_2 = backbone[2]
            self.down_3 = backbone[3]
            self.down_4 = backbone[4]
            in_channel = 1024
        self.up_4 = nn.ConvTranspose2d(in_channel,in_channel//2,kernel_size=2,stride=2,padding=0)
        self.up_3 = nn.Sequential(
            ConvBlock(in_channel//2+512,512,padding_mode="reflect"),
            nn.ConvTranspose2d(512,256,kernel_size=2,stride=2,padding=0)
        )
        self.up_2 = nn.Sequential(
            ConvBlock(256+256,256,padding_mode="reflect"),
            nn.ConvTranspose2d(256,128,kernel_size=2,stride=2,padding=0)
        )
        self.up_1 = nn.Sequential(
            ConvBlock(128+128,128,padding_mode="reflect"),
            nn.ConvTranspose2d(128,64,kernel_size=2,stride=2,padding=0)
        )
        self.result = nn.Sequential(
            ConvBlock(64+64,64,padding_mode="reflect"),
            nn.Conv2d(64,C,kernel_size=1,padding=0,bias=True)
        )

    def get_backbone(self,is_pretrained=False,is_vgg19_bn=False):
        if is_vgg19_bn:
            backbone=vgg19_bn(is_pretrained).features
        else:
            net = Backbone4Unet()
            if is_pretrained:
                net.load_state_dict(torch.load(r"model\best_unet_backbone.pth"))
            backbone=net.features
        return backbone
    def forward(self,input):
        d_0 = self.down_0(input)
        d_1 = self.down_1(d_0)
        d_2 = self.down_2(d_1)
        d_3 = self.down_3(d_2)
        d_4 = self.down_4(d_3)
        u_3 = self.up_4(d_4)
        u_3 = torch.cat((u_3,d_3),dim=1)
        u_2 = self.up_3(u_3)
        u_2 = torch.cat((u_2,d_2),dim=1)
        u_1 = self.up_2(u_2)
        u_1 = torch.cat((u_1,d_1),dim=1)
        u_0 = self.up_1(u_1)
        u_0 = torch.cat((u_0,d_0),dim=1)
        return self.result(u_0)

def weights_init(net,init_gain = 0.02):
    def init_func(m):
        if isinstance(m, nn.Conv2d):
            #卷积层权重均值为0，标准差默认0.02的正太分布初始化
            torch.nn.init.normal_(m.weight.data, 0, init_gain)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            #BN层权重均值为1，标准差默认0.02的正态分布初始化
            torch.nn.init.normal_(m.weight.data, 1., init_gain)
            torch.nn.init.constant_(m.bias.data, 0.)
    print('initialize network with normal')
    net.apply(init_func)





if __name__=="__main__":
    input = torch.rand(2,3,320,320)
    net = myUnet(is_vgg19_bn=False)
    weights_init(net)
    pred = net(input)
    print(pred.size())
