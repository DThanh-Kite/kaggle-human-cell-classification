import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet34
from base import BaseModel
from .loss import ArcFaceLoss

# This model is taken from
# https://github.com/CellProfiling/HPA-competition-solutions/blob/master/bestfitting/src/networks/resnet.py
class Resnet(BaseModel):
    def __init__(self,
                 backbone='resnet34',
                 num_classes=0,
                 in_channels=3,
                 dropout=False,
                 pretrained=True,
                 config={}):
        super().__init__(config)
        self.dropout = dropout

        if backbone == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
            self.EX = 1
        elif backbone=='resnet34':
            self.resnet = resnet34(pretrained=pretrained)
            self.EX=1
        elif backbone=='resnet50':
            self.resnet = resnet50(pretrained=pretrained)
            self.EX = 4
        elif backbone=='resnet101':
            self.resnet = resnet101(pretrained=pretrained)
            self.EX = 4
        elif backbone=='resnet152':
            self.resnet = resnet152(pretrained=pretrained)
            self.EX = 4
        
        self.in_channels = in_channels
        if self.in_channels > 3:
            # https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
            w = self.resnet.conv1.weight
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3, 3), bias=False)
            self.resnet.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:,:1,:,:]),dim=1))
        
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.logit = nn.Linear(512 * self.EX, num_classes)

        # https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        if self.dropout:
            self.bn1 = nn.BatchNorm1d(1024 * self.EX)
            self.fc1 = nn.Linear(1024 * self.EX, 512 * self.EX)
            self.bn2 = nn.BatchNorm1d(512 * self.EX)
            self.relu = nn.ReLU(inplace=True)
        self.extract_feature = False
    
    def set_configs(self, extract_feature=False, **kwargs):
        self.extract_feature = extract_feature
        
    def forward(self, x):
        # This is from previous competition
        mean = [0.074598, 0.050630, 0.050891, 0.076287] # rgby
        std =  [0.122813, 0.085745, 0.129882, 0.119411]
        for i in range(self.in_channels):
            x[:,i,:,:] = (x[:,i,:,:] - mean[i]) / std[i]

        x = self.encoder1(x)
        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        if self.dropout:
            x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
            x = x.view(x.size(0), -1)
            x = self.bn1(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.bn2(x)
            x = F.dropout(x, p=0.5, training=self.training)
        else:
            x = self.avgpool(e5)
        feature = x.view(x.size(0), -1)
        x = self.logit(feature)

        if self.extract_feature:
            return x, feature
        else:
            return x


class ArcMarginProduct(pl.LightningModule):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


# class ArcFaceModel(nn.Module):
#     def __init__(self,....
#         ... ...
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.arc_margin_product=ArcMarginProduct(512, num_classes)
#         self.bn1 = nn.BatchNorm1d(1024 * self.EX)
#         self.fc1 = nn.Linear(1024 * self.EX, 512 * self.EX)
#         self.bn2 = nn.BatchNorm1d(512 * self.EX)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(512 * self.EX, 512)
#         self.bn3 = nn.BatchNorm1d(512)

#     def forward(self, x):
#         ... ...
#         x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
#         x = x.view(x.size(0), -1)
#         x = self.bn1(x)
#         x = F.dropout(x, p=0.25)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.bn2(x)
#         x = F.dropout(x, p=0.5)

#         x = x.view(x.size(0), -1)

#         x = self.fc2(x)
#         feature = self.bn3(x)

#         cosine=self.arc_margin_product(feature)
#         if self.extract_feature:
#             return cosine, feature
#         else:
#             return cosine