# -- coding: utf-8 --
from .VGG16 import vgg16
from .DRCNN import DRCNN
from .encoding.NetFV import NetFV
from .BasicModule import BasicModule
import torch.nn as nn
import torch
import torchvision.transforms.functional as F
import numpy as np
import torchvision.models as models
from torch.autograd import Variable


# multi layer for feature extraction
class CGFA_CNN_NetFV(BasicModule):
    def __init__(self, load_model_path, num_clusters, with_GUC):
        super(CGFA_CNN_NetFV, self).__init__()
        self.model_name = "CGFA_CNN_NetFV"
        self.with_GUC = with_GUC
        # self.base_model_aut = models.vgg16(pretrained=True).eval()
        self.base_model_aut = vgg16(pretrained=True)
        self.base_model_syn = DRCNN(classes_num=41)
        self.load_model_path = load_model_path
        if (self.load_model_path):
            self.base_model_syn.load(self.load_model_path)

        self.net_fv = NetFV(dim=128, num_clusters=num_clusters)
        self.fc = nn.Linear(128*2*num_clusters, 41)
        self.regression = nn.Linear(41, 1)
        self.conv6 = nn.Conv2d(384+960, 128, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        # features extract from DRCNN pretrained on self-built dataset
        #
        y1 = self.base_model_syn.conv_features[0:9](x)
        y2 = self.base_model_syn.conv_features[9:15](y1)
        y3 = self.base_model_syn.conv_features[15:21](y2)
        y4 = self.base_model_syn.conv_features[21:27](y3)
        y1 = y1.permute(2, 3, 0, 1)
        np.random.shuffle(y1)
        y1 = y1.permute(1, 0, 2, 3)
        np.random.shuffle(y1)
        y1 = y1.permute(2, 3, 0, 1)
        y1 = y1[0:y1.size(0), 0:y1.size(1), 0:y4.size(2), 0:y4.size(3)]

        y2 = y2.permute(2, 3, 0, 1)
        np.random.shuffle(y2)
        y2 = y2.permute(1, 0, 2, 3)
        np.random.shuffle(y2)
        y2 = y2.permute(2, 3, 0, 1)
        y2 = y2[0:y2.size(0), 0:y2.size(1), 0:y4.size(2), 0:y4.size(3)]

        y3 = y3.permute(2, 3, 0, 1)
        np.random.shuffle(y3)
        y3 = y3.permute(1, 0, 2, 3)
        np.random.shuffle(y3)
        y3 = y3.permute(2, 3, 0, 1)
        y3 = y3[0:y3.size(0), 0:y3.size(1), 0:y4.size(2), 0:y4.size(3)]

        x1 = self.base_model_aut.features[0:4](x)
        x2 = self.base_model_aut.features[4:9](x1)
        x3 = self.base_model_aut.features[9:16](x2)
        x4 = self.base_model_aut.features[16:23](x3)

        x1 = x1.permute(2, 3, 0, 1)
        np.random.shuffle(x1)
        x1 = x1.permute(1, 0, 2, 3)
        np.random.shuffle(x1)
        x1 = x1.permute(2, 3, 0, 1)
        x1 = x1[0:x1.size(0), 0:x1.size(1), 0:y4.size(2), 0:y4.size(3)]


        x2 = x2.permute(2, 3, 0, 1)
        np.random.shuffle(x2)
        x2 = x2.permute(1, 0, 2, 3)
        np.random.shuffle(x2)
        x2 = x2.permute(2, 3, 0, 1)
        x2 = x2[0:x2.size(0), 0:x2.size(1), 0:y4.size(2), 0:y4.size(3)]


        x3 = x3.permute(2, 3, 0, 1)
        np.random.shuffle(x3)
        x3 = x3.permute(1, 0, 2, 3)
        np.random.shuffle(x3)
        x3 = x3.permute(2, 3, 0, 1)
        x3 = x3[0:x3.size(0), 0:x3.size(1), 0:y4.size(2), 0:y4.size(3)]


        x4 = x4.permute(2, 3, 0, 1)
        np.random.shuffle(x4)
        x4 = x4.permute(1, 0, 2, 3)
        np.random.shuffle(x4)
        x4 = x4.permute(2, 3, 0, 1)
        x4 = x4[0:x4.size(0), 0:x4.size(1), 0:y4.size(2), 0:y4.size(3)]


        feat_map = torch.cat((x1,x2,x3,x4, y1, y2, y3, y4),1)
        feat_map = self.conv6(feat_map)
        feat = self.net_fv(feat_map)
        feat = self.fc(feat)

        if(self.with_GUC):
            p = self.base_model_syn(x)
            feat = feat*p

        y = self.regression(feat)
        return y
