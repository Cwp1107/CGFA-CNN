from config import DefaultConfig
from models import CGFA_CNN_NetFV
from data.Dataloader_IQA import load_data_tr_te

from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from torchnet import meter
from scipy.stats import spearmanr, pearsonr

import numpy as np
import csv

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
])

options = {'fc': True}
root = 'your path of model'
model = CGFA_CNN_NetFV(load_model_path=root,num_clusters=opt.num_clusters, with_GUC=True).cuda()
model_name = type(model).__name__
print(model)

ckpt = "your path of the checkpoint file"
image_name = "your path of test image"
checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint)

model.eval()

I = Image.open(image_name)
I = test_transform(I)
I = torch.unsqueeze(I, dim=0)
I = I.to(device)
with torch.no_grad():
    score = model(I)

format_str = 'Prediction = %.4f'
print(format_str % score)






