import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from resnet50_vggface import resnet50



class fc_block(nn.Module):
    def __init__(self, inplanes, planes, drop_rate):
        super(fc_block, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        else:
            self.dropout = nn.Dropout(0.01)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x



class Resnet50_celeba(torch.nn.Module):


    def __init__(self):

        super(Resnet50_celeba, self).__init__()


        model = resnet50(weights_path= './resnet50_ft_weight.pkl')
        self.backbone = model
        self.fc_b1 = fc_block(2048, 512, 0.1)
        # self.fc_b2 = fc_block(2048, 1024, 0.1)
        # self.fc_b3 = fc_block(1024, 512, 0.1)

        for i in range(6):
            setattr(self, 'classifier' + str(i).zfill(2), nn.Sequential(fc_block(512, 128, 0.15), fc_block(128, 64, 0.15),  nn.Linear(64, 2)))

    def forward(self, data):
        #batch_size, 3, 224,224
        x = self.backbone(data)
        # x = x.view(x.size(0), -1) # batch_size,51200
        x = self.fc_b1(x)
        # x = self.fc_b2(x)
        # x = self.fc_b3(x)

        y = []
        for i in range(2):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            y.append(classifier(x))

        return y
