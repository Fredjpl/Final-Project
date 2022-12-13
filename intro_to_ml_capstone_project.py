import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

import os 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import cv2
import pickle as pkl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LazyLoadDataset(Dataset):
  def __init__(self, path, train=True, transform=None):
    self.transform = transform
    path = path + ("train/" if train else "test/")

    self.pathX = path + "X/"
    self.pathY = path + "Y/"

    self.data = os.listdir(self.pathX)

  def __getitem__(self, idx):
    f = self.data[idx]

    # X
    # read rgb images
    img0 = cv2.imread(self.pathX + f + "/rgb/0.png")
    img1 = cv2.imread(self.pathX + f + "/rgb/1.png")
    img2 = cv2.imread(self.pathX + f + "/rgb/2.png")
    if self.transform is not None:
      img0 = self.transform(img0)
      img1 = self.transform(img1)
      img2 = self.transform(img2)

    # read depth images 
    depth = np.load(self.pathX + f + "/depth.npy")
    # read field ID
    field_id = pkl.load(open(self.pathX + f + "/field_id.pkl", "rb"))

    # Y
    Y = np.load(self.pathY + f + ".npy")

    return (img0, img1, img2, depth, field_id), Y

  def __len__(self):
    return len(self.data)
  
data = LazyLoadDataset("./lazydata/", train = True, transform = transforms.Compose([transforms.ToTensor()]))
  
means = torch.rand_like(torch.tensor([0, 0, 0]), dtype=torch.float)
stds = torch.rand_like(torch.tensor([0, 0, 0]), dtype=torch.float)


for x, y in data:
  img0, img1, img2, depth, field_id = x

  img0_mean = img0.reshape(3, -1).mean(axis=1)
  img1_mean = img1.reshape(3, -1).mean(axis=1)
  img2_mean = img2.reshape(3, -1).mean(axis=1)

  img0_std = img0.reshape(3, -1).std(axis=1)
  img1_std = img1.reshape(3, -1).std(axis=1)
  img2_std = img2.reshape(3, -1).std(axis=1)

  img_mean = (img0_mean + img1_mean + img2_mean) / 3
  img_std = (img0_std + img1_std + img2_std) / 3
  
  means += img_mean
  stds += img_std 


means = means / len(data.data)
stds = stds / len(data.data)

print(means)
print(stds)

train_dataset = LazyLoadDataset("./lazydata/",train = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = means, std = stds)]))

(img0, img1, img2, depth, field_id), Y = train_dataset[0]

len(train_dataset.data)

img0.shape, img1.shape, img2.shape, depth.shape, Y.shape

field_id

import matplotlib.pyplot as plt

plt.imshow(img0[0])
#plt.imshow(depth[0])
plt.show()

depth = np.array(depth)
depth = torch.from_numpy(depth)

type(depth)

img0 = img0.to(device)
img1 = img1.to(device)
img2 = img2.to(device)
depth = depth.to(device)

images = torch.cat((img0, img1, img2, depth), dim=0)

images.shape

import torch
import torch.nn as nn



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=12, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(12, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)



def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

model = resnet152().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

len(train_dataloader.dataset)

for x, y in train_dataloader:
  img0, img1, img2, depth, field_id = x

  depth = np.array(depth)
  depth = torch.from_numpy(depth)

  img0 = img0.to(device)
  img1 = img1.to(device)
  img2 = img2.to(device)
  depth = depth.to(device)

  x = torch.cat((img0, img1, img2, depth), dim=1)

  print(x.shape)

  x = x.to(device)
  y = y.to(device)

  pred = model(x)
  print(pred.shape)
  print(y.shape)

  break

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # send to device
        img0, img1, img2, depth, field_id = X

        depth = np.array(depth)
        depth = torch.from_numpy(depth)

        depth /= 1000

        min_d = 0
        max_d = 65.535

        depth = (depth - min_d) / (max_d - min_d)

        img0 = img0.to(device)
        img1 = img1.to(device)
        img2 = img2.to(device)
        depth = depth.to(device)

        X = torch.cat((img0, img1, img2, depth), dim=1)

        X = X.type(torch.float)

        X = X.to(device)

        y = y.type(torch.float)

        y = y.to(device)
        
        pred = model(X) / 1000
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
          loss, current = loss.item(), batch * len(X)
          print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

epochs = 100

for t in range(epochs):
  print(f"Epoch {t+1} \n------------------------------")
  train(train_dataloader, model, loss_fn, optimizer)
print("Done!")

class LazyLoadTestDataset(Dataset):
  def __init__(self, path, test=True, transform=None):
    self.transform = transform
    path = path + "test/"

    self.pathX = path + "X/"

    self.data = os.listdir(self.pathX)

  def __getitem__(self, idx):
    f = self.data[idx]

    # X
    # read rgb images
    img0 = cv2.imread(self.pathX + f + "/rgb/0.png")
    img1 = cv2.imread(self.pathX + f + "/rgb/1.png")
    img2 = cv2.imread(self.pathX + f + "/rgb/2.png")
    if self.transform is not None:
      img0 = self.transform(img0)
      img1 = self.transform(img1)
      img2 = self.transform(img2)

    # read depth images 
    depth = np.load(self.pathX + f + "/depth.npy")
    # read field ID
    field_id = pkl.load(open(self.pathX + f + "/field_id.pkl", "rb"))


    return (img0, img1, img2, depth, field_id)

  def __len__(self):
    return len(self.data)

test_dataset = LazyLoadTestDataset("./lazydata/", transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.435, 0.462, 0.485], std = [0.236, 0.221, 0.223] )]))

test_dataloader = DataLoader(test_dataset, batch_size = 64)

import pickle
import pandas as pd

outfile = 'submission.csv'

output_file = open(outfile, 'w')

titles = ['ID', 'FINGER_POS_1', 'FINGER_POS_2', 'FINGER_POS_3', 'FINGER_POS_4', 'FINGER_POS_5', 'FINGER_POS_6',
         'FINGER_POS_7', 'FINGER_POS_8', 'FINGER_POS_9', 'FINGER_POS_10', 'FINGER_POS_11', 'FINGER_POS_12']
preds = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_ids = []

model.eval()


size = len(test_dataloader.dataset)

for batch, X in enumerate(test_dataloader):
        # send to device
        img0, img1, img2, depth, field_id = X

        depth = np.array(depth)
        depth = torch.from_numpy(depth)

        depth /= 1000

        min_d = 0
        max_d = 65.535

        depth = (depth - min_d) / (max_d - min_d)

        img0 = img0.to(device)
        img1 = img1.to(device)
        img2 = img2.to(device)
        depth = depth.to(device)

        X = torch.cat((img0, img1, img2, depth), dim=1)

        X = X.type(torch.float)

        X = X.to(device)

        output = model(X) / 1000

        print(batch)

        for i in range(output.shape[0]):
          preds.append(output[i].cpu().detach().numpy())
          file_ids.append(field_id[i])

df = pd.concat([pd.DataFrame(file_ids), pd.DataFrame.from_records(preds)], axis = 1, names = titles)
df.columns = titles
df.to_csv(outfile, index = False)
print("Written to csv file {}".format(outfile))
