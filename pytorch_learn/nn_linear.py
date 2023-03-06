import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.linear = Linear(196608, 10)

    def forward(self, input):
        return self.linear(input)


test = Test()

for data in tqdm(dataloader):
    imgs, targets = data
    output = test(torch.flatten(imgs))
    print(imgs.shape)
    print(output.shape)
