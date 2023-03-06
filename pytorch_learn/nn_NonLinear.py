import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output


test = Test()

writer = SummaryWriter("logs")
step = 0
for data in tqdm(dataloader):
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = test(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
