import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6,
                            kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


test = Test()

writer = SummaryWriter("logs")
step = 0

for data in tqdm(dataloader):
    imgs, targets = data
    output = test(imgs)
    # print(imgs.shape) # shape(64, 3, 32, 32)
    # print(output.shape)  # shape(64, 6, 30, 30)
    writer.add_images("input", imgs, step)

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step += 1

    # print('%.2f %%' % ((index / len(dataloader)) * 100))

writer.close()
