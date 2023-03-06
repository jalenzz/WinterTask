import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import *

trans = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.CIFAR10("/home/alee/Desktop/deep_learning/Datasets", train=True, transform=trans,
                                          download=True)
test_data = torchvision.datasets.CIFAR10("/home/alee/Desktop/deep_learning/Datasets", train=True, transform=trans,
                                         download=True)

batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True)


class My_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(16384, 1024),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


my_nn = My_nn()
my_nn = my_nn.cuda()
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
learning_rate = 0.01
optimizer = torch.optim.SGD(my_nn.parameters(), lr=learning_rate)
writer = SummaryWriter(log_dir="../task2_logs")

epoch = 50

train_step = 0
for i in range(epoch):
    train_bar = tqdm(train_loader, desc=f"Train_Epoch[{i+1}/{epoch}]")
    train_loss = 0
    for data in train_bar:
        images, targets = data
        images = images.cuda()
        targets = targets.cuda()
        outputs = my_nn(images)
        loss = loss_fn(outputs, targets)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.add_scalar(tag="Loss", scalar_value=train_loss, global_step=i)
    with torch.no_grad():
        accuracy = 0
        test_bar = tqdm(test_loader, desc=f"Test_Epoch[{i+1}/{epoch}")
        for data in test_bar:
            images, targets = data
            images = images.cuda()
            targets = targets.cuda()
            outputs = my_nn(images)
            loss = loss_fn(outputs, targets)
            acc = (outputs.argmax(1) == targets).sum().item()
            accuracy += acc

            test_bar.set_postfix(acc=accuracy/len(test_data))
        writer.add_scalar("Accuracy", scalar_value=accuracy/len(test_data), global_step=i)

writer.close()