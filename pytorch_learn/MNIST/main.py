import torch
import torchvision
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Number(nn.Module):
    def __init__(self):
        super(Number, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=4*4*256, out_features=10)
        )

    def forward(self, x):
        return self.module(x)


train_data = torchvision.datasets.MNIST("./dataset", train=True,
                                        transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

train_dataloader = DataLoader(train_data, batch_size=64, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=64, drop_last=True)

# test = Number()
#
# for data in train_dataloader:
#     images, targets = data
#     output = test(images)
#     print(images.shape)
#     print(output.shape)
#     break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test = Number().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
learning_rate = 1e-2
optimizer = torch.optim.SGD(test.parameters(), lr=learning_rate)

writer = SummaryWriter("logs")
train_loss = 0
test_loss = 0
total_train_step = 0
total_test_step = 0
epoch = 10

for i in range(epoch):
    # train
    train_bar = tqdm(train_dataloader, desc="TrainEpoch[{}/{}]".format(i, epoch))
    for data in train_bar:
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)
        output = test(images)
        train_loss = loss_fn(output, targets)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_train_step += 1
    writer.add_scalar("tag=""Loss", scalar_value=train_loss, global_step=i)

    # test
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        test_bar = tqdm(test_dataloader, desc="TestEpoch[{}/{}]".format(i, epoch))
        for data in test_bar:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            output = test(images)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            total_accuracy += (output.argmax(1) == targets).sum().item()
            total_accuracy_rate = total_accuracy / len(test_data)
            test_bar.set_postfix(acc=total_accuracy_rate)

    writer.add_scalar(tag="test_loss", scalar_value=total_test_loss, global_step=test_loss)
    writer.add_scalar(tag="test_accuracy", scalar_value=total_accuracy_rate, global_step=total_test_step)
    total_test_step += 1

    torch.save(test, "test_{}.pth".format(i))

writer.close()
