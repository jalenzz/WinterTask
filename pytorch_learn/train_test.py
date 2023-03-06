import torch
import torchvision
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

train_dataloader = DataLoader(train_data, batch_size=128, drop_last=True, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, drop_last=True, shuffle=True)


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=4096 * 4, out_features=1024),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        x = self.module(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test = TestModel().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
learning_rate = 0.01
optimizer = torch.optim.SGD(test.parameters(), lr=learning_rate)

writer = SummaryWriter("logs")
total_train_step = 0
total_test_step = 0
epoch = 100
train_loss = 0
test_loss = 0

for i in range(epoch):
    # train
    train_bar = tqdm(train_dataloader, desc="TrainEpoch[{}/{}]".format(i + 1, epoch))
    for data in train_bar:
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)
        output = test(images)
        loss = loss_fn(output, targets)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
    writer.add_scalar("tag=""Loss", scalar_value=train_loss, global_step=i)

    # test
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        accuracy = 0
        test_bar = tqdm(test_dataloader, desc=f"Test_Epoch[{i + 1}/{epoch}")
        for data in test_bar:
            images, targets = data
            images = images.cuda()
            targets = targets.cuda()
            outputs = test(images)
            loss = loss_fn(outputs, targets)
            acc = (outputs.argmax(1) == targets).sum().item()
            accuracy += acc

            test_bar.set_postfix(acc=accuracy / len(test_data))
        writer.add_scalar("Accuracy", scalar_value=accuracy / len(test_data), global_step=i)

    torch.save(test, "test_{}.pth".format(i))

writer.close()
