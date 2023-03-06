import torch
import torchvision
from torch import nn
from tqdm import tqdm
from model import VGG16Model as TestModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("./dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

train_dataloader = DataLoader(train_data, batch_size=128, drop_last=True, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, drop_last=True, shuffle=True)

# test = TestModel()
#
# for data in train_dataloader:
#     imgs, ta = data
#     print(test(imgs).shape)
#     break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test = TestModel().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
learning_rate = 0.01
optimizer = torch.optim.SGD(test.parameters(), lr=learning_rate)

writer = SummaryWriter("logs/test")
total_train_step = 0
total_test_step = 0
epoch = 30

for i in range(epoch):
    # train
    train_bar = tqdm(train_dataloader, desc="TrainEpoch[{}/{}]".format(i + 1, epoch))
    total_train_loss = 0
    for data in train_bar:
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)
        output = test(images)
        loss = loss_fn(output, targets)
        total_train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar(tag="TrainLoss", scalar_value=loss, global_step=total_train_step)
        total_train_step += 1
    # writer.add_scalar(tag="TrainLoss", scalar_value=total_train_loss, global_step=i)

    # test
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        accuracy = 0
        test_bar = tqdm(test_dataloader, desc=f"Test_Epoch[{i + 1}/{epoch}]")
        for data in test_bar:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            outputs = test(images)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            acc = (outputs.argmax(1) == targets).sum().item()
            accuracy += acc

            test_bar.set_postfix(acc=accuracy / len(test_data))
            writer.add_scalar("TestLoss", scalar_value=loss, global_step=total_test_step)
            total_test_step += 1

        writer.add_scalar("Accuracy", scalar_value=accuracy / len(test_data), global_step=i)

    torch.save(test, "model/test_{}.pth".format(i))

writer.close()
