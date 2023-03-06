import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=test_transform, download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=test_transform, download=True)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# images, target = test_data[0]
# print(images.shape)
# print(target)

writer = SummaryWriter("dataloader")
step = 0

for index, data in enumerate(test_loader):
    imgs, target = data
    writer.add_images("test_data", imgs, step)
    print('%.2f %%' %((index/len(test_loader))*100))
    step += 1
writer.close()
