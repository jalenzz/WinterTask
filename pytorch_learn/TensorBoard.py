from torch.utils.tensorboard import SummaryWriter
import os
import cv2
# import torch

write = SummaryWriter("logs")

dirPath = "data/train"
labelPath = "ants"
path = os.path.join(dirPath, labelPath)
image_list = os.listdir(path)

for i, imageName in enumerate(image_list, 1):
    imagePath = os.path.join(path, imageName)
    image = cv2.imread(imagePath)
    write.add_images("test", image, global_step=i, dataformats="HWC")

for i in range(100):
    write.add_scalar("y=x", i, 2*i)

write.close()
