from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "data/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("log")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_images("Tensor_img", tensor_img)

writer.close()
