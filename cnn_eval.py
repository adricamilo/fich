import os
import cnn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from natsort import natsorted


class ImageDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


def correcting_orientations(folder: str) -> list[tuple]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((400, 400), antialias=True)
    ])
    dataset = ImageDataSet(folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    predicted = cnn.evaluate(dataloader)
    correcting = list()
    for index, file in enumerate(dataset.total_imgs):
        correcting.append((file, predicted[index]))
    return correcting
