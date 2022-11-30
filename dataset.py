import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

normalize_sec = transforms.Normalize(mean=0.5,std=0.5)

# trans?????????
trans = transforms.Compose([transforms.RandomCrop(256),
                            transforms.ToTensor(),
                            normalize])
trans_sec = transforms.Compose([transforms.RandomCrop(256),
                            #transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(),
                            normalize_sec])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def denorm_se(tensor, device):
    std = torch.Tensor(0.5).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(0.5).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


class PreprocessDataset(Dataset):
    def __init__(self, content_dir, style_dir, secret_dir, transforms=trans,transform_se =trans_sec ):
        content_dir_resized = content_dir + '_resized'
        style_dir_resized = style_dir + '_resized'
        secret_dir_resized = secret_dir + '_resized'
        if not os.path.exists(content_dir_resized):
            os.mkdir(content_dir_resized)
            self._resize(content_dir, content_dir_resized)
        if not os.path.exists(style_dir_resized):
            os.mkdir(style_dir_resized)
            self._resize(style_dir, style_dir_resized)
        if not os.path.exists(secret_dir_resized):
            os.mkdir(secret_dir_resized)
            self._resize(secret_dir, secret_dir_resized)


        # ??????????????????
        content_images = glob.glob((content_dir_resized + '/*'))
        np.random.shuffle(content_images)
        style_images = glob.glob(style_dir_resized + '/*')
        np.random.shuffle(style_images)
        secret_images = glob.glob(secret_dir_resized + '/*')
        np.random.shuffle(secret_images)
        self.images_pairs = list(zip(content_images, style_images ,secret_images))
        self.transforms = transforms
        self.transforms_ser = transform_se

        # ???????

    @staticmethod
    def _resize(source_dir, target_dir):
        print(f'Start resizing {source_dir} ')
        for i in tqdm(os.listdir(source_dir)):
            filename = os.path.basename(i)
            try:
                image = io.imread(os.path.join(source_dir, i))
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    H, W, _ = image.shape
                    if H < W:
                        ratio = W / H
                        H = 512
                        W = int(ratio * H)
                    else:
                        ratio = H / W
                        W = 512
                        H = int(ratio * W)
                    image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)
                    io.imsave(os.path.join(target_dir, filename), image)
            except:
                continue

    #???????????????????????????

    def __len__(self):
        return len(self.images_pairs)

    def __getitem__(self, index):
        content_image, style_image, secret_images = self.images_pairs[index]
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)
        secret_image = Image.open(secret_images)
        # content_image = io.imread(content_image, plugin='pil')
        # style_image = io.imread(style_image, plugin='pil')
        # Unfortunately,RandomCrop doesn't work with skimage.io
        if self.transforms:
            content_image = self.transforms(content_image)
            style_image = self.transforms(style_image)
            secret_image = self.transforms_ser(secret_image)
        return content_image, style_image, secret_image
