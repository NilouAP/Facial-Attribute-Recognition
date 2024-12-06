import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import os
import os.path
import PIL
import numpy as np

def transform_resnet(img, mean_bgr = np.array([91.4953, 103.8827, 131.0912])):
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)

    img -=  mean_bgr
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()

    return img


class Celebadataset(data.Dataset):
    def __init__(self, root, ann_file, transform=transform_resnet, target_transform=None):
        images = []
        targets = []
        n_attr = [21,40]

        root_ = './'
        for line in open(os.path.join(root_, ann_file), 'r'):
            sample = line.split()
            images.append(sample[0])
            # targets.append([int(i) for i in sample[1:]])
            targets.append([int(sample[index]) for index in n_attr])
        self.imagename=images
        self.images = [os.path.join(root, img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform


		
    def __getitem__(self, index):
        path = self.images[index]
        imagename = self.imagename[index]
        sample = PIL.Image.open(path)
        sample = torchvision.transforms.Resize(224)(sample)
        sample = np.array(sample, dtype=np.uint8)
        if self.transform is not None:
            sample = self.transform(sample)
        target = self.targets[index]
        target = torch.LongTensor(target)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        #return sample, target
        return sample, target,imagename

    def __len__(self):
        return len(self.images)

