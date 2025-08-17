import cv2
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset

class EncoderDataSet(Dataset):
    def __init__(self, src_path, transform=False):
        super().__init__()
        self.transform = transform

        file_list = os.scandir(src_path)

        self.data_list = []

        for file in file_list:
            if file.name.endswith('png'):
                self.data_list.append(file.path)

        print(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        data_path = self.data_list[idx]
        image = cv2.imread(data_path)
        image_target = image.copy()

        if self.transform:

            if random.random() < 0.5:
                image = np.fliplr(image)
                image_target = np.fliplr(image_target)
                
            num_mask = 500 #random.randint(1, 10)
            height, width, _ = image.shape

            for i in range(num_mask):
                mask_size = random.randint(1, 3)
                x = random.randint(0, width-mask_size-1)
                y = random.randint(0, height-mask_size-1)

                image[y:y+mask_size, x:x+mask_size, :] = 0
                # image[y:y+mask_size, x:x+mask_size, 2] = 255
                    

        image = np.transpose(image, [2, 0, 1])
        image = image.astype(np.float32) / 255
        image = torch.tensor(image)
    
        image_target = np.transpose(image_target, [2, 0, 1])
        image_target = image_target.astype(np.float32) / 255
        image_target = torch.tensor(image_target)

        return image, image_target
    
        

if '__main__' == __name__:
    dataset = EncoderDataSet('./data')
    image, image_target = dataset[0]
