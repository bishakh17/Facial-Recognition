import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class Config():
    training_dir = "./data/training/"
    testing_dir = "./data/testing/"
    own_dir = "./data/own/"
    train_batch_size = 64
    train_number_epochs = 100



class FaceDataset(Dataset):

    def __init__(self, face_dir, transform=None):
        self.face_dir = face_dir
        self.transform = transform

    def __len__(self):
        return len(self.face_dir.imgs)

    def __getitem__(self, index):
        img0_tuple = random.choice(self.face_dir.imgs)

        should_get_same_class = random.randint(0,1) 

        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.face_dir.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.face_dir.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    



if __name__ == "__main__":
    folder_dataset = dset.ImageFolder(root=Config.training_dir)

    TRANSFORM = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])
    
    siamese_dataset = FaceDataset(face_dir=folder_dataset,transform=TRANSFORM)

    print(f"There are {len(siamese_dataset)} samples in the dataset.")
    img1,img2,label = siamese_dataset[0]
    print(f"Shape of image: {img1.shape}")          