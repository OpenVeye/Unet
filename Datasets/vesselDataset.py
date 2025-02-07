import glob
import os

import matplotlib.pyplot as plt
import torch

from my_utils import get_transform,images_show
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms
from PIL import Image
os.chdir(r"F:\segmentation\Unet")

class vesselDataset(Dataset):
    def __init__(self,data_dir=r"data\DRIVE",image_size=(320,320),is_train=True,transform=None):
        super().__init__()
        self.images_name_list,self.masks_name_list=self.read_data_from_dir(data_dir,is_train)
        self.image_size=image_size
        self.transform=transform
        self.tf=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.is_train = is_train
    def __len__(self):
        return len(self.images_name_list)
    def __getitem__(self, item):
        image_name = self.images_name_list[item]
        mask_name = self.masks_name_list[item]
        image_pil = Image.open(image_name).convert("RGB")
        mask_pil = Image.open(mask_name)
        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil)
        aug_dict = {"image":image_np,"mask":mask_np}
        albument=get_transform(self.is_train,self.image_size)(**aug_dict)
        image_np = albument["image"]
        mask_np = albument["mask"]
        image = self.tf(image_np)
        mask_np[mask_np==255]=1
        mask=torch.from_numpy(mask_np)
        return image,mask.long()

    def read_data_from_dir(self,data_dir,is_train=True):
        if is_train:
            data_dir = os.path.join(data_dir,"training")
        else:
            data_dir = os.path.join(data_dir,"test")
        images_name = glob.glob(os.path.join(data_dir,"images","*.tif"))
        masks_name = [os.path.join(data_dir,"1st_manual",os.path.basename(img_name).split("_")[0]+"_manual1.gif")for img_name in images_name]
        # print(images_name[:5])
        # print(masks_name[:5])
        return images_name,masks_name
def get_dataloader(image_size=(320,320),is_train=True,batch_size=4,num_workers=0):
    return DataLoader(vesselDataset(image_size=image_size,is_train=is_train,transform=get_transform(is_train,image_size)),batch_size,is_train,num_workers=num_workers,pin_memory=True)

if __name__=="__main__":
    for images,masks in get_dataloader():
        images_show(images,masks)