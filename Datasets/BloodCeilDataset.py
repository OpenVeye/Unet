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

class BloodCeilDataset(Dataset):
    def __init__(self,data_dir =r"data\BCCD Dataset with mask",image_size=(320,320),is_train=True,transform=None):
        super().__init__()
        self.images_name_list,self.masks_name_list = self.read_data_from_dir(data_dir,is_train)
        self.is_train=is_train
        self.image_size=image_size
        self.transform = transform
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]
        )
    def __len__(self):
        return len(self.images_name_list)
    def __getitem__(self, item):
        image_name = self.images_name_list[item]
        mask_name = self.masks_name_list[item]
        image_pil = Image.open(image_name).convert("RGB")
        mask_pil = Image.open(mask_name)
        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil)
        aug_dict={"image":image_np,"mask":mask_np}
        albument = get_transform(self.is_train,self.image_size)(**aug_dict)
        image_np=albument["image"]
        mask_np = albument["mask"]
        mask_np = mask_np[:,:,0]|mask_np[:,:,1]|mask_np[:,:,2]
        image = self.tf(image_np)
        mask_np[mask_np==255]=1
        mask = torch.from_numpy(mask_np).long()

        return image,mask

    def read_data_from_dir(self,data_dir,is_train=True):
        if is_train:
            data_dir = os.path.join(data_dir,"train")
        else:
            data_dir = os.path.join(data_dir,"test")
        images_name_1=glob.glob(os.path.join(data_dir,"original","*.png"))
        images_name_2=glob.glob(os.path.join(data_dir,"original","*.jpg"))
        images_name = images_name_1+images_name_2
        masks_name = [os.path.join(data_dir,"mask",os.path.basename(img_name).split(".")[0]+".png")for img_name in images_name]
        return images_name,masks_name
def get_dataloader(image_size=(320,320),is_train=True,batch_size=4,num_workers=0):
    return DataLoader(BloodCeilDataset(image_size=image_size,is_train=is_train,transform=get_transform(is_train,image_size)),batch_size,is_train,num_workers=num_workers,pin_memory=True)
if __name__=="__main__":
    for n,(images,masks) in enumerate(get_dataloader(is_train=False)):
        if n>2:
            break
        images_show(images,masks)
