import torch.cuda
import os
from my_utils import save_roi_images,vedio_concatenate_show
from Unet import myUnet

def predict4images(images_dir,image_size=(320,256),C=2,weight_file=r"model\best_unet.pth"):
    device="cuda"if torch.cuda.is_available() else"cpu"
    #建立网络
    net=myUnet(C).to(device)
    #导入Unet权重
    is_pretrained=True
    if is_pretrained:

        if os.path.exists(weight_file):
            weight_dict = torch.load(weight_file)
            net_dict = net.state_dict()
            temp_dict = {k: v for k, v in weight_dict.items() if
                         k in net_dict.keys() and net_dict[k].numel() == v.numel()}
            for k, v in temp_dict.items():
                net_dict[k] = v
            net.load_state_dict(net_dict)
            print(f"权重文件存在，模型成功导入参数")
    net.eval()
    #新建抠图前景图片保存文件夹
    if not os.path.exists("predict"):
        os.mkdir("predict")
    #获取给定文件里路径下的图片名称
    images_name_list = os.listdir(images_dir)
    images_name_list=[os.path.join(images_dir,img_name)for img_name in images_name_list]
    save_roi_images(net,images_name_list,image_size,is_show=False)

#视频流前景实时抠图
def predict4video(image_size=(320,320),video_name=r"data\ceil.mp4",weight_file=r"model\best_unet.pth",C=2):
    device="cuda"if torch.cuda.is_available()else "cpu"
    #搭建网络
    net = myUnet(C).to(device)
    #需要训练网络的权重
    is_pretrained=True
    if is_pretrained:

        if os.path.exists(weight_file):
            weight_dict = torch.load(weight_file)
            net_dict = net.state_dict()
            temp_dict = {k: v for k, v in weight_dict.items() if
                         k in net_dict.keys() and net_dict[k].numel() == v.numel()}
            for k, v in temp_dict.items():
                net_dict[k] = v
            net.load_state_dict(net_dict)
            print(f"权重文件存在，模型成功导入参数")
    net.eval()
    vedio_concatenate_show(net,video_name,image_size,is_roi_or_mask=False)

if __name__=="__main__":
    predict4images(r"data\BCCD Dataset with mask\test\original",(400,400)) #默认测试红细胞数据集
