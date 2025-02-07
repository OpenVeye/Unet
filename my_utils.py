import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from matplotlib.colors import Normalize
from torchvision.transforms import transforms
from PIL import Image
import os
from IPython.display import display,clear_output
# import matplotlib
# matplotlib.use("Qt5Agg") #使用Qt5作为后端

#数据增强albumentations对象
def get_transform(is_train=True,image_size=(320,256)):
    if is_train:
        transform=A.Compose(#组合调用
            [
                A.LongestMaxSize(max_size=max(image_size)+16),#图像缩放，保持原图纵横比
                A.PadIfNeeded(image_size[0]+16,image_size[1]+16,border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0),#图片填充
                A.RandomResizedCrop(image_size),#随机裁剪
                A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.1,rotate_limit=20,p=0.5),#平移缩放旋转
                A.OneOf(#随机选择一个增强模块
                    [
                        A.MotionBlur(p=0.2),#随机大小的核用于模糊图片
                        A.RandomGamma(gamma_limit=(60,120),p=0.5),#随机gamma变换
                        A.RandomBrightnessContrast(brightness_limit=0.2,p=0.6),#对比度和亮度随机
                        A.CLAHE(clip_limit=4.,tile_grid_size=(4,4),p=0.5)#自适应直方图均衡化
                    ],p=1.
                ),
                A.HorizontalFlip(p=0.5)#随机水平翻转
            ]
        )
    else:
        transform = A.Compose(
            [
                A.LongestMaxSize(max_size=max(image_size)),  # 图像缩放，保持原图纵横比
                A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Resize(image_size[0],image_size[1])
            ]
        )
    return transform

#原图、掩膜、前景抠图显示函数
def images_show(images,masks):
    images=images.cpu().permute(0,2,3,1).contiguous().numpy()
    masks = masks.cpu().numpy()*255
    mean = np.array([0.485,0.456,0.406]).reshape(-1,1,1,3)
    std = np.array([0.229,0.224,0.225]).reshape(-1,1,1,3)
    images = ((images*std+mean)*255).astype(np.uint8)
    n_row=images.shape[0]
    fig,axes=plt.subplots(n_row,3)
    for i, img in enumerate(images):
        mask = masks[i]
        aoi_mask = mask>0
        aoi_img = np.zeros_like(img).astype(np.uint8)
        aoi_img[aoi_mask]=img[aoi_mask]
        if n_row==1:
            axes[0].imshow(img)
            axes[1].imshow(mask)
            axes[2].imshow(aoi_img)
        else:
            axes[i,0].imshow(img)
            axes[i,1].imshow(mask)
            axes[i,2].imshow(aoi_img)
    plt.show()

#计算pa,mpa,miou,matrix评估指标
def val_score(predict_masks,true_masks,C=2):
    with torch.no_grad():
        predict_masks = predict_masks.permute(0,2,3,1).contiguous().cpu()
        predict_masks = torch.softmax(predict_masks,dim=-1)
        predict_masks = torch.argmax(predict_masks,dim=-1)
        predict_masks = torch.flatten(predict_masks)
        true_masks = torch.flatten(true_masks).cpu()
        true_label = true_masks.unique()
        class_id = true_masks*C+predict_masks
        ids,value=torch.unique(class_id,return_counts=True)

        matrix = torch.flatten(torch.zeros(C,C))
        for i, id in enumerate(ids):
            matrix[id]=value[i]
        matrix = matrix.view(-1,C)
        pa = torch.diag(matrix).sum()/matrix.sum()
        mpa = 0.
        miou = 0.
        for i in range(C):
            mpa = mpa+matrix[i,i]/max(matrix[i].sum(),1)
            miou = miou+matrix[i,i]/max((matrix[i].sum()+matrix[:,i].sum()-matrix[i,i]),1)
        mpa /= max(true_label.size(0),1)
        miou /= max(true_label.size(0),1)
        return pa,mpa,miou,matrix
#混淆矩阵可视化和保存图片

def show_confusion_matrix(matrix_np,class_name):

    fig,axes = plt.subplots(figsize=(10,10))
    #绘制网格
    axes.imshow(matrix_np,cmap=cm.Blues)
    #设置x,y轴标签
    axes.set_xticks(np.arange(matrix_np.shape[1]))
    axes.set_yticks(np.arange(matrix_np.shape[0]))
    #x,y轴标签名称
    axes.set_xticklabels(class_name,rotation=45)
    axes.set_yticklabels(class_name)
    #网格填充数值
    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            axes.text(x=j,y=i,s=f"{matrix_np[i, j]*100:.3f}%",va="center",ha="center",color="red")
    axes.set_title("Confusion Matrix")
    #颜色映射按照混淆矩阵最值
    normalize=Normalize(vmin=matrix_np.min(),vmax=matrix_np.max())
    #可视化
    plt.colorbar(axes.imshow(matrix_np,cmap=cm.Blues,norm=normalize))
    plt.tight_layout()
    fig.savefig(r"val_masks\confusion_matrix.png")
    plt.show()

#图像可视化以及保存前景图片
def save_roi_images(net,images_name_list,image_size=(320,256),is_show=False):
    device="cuda"if torch.cuda.is_available() else"cpu"
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    for n, img_name in enumerate(images_name_list):

        image_pil = Image.open(img_name).convert("RGB")
        image_np = np.array(image_pil)
        albument = get_transform(False, image_size=image_size)(image=image_np)
        image_np = albument["image"]
        image_tensor = tf(image_np).unsqueeze(0)
        predict_mask = net(image_tensor.to(device))
        # 归一化
        predict_mask = torch.softmax(predict_mask, dim=1)
        predict_mask = predict_mask.argmax(dim=1).cpu()
        # 可视化
        if is_show:
            images_show(image_tensor, predict_mask)
        # 前景图片
        roi_image = np.zeros_like(image_np).astype(np.uint8)
        roi_area = predict_mask.squeeze(0).numpy() > 0
        roi_image[roi_area] = image_np[roi_area]
        roi_image = Image.fromarray(roi_image)
        roi_image.save(os.path.join(r"predict", os.path.basename(img_name)))

#视频流原视频帧和抠图前景图片(掩膜)拼接显示
def vedio_concatenate_show(net,video_name,image_size=(320,256),color_list=[[0,0,0],[255,255,255]],is_roi_or_mask=True):
    device="cuda"if torch.cuda.is_available()else"cpu"
    # 定义数据归一标准化模块
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 1, 3)
    # 获取视频对象
    cap = cv2.VideoCapture(video_name)
    while True:
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[0], frame.shape[1]
            albument = get_transform(False, image_size=image_size)(image=frame)
            frame = albument["image"]
            frame = tf(frame).unsqueeze(0).to(device)
            predict = net(frame)
            # 获取预测掩膜布尔张量
            mask = torch.softmax(predict, dim=1).argmax(dim=1).squeeze(0)
            frame = ((frame.squeeze(0).permute(1, 2, 0).contiguous() * std + mean) * 255)
            roi_frame = torch.zeros_like(frame, device=device)
            if is_roi_or_mask:
                mask = mask.type(torch.bool)
                roi_frame[mask] = frame[mask]
            else:
                for label_id in mask.unique():
                    roi_frame[mask==label_id]=torch.tensor(color_list[label_id],device=device).float()

            frame = frame.cpu().numpy().astype(np.uint8)
            roi_frame = roi_frame.type(torch.uint8).cpu().numpy()
            # 将预测前景帧图片尺寸调整到原视频帧尺寸大小
            roi_frame = cv2.resize(roi_frame, (w, h), interpolation=0)
            frame = cv2.resize(frame, (w, h), interpolation=0)
            # 将原视频帧和预测前景帧拼接
            cv2.imshow("image", np.concatenate((frame, roi_frame), axis=1))
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

#定义训练过程中训练验证集损失和mean_iou图表可视化并保存图片
def train_process_show(axes,fig,train_info,val_info,epoch_add):
#loss 和 miou图表可视化
    axes[0].cla()   #清理上一帧画布
    axes[0].plot(list(range(epoch_add)),[n_row[0]for n_row in train_info],label="train")
    axes[0].plot(list(range(epoch_add)),[n_row[0]for n_row in val_info],label="val")
    axes[0].legend()
    axes[0].set_title("aver_loss")
    axes[1].cla()   #清理上一帧画布
    axes[1].plot(list(range(epoch_add)), [n_row[1] for n_row in train_info], label="train")
    axes[1].plot(list(range(epoch_add)), [n_row[1] for n_row in val_info], label="val")
    axes[1].legend()
    axes[1].set_title("miou")

    clear_output(wait=True)
    display(fig)
    plt.pause(0.01)

    plt.ioff()
    fig.savefig(r"val_masks\train_val_plot.png")

if __name__=="__main__":
    pass



