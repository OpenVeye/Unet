
import torch.cuda
import tqdm
from my_utils import val_score,train_process_show,show_confusion_matrix
from seg_loss_fn import MultiLossFunction
from Unet import myUnet,weights_init
#from Datasets.vesselDataset import * #眼底视网膜血管数据集加载
from Datasets.BloodCeilDataset import * #默认加载红细胞数据集
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
plt.ion()
def train_one_epoch(model,loss_fn,optimizer,dataloader,logger,epoch,max_epoch,C=2):
    model.train()
    device="cuda" if torch.cuda.is_available() else"cpu"
    dataloader_tqdm=tqdm.tqdm(dataloader)
    aver_loss = 0.
    mean_iou = 0.
    accuracy = 0.
    aver_accuracy=0.
    for i, (images_data,masks_data) in enumerate(dataloader_tqdm):
        dataloader_tqdm.set_description(f"training......{epoch+1}/{max_epoch}")
        images_data,masks_data = images_data.to(device),masks_data.to(device)
        predict_masks = model(images_data)
        loss_result  =loss_fn(predict_masks,masks_data)
        optimizer.zero_grad()
        loss_result.backward()
        optimizer.step()
        aver_loss = aver_loss+loss_result.item()
        pa,mpa,miou,matrix=val_score(predict_masks,masks_data,C)
        mean_iou+=miou.item()
        accuracy+=pa.item()
        aver_accuracy+=mpa.item()
        tqdm_dict={
            "iter":i+1,
            "loss":loss_result.item(),
            "accuracy":pa.item(),
            "aver accuracy":mpa.item(),
            "miou":miou.item()
        }
        dataloader_tqdm.set_postfix(tqdm_dict)
        dataloader_tqdm.update()
    len_dataloader = len(dataloader)
    #网页可视化通过tensorboard查看
    aver_loss /=max(1,len_dataloader)
    mean_iou /=max(1,len_dataloader)
    accuracy /=max(1,len_dataloader)
    aver_accuracy /=max(1,len_dataloader)
    logger.add_scalar("train loss",aver_loss,global_step=epoch+1)
    logger.add_scalar("train miou",mean_iou,global_step=epoch+1)
    logger.add_scalar("train accuracy",accuracy,global_step=epoch+1)
    logger.add_scalar("train aver accuracy",aver_accuracy,global_step=epoch+1)
    return aver_loss,mean_iou

def val_one_epoch(model,loss_fn,dataloader,logger,epoch,max_epoch,C=2):

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader_tqdm = tqdm.tqdm(dataloader)
    aver_loss = 0.
    mean_iou = 0.
    accuracy = 0.
    aver_accuracy=0.
    confusion_matrix = torch.zeros(1)
    with torch.no_grad():
        for i, (images_data,masks_data) in enumerate(dataloader_tqdm):
            dataloader_tqdm.set_description(f"testing....{epoch+1}/{max_epoch}")
            images_data,masks_data = images_data.to(device),masks_data.to(device)
            predict_masks = model(images_data) #predict_masks size (bs,n_class,h,w)
            loss_result = loss_fn(predict_masks,masks_data)
            aver_loss = aver_loss+loss_result.item()
            pa,mpa,miou,matrix = val_score(predict_masks,masks_data,C=C)
            mean_iou += miou.item()
            accuracy += pa.item()
            aver_accuracy += mpa.item()
            confusion_matrix = confusion_matrix+matrix
            tqdm_dict = {
            "iter":i+1,
            "loss":loss_result.item(),
            "accuracy":pa.item(),
            "mean accuracy":mpa.item(),
            "miou":miou.item()
        }
            dataloader_tqdm.set_postfix(tqdm_dict)

            dataloader_tqdm.update()
            if (i + 1) % 10==0 or i + 1 == len(dataloader):
                predict_masks = torch.softmax(predict_masks, dim=1)
                predict_masks = torch.argmax(predict_masks, dim=1).float().unsqueeze(1)
                save_image(predict_masks, os.path.join("val_masks", f"{i}.png"))

    aver_loss /= max(1,len(dataloader))
    mean_iou /= max(1,len(dataloader))
    accuracy /= max(1,len(dataloader))
    aver_accuracy /= max(1,len(dataloader))
    logger.add_scalar("val loss",aver_loss,global_step=epoch+1)
    logger.add_scalar("val miou",mean_iou,global_step=epoch+1)
    logger.add_scalar("val accuracy",accuracy,global_step=epoch+1)
    logger.add_scalar("val aver_accuracy",aver_accuracy,global_step=epoch+1)
    #计算混淆矩阵归一化
    confusion_matrix=confusion_matrix/confusion_matrix.sum()
    return  aver_loss,mean_iou,confusion_matrix.numpy()

def main():
    #新建日志、模型权重等相关存放文件夹
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists("model"):
        os.mkdir("model")
    if not os.path.exists("val_masks"):
        os.mkdir("val_masks")
    #训练和验证数据集
    image_size=(320,320)
    n_workers = 0
    batch_size=1
    train_dataloader = get_dataloader(image_size,True,batch_size,num_workers=n_workers)
    val_dataloader = get_dataloader(image_size,False,batch_size,num_workers=n_workers)
    #建立网络
    device = "cuda"if torch.cuda.is_available() else "cpu"
    C=2
    is_bb_pretrained=False
    net = myUnet(C,is_bb_pretrained).to(device)
    is_pretrained = False
    if is_pretrained:
        weight_file = r"model\best_unet.pth"
        if os.path.exists(weight_file):
            weight_dict = torch.load(weight_file)
            net_dict = net.state_dict()
            temp_dict = {k: v for k, v in weight_dict.items() if
                         k in net_dict.keys() and net_dict[k].numel() == v.numel()}
            for k, v in temp_dict.items():
                net_dict[k] = v
            net.load_state_dict(net_dict)
            print(f"权重文件存在，模型成功导入参数")
    else:
        if not is_bb_pretrained:
            weights_init(net)
    #优化器
    lr = 0.0001
    optimizer= torch.optim.Adam(net.parameters(),lr=lr,betas=(0.9,0.99),weight_decay=5.e-4)
    #损失函数
    loss_function=MultiLossFunction()
    #tensorboard 日志对象
    logger = SummaryWriter("log")
    train_info = []
    val_info=[]
    best_loss =np.inf
    best_miou=0.
    best_matrix=0.
    max_epoch=60
    # class_name=["background","vessel"] #眼底视网膜血管数据集类别
    class_name=["background","ceil"] #默认红细胞类别
    #动态显示训练图表
    fig, axes = plt.subplots(1, 2)

    for epoch in range(max_epoch):
        if epoch==20:
            lr = 0.00001
        elif epoch==40:
            lr = 0.000001
        for param in optimizer.param_groups:
            param["lr"]=lr
        aver_loss,miou = train_one_epoch(net,loss_function,optimizer,train_dataloader,logger,epoch,max_epoch,C)
        train_info.append([aver_loss,miou])
        aver_loss,miou,c_matrix=val_one_epoch(net,loss_function,val_dataloader,logger,epoch,max_epoch,C)
        val_info.append([aver_loss,miou])
        if aver_loss<best_loss or miou>best_miou:
            if aver_loss<best_loss:
                best_loss=aver_loss
            if miou>best_miou:
                best_miou=miou
                best_matrix = c_matrix
                torch.save(net.state_dict(),r"model\best_unet.pth",_use_new_zipfile_serialization=False)
        train_process_show(axes,fig,train_info,val_info,epoch+1)

    logger.close()
    # 混淆矩阵可视化
    show_confusion_matrix(best_matrix, class_name=class_name)

if __name__=="__main__":
    main()