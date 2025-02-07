import torch
import torch.nn as nn
#多分类FocalLoss
class MultiFocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2,ignore_id=-1):
        super().__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.ignore_id = ignore_id
    def forward(self,predict,target):
        predict = predict.permute(0,2,3,1).contiguous()
        predict = torch.softmax(predict,dim=-1)
        b,c = predict.size(0),predict.size(3)
        mask = target!=self.ignore_id #size(b,h,w)
        predict = predict[mask].view(-1,c)
        target = target[mask].view(-1)
        one_hot = torch.eye(c,device=predict.device)
        target = one_hot[target].view(-1,c).float()
        f_loss = (-self.alpha*((1-predict)**self.gamma)*target*torch.log(predict+1.e-8)).sum(dim=-1)*b

        return f_loss.mean()

#多分类DiceLoss
class MultiDiceLoss(nn.Module):
    def __init__(self,ignore_id = -1):
        super().__init__()
        self.ignore_id = ignore_id
    def forward(self,predict,target):
        predict = predict.permute(0,2,3,1).contiguous()
        predict = torch.softmax(predict,dim=-1)
        mask = target!=self.ignore_id
        c = predict.size(3)
        predict = predict[mask].view(-1,c)
        target=target[mask].view(-1)
        one_hot = torch.eye(c,device=predict.device)
        target = one_hot[target].view(-1,c).float()
        dice_score = (2*predict*target).sum(0)/(predict+target+1.e-8).sum(0).mean()
        d_loss = 1.-dice_score
        return d_loss

#语义分割损失函数
class MultiLossFunction(nn.Module):
    def __init__(self,alpha=0.25,gamma=2,ignore_id=-1,type="CE"):   #CrossEntropyLoss "CE", FocalLoss "FL",DiceLoss "DL"
        super().__init__()
        if type=="CE":
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_id)
        elif type=="FL":
            self.loss_fn=MultiFocalLoss(alpha,gamma,ignore_id)
        else:
            self.loss_fn=MultiDiceLoss(ignore_id)
    def forward(self,predict,target):
        return self.loss_fn(predict,target)