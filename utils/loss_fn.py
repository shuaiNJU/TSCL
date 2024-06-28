'''
my_cl_loss_fn2 function is adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
which is originally licensed under BSD-2-Clause.
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.loss import SoftTargetCrossEntropy
import time


class DisLoss(nn.Module):
    '''
    Dispersion Loss with EMA prototypes
    '''
    def __init__(self, num_classes, model, loader):
        super(DisLoss, self).__init__()
        self.num_classes = num_classes
        self.register_buffer("prototypes", torch.zeros(self.num_classes,128))
        self.model = model
        self.loader = loader
        self.init_class_prototypes()

    def forward(self, features, labels):    

        prototypes = self.prototypes
        for j in range(len(features)):
            prototypes[labels[j].item()] = F.normalize(prototypes[labels[j].item()] *0.5 + features[j]*(1-0.5), dim=0)
        self.prototypes = prototypes.detach()
       
        return self.prototypes

    def init_class_prototypes(self):
        """Initialize class prototypes"""
        self.model.eval()
        start = time.time()
        prototype_counts = [0]*self.num_classes 
        with torch.no_grad():
            prototypes = torch.zeros(self.num_classes,128).cuda()
            for i, (input, target) in enumerate(self.loader):
                input, target = input.cuda(), target.cuda()
                all_f, all_logits, centers = self.model.forward_projection(input) 
                # features = self.model(input)
                for j, feature in enumerate(all_f):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            for cls in range(self.num_classes):
                prototypes[cls] /=  prototype_counts[cls] 
            # measure elapsed time
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes

def my_cl_loss(f_id, f_ood, labels, temperature=0.07, ls=False, tau_list=None, reweighting=False, w_list=None):
    '''
    A variant of supervised contrastive loss: 
    push ID samples from ID samples of different classes;
    push ID samples from OOD samples, but using different push strength according to prior distribution P(y);
    pull ID samples within the same classes.

    Args:
        f_id: features of ID_tail samples. Tensor. Shape=(N_id+N_ood,N_view,d)  尾部ID样本
        f_ood: features of OE samples. Tensor. Shape=(N_ood,d)
        labels: labels of ID_tail samples.  尾部ID样本标签
        ls: Bool. True if do label smoothing on CL loss labels.
        tau_list: list of floats. len=num_classes. Label smoothing parameter for each class based on prior p(y).
    '''
    # print('f_id.shape',f_id.shape)  f_id.shape= torch.Size([N_id, 2, 128])
    f_id = f_id.view(f_id.shape[0], f_id.shape[1], -1) # shape=(N_id,2,d), i.e., 2 views
    # print('f_id.shape',f_id.shape)  f_id.shape= torch.Size([N_id, 2, 128])

    N_id = f_id.shape[0] # N_id = 256
    # print('N_id=',N_id)
    N_ood = f_ood.shape[0]  # 256，   f_ood.shape= torch.Size([256,128])
    labels = labels.contiguous().view(-1, 1)  # labels.shape= torch.Size([N_id, 1]),labels中全是≥5的标签
    # print('labels=',labels)
    
    N_views = f_id.shape[1] # = 2 views
    f_id = torch.cat(torch.unbind(f_id, dim=1), dim=0) # f_id.shape = torch.Size([N_id*2, 128])=(N_id*2,d), f_id表示所有尾部类的样本

    # compute logits:    torch.matmul()就是矩阵相乘
    
    anchor_dot_contrast = torch.div(  # anchor_dot_contrast.shape=torch.Size([2*N_id, 2*N_id+256])
        torch.matmul(f_id, torch.cat((f_id, f_ood), dim=0).T),temperature) 
        # torch.cat((f_id, f_ood), dim=0).shape = torch.Size([2*N_id+256, 128])
    # print('anchor_dot_contrast.shape',anchor_dot_contrast.shape)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # dim=1 is the KL dim.
    # logits_max.shape= torch.Size([2*N_id, 1])
    logits = anchor_dot_contrast - logits_max.detach() # shape=(2N_id,2*N_id+N_ood)
    logits = logits.masked_select(~torch.eye(logits.shape[0], logits.shape[1], dtype=bool).to(logits.device)).view(logits.shape[0], logits.shape[1]-1)  # remove self-contrast cases (diag elements)
    # print('logits.shape=',logits.shape)  logits.shape= torch.Size([2N_id, 2*N_id+N_ood-1])
   
    # labels for CL:
    mask = torch.eq(labels, labels.T).float().to(labels.device) # shape=(N_id,N_id). 1 -> positive pair, 即找出与每个x同属一个类别的所有样本
    mask = mask.repeat(N_views, N_views) # shape=(2*N_id,2*N_id)
    mask = torch.cat((mask, torch.zeros(mask.shape[0],N_ood).to(mask.device)),dim=1) # all ood samples are negative samples to ID samples. shape=(2*N_id,2*N_id+N_ood)
    mask = mask.masked_select(~torch.eye(mask.shape[0], mask.shape[1], dtype=bool).to(mask.device)).view(mask.shape[0], mask.shape[1]-1) # remove self-contrast cases (diag elements). shape=(2*N_id,2*N_id-1+N_ood)
    # 此时的mask去除了原始mask中对角线上的元素，保留了所有剩余元素
    cl_labels = nn.functional.normalize(mask, dim=1, p=1) # so that each row has sum 1, 相当于对每一行对应的x提前计算出了loss均值 
    # cl_labels.shape=(2*N_id,2*N_id-1+N_ood)
    
    # label smoothing:
    if ls:  # false
        for _c, tau in enumerate(tau_list):
            _c_idx = labels == _c
            _c_idx = torch.cat([_c_idx,_c_idx], dim=0).squeeze()
            cl_labels[_c_idx] *= 1 - tau
            cl_labels[_c_idx,2*N_id:] = tau / N_ood

    # loss
    # 因为logits本身就是二维的, 所以F.log_softmax(logits, dim=-1)中dim=-1和dim=1的效果一样，都是对每一行实行softmax
    loss = torch.sum(-cl_labels * F.log_softmax(logits, dim=-1), dim=-1)   # torch.sum(a,dim=-1)即按行求和, loss.shape= torch.Size([2*N_id])
    

    # reweighting: 感觉这一步对loss没有什么影响,因为w=1
    if reweighting:
        assert ls is False
        for _c, w in enumerate(w_list):  # w_list =  tensor([-1., -1., -1., -1., -1., 1.,  1.,  1.,  1.,  1.])
            # print('_c=',_c)
            # print('w=',w)
            # _c 表示索引：0-9； w依次表示w_list中的每个元素
            _c_idx = labels == _c   # k=0.5时，labels中全是≥5的标签
            if torch.sum(_c_idx) > 0:
                assert w > 0, ("Negative loss weight value detected: %s among %s when c=%s among %s" % (w, w_list, _c, torch.unique(labels)))
                _c_idx = torch.cat([_c_idx,_c_idx], dim=0).squeeze()  # 此时_c_idx.shape= torch.Size([num])，元素全是True或者false
                loss[_c_idx] *= w    # w=1所以对loss没啥影响 
    
    # mean over the batch:
    loss = loss.mean() # average among all rows, 即loss_sum/(2*N_id)

    return loss



def prior_to_tau(prior, tau0=0.1):
    '''
    Args:
        prior: iterable with len=num_classes
    
    Returns:
        tau: iterable with len=num_classes
    '''
    tau = tau0 / (prior[0]-prior[-1]) * (prior-prior[-1])
    return tau

def my_cl_loss_fn3(f_id, f_ood, labels, temperature=0.07, ls=False, tau_list=None, reweighting=False, w_list=None):
    '''
    A variant of supervised contrastive loss: 
    push ID samples from ID samples of different classes;
    push ID samples from OOD samples, but using different push strength according to prior distribution P(y);
    pull ID samples within the same classes.

    Args:
        f_id: features of ID_tail samples. Tensor. Shape=(N_id+N_ood,N_view,d)  尾部ID样本
        f_ood: features of OE samples. Tensor. Shape=(N_ood,d)
        labels: labels of ID_tail samples.  尾部ID样本标签
        ls: Bool. True if do label smoothing on CL loss labels.
        tau_list: list of floats. len=num_classes. Label smoothing parameter for each class based on prior p(y).
    '''
    # print('f_id.shape',f_id.shape)  f_id.shape= torch.Size([N_id, 2, 128])
    f_id = f_id.view(f_id.shape[0], f_id.shape[1], -1) # shape=(N_id,2,d), i.e., 2 views
    # print('f_id.shape',f_id.shape)  f_id.shape= torch.Size([N_id, 2, 128])

    N_id = f_id.shape[0] # N_id表示tail_class_ID的样本数量，可能是17或者20这种
    # print('N_id=',N_id)
    N_ood = f_ood.shape[0]  # 256，   f_ood.shape= torch.Size([256,128])
    labels = labels.contiguous().view(-1, 1)  # labels.shape= torch.Size([N_id, 1]),labels中全是≥5的标签
    # print('labels=',labels)
    
    N_views = f_id.shape[1] # = 2 views
    f_id = torch.cat(torch.unbind(f_id, dim=1), dim=0) # f_id.shape = torch.Size([N_id*2, 128])=(N_id*2,d), f_id表示所有尾部类的样本

    # compute logits:    torch.matmul()就是矩阵相乘
    
    anchor_dot_contrast = torch.div(  # anchor_dot_contrast.shape=torch.Size([2*N_id, 2*N_id+256])
        torch.matmul(f_id, torch.cat((f_id, f_ood), dim=0).T),temperature) 
        # torch.cat((f_id, f_ood), dim=0).shape = torch.Size([2*N_id+256, 128])
    # print('anchor_dot_contrast.shape',anchor_dot_contrast.shape)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # dim=1 is the KL dim.
    # logits_max.shape= torch.Size([2*N_id, 1])
    logits = anchor_dot_contrast - logits_max.detach() # shape=(2N_id,2*N_id+N_ood)
    logits = logits.masked_select(~torch.eye(logits.shape[0], logits.shape[1], dtype=bool).to(logits.device)).view(logits.shape[0], logits.shape[1]-1)  # remove self-contrast cases (diag elements)
    # print('logits.shape=',logits.shape)  logits.shape= torch.Size([2N_id, 2*N_id+N_ood-1])
   
    # labels for CL:
    mask = torch.eq(labels, labels.T).float().to(labels.device) # shape=(N_id,N_id). 1 -> positive pair, 即找出与每个x同属一个类别的所有样本
    mask = mask.repeat(N_views, N_views) # shape=(2*N_id,2*N_id)
    mask = torch.cat((mask, torch.zeros(mask.shape[0],N_ood).to(mask.device)),dim=1) # all ood samples are negative samples to ID samples. shape=(2*N_id,2*N_id+N_ood)
    mask = mask.masked_select(~torch.eye(mask.shape[0], mask.shape[1], dtype=bool).to(mask.device)).view(mask.shape[0], mask.shape[1]-1) # remove self-contrast cases (diag elements). shape=(2*N_id,2*N_id-1+N_ood)
    # 此时的mask去除了原始mask中对角线上的元素，保留了所有剩余元素
    cl_labels = nn.functional.normalize(mask, dim=1, p=1) # so that each row has sum 1, 相当于对每一行对应的x提前计算出了loss均值 
    # cl_labels.shape=(2*N_id,2*N_id-1+N_ood)
    
    # label smoothing:
    if ls:  # false
        for _c, tau in enumerate(tau_list):
            _c_idx = labels == _c
            _c_idx = torch.cat([_c_idx,_c_idx], dim=0).squeeze()
            cl_labels[_c_idx] *= 1 - tau
            cl_labels[_c_idx,2*N_id:] = tau / N_ood

    # loss
    # 因为logits本身就是二维的, 所以F.log_softmax(logits, dim=-1)中dim=-1和dim=1的效果一样，都是对每一行实行softmax
    loss = torch.sum(-cl_labels * F.log_softmax(logits, dim=-1), dim=-1)   # torch.sum(a,dim=-1)即按行求和, loss.shape= torch.Size([2*N_id])
    

    # reweighting: 感觉这一步对loss没有什么影响,因为w=1
    if reweighting:
        assert ls is False
        for _c, w in enumerate(w_list):  # w_list =  tensor([-1., -1., -1., -1., -1., 1.,  1.,  1.,  1.,  1.])
            # print('_c=',_c)
            # print('w=',w)
            # _c 表示索引：0-9； w依次表示w_list中的每个元素
            _c_idx = labels == _c   # k=0.5时，labels中全是≥5的标签
            if torch.sum(_c_idx) > 0:
                assert w > 0, ("Negative loss weight value detected: %s among %s when c=%s among %s" % (w, w_list, _c, torch.unique(labels)))
                _c_idx = torch.cat([_c_idx,_c_idx], dim=0).squeeze()  # 此时_c_idx.shape= torch.Size([num])，元素全是True或者false
                loss[_c_idx] *= w    # w=1所以对loss没啥影响 
    
    # mean over the batch:
    loss = loss.mean() # average among all rows, 即loss_sum/(2*N_id)

    return loss

def my_cl_loss_fn(f_id, centers_tail,f_ood, labels, k_tail_labels, temperature=0.07, ls=False, tau_list=None, reweighting=False, w_list=None):
    '''
    A variant of supervised contrastive loss: 
    push ID samples from ID samples of different classes;
    push ID samples from OOD samples, but using different push strength according to prior distribution P(y);
    pull ID samples within the same classes.

    Args:
        f_id: features of ID_tail samples. Tensor. Shape=(N_id+N_ood,N_view,d)  尾部ID样本
        f_ood: features of OE samples. Tensor. Shape=(N_ood,d)
        labels: labels of ID_tail samples.  尾部ID样本标签
        ls: Bool. True if do label smoothing on CL loss labels.
        tau_list: list of floats. len=num_classes. Label smoothing parameter for each class based on prior p(y).
    '''
    # print('f_id.shape',f_id.shape)  f_id.shape= torch.Size([N_id, 2, 128])
    f_id = f_id.view(f_id.shape[0], f_id.shape[1], -1) # shape=(N_id,2,d), i.e., 2 views
    # print('f_id.shape',f_id.shape)  # f_id.shape= torch.Size([N_id, 2, 128])

    N_id = f_id.shape[0] # N_id表示tail_class_ID的样本数量，可能是17或者20这种
    # print('N_id=',N_id)
    N_ood = f_ood.shape[0]  # 256，   f_ood.shape= torch.Size([256,128])
    labels = labels.contiguous().view(-1, 1)  # labels.shape= torch.Size([N_id, 1]),labels中全是≥5的标签
    # print('labels.shape=',labels.shape)
    k_tail_labels = k_tail_labels.contiguous().view(-1, 1)
    # print('k_tail_labels.shape=',k_tail_labels.shape)
    all_labels_tail = torch.cat((labels,labels, k_tail_labels), dim=0)
    # print('all_labels_tail.shape=',all_labels_tail.shape)
    
    N_views = f_id.shape[1] # = 2 views
    f_id = torch.cat(torch.unbind(f_id, dim=1), dim=0) # f_id.shape = torch.Size([N_id*2, 128])=(N_id*2,d), f_id表示所有尾部类的样本
    # print('f_id.shape=',f_id.shape)
    ###### 加入尾部类的类质心  #######
    # f_id = torch.cat((f_id, centers_tail), dim=0)
    # print('f_id.shape=',f_id.shape)
    # compute logits:    torch.matmul()就是矩阵相乘
    
    anchor_dot_contrast = torch.div(  # anchor_dot_contrast.shape=torch.Size([2*N_id+k_tail, 2*N_id+k_tail+256])
        torch.matmul(torch.cat((f_id, centers_tail),dim=0), torch.cat((f_id, centers_tail,f_ood), dim=0).T),temperature) 
        # torch.cat((f_id, f_ood), dim=0).shape = torch.Size([2*N_id+256, 128])
    # print(' torch.cat((f_id, centers_tail), dim=0).shape=', torch.cat((f_id, centers_tail), dim=0).shape)
    # print('anchor_dot_contrast.shape',anchor_dot_contrast.shape)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # dim=1 is the KL dim.
    # logits_max.shape= torch.Size([2*N_id, 1])
    logits = anchor_dot_contrast - logits_max.detach() # shape=(2N_id,2*N_id+N_ood)
    logits = logits.masked_select(~torch.eye(logits.shape[0], logits.shape[1], dtype=bool).to(logits.device)).view(logits.shape[0], logits.shape[1]-1)  # remove self-contrast cases (diag elements)
    # print('logits.shape=',logits.shape)  # logits.shape= torch.Size([2N_id, 2*N_id+N_ood-1])
    # logits.shape= torch.Size([39, 294]

    # labels for CL:
    # all_labels_tail = torch.cat([labels_tail, tail_labels], dim=0)
    # print('all_labels_tail.shape=',all_labels_tail.shape)  # all_labels_tail.shape= torch.Size([29])        
    mask = torch.eq(all_labels_tail, all_labels_tail.T).float().to(labels.device) # shape=(2*N_id+k_tail,2*N_id+k_tail). 1 -> positive pair, 即找出与每个x同属一个类别的所有样本
    # print('mask.shape=',mask.shape)
    
    # mask = mask.repeat(N_views, N_views) # shape=(2*N_id,2*N_id)
    # print('mask.shape=',mask.shape)
    mask = torch.cat((mask, torch.zeros(mask.shape[0],N_ood).to(mask.device)),dim=1) # all ood samples are negative samples to ID samples. shape=(2*N_id,2*N_id+N_ood)
    # print('mask.shape=',mask.shape)
    mask = mask.masked_select(~torch.eye(mask.shape[0], mask.shape[1], dtype=bool).to(mask.device)).view(mask.shape[0], mask.shape[1]-1) # remove self-contrast cases (diag elements). shape=(2*N_id,2*N_id-1+N_ood)
    # print('mask.shape=',mask.shape)
    # 此时的mask去除了原始mask中对角线上的元素，保留了所有剩余元素
    cl_labels = nn.functional.normalize(mask, dim=1, p=1) # so that each row has sum 1, 相当于对每一行对应的x提前计算出了loss均值 
    # print('cl_labels.shape=',cl_labels.shape)  # cl_labels.shape= torch.Size([44, 299])
    # cl_labels.shape=(2*N_id,2*N_id-1+N_ood)
    
    # label smoothing:
    if ls:  # false
        for _c, tau in enumerate(tau_list):
            _c_idx = labels == _c
            _c_idx = torch.cat([_c_idx,_c_idx], dim=0).squeeze()
            cl_labels[_c_idx] *= 1 - tau
            cl_labels[_c_idx,2*N_id:] = tau / N_ood

    # loss
    # 因为logits本身就是二维的, 所以F.log_softmax(logits, dim=-1)中dim=-1和dim=1的效果一样，都是对每一行实行softmax
    loss = torch.sum(-cl_labels * F.log_softmax(logits, dim=-1), dim=-1)   # torch.sum(a,dim=-1)即按行求和, loss.shape= torch.Size([2*N_id])
    # print('loss.shape=',loss.shape)

    # reweighting: 感觉这一步对loss没有什么影响,因为w=1
    if reweighting:
        assert ls is False
        for _c, w in enumerate(w_list):  # w_list =  tensor([-1., -1., -1., -1., -1., 1.,  1.,  1.,  1.,  1.])
            # print('_c=',_c)
            # print('w=',w)
            # _c 表示索引：0-9； w依次表示w_list中的每个元素
            _c_idx = all_labels_tail == _c   # k=0.5时，labels中全是≥5的标签
            # print('_c_idx.shape=',_c_idx.shape)
            if torch.sum(_c_idx) > 0:
                assert w > 0, ("Negative loss weight value detected: %s among %s when c=%s among %s" % (w, w_list, _c, torch.unique(labels)))
                # _c_idx = torch.cat([_c_idx,_c_idx], dim=0).squeeze()  # 此时_c_idx.shape= torch.Size([num])，元素全是True或者false
                _c_idx = _c_idx.squeeze()
                # print('_c_idx.shape=',_c_idx.shape)
                loss[_c_idx] *= w    # w=1所以对loss没啥影响 
    
    # mean over the batch:
    loss = loss.mean() # average among all rows, 即loss_sum/(2*N_id)

    return loss


def oe_loss_fn(logits):
    '''
    The original instable implementation. torch.logsumexp is not numerically stable.
    '''
    return -(logits.mean(1) - torch.logsumexp(logits, dim=1)).mean()  #  logits.mean(1) 按行对每一行求均值

