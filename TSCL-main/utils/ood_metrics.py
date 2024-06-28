import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np 
# import sklearn.covariance
# from advertorch.utils import clamp


def get_msp_scores(model, images):

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    msp = probs.max(dim=1).values

    scores = - msp # The larger MSP, the smaller uncertainty
    # scores = msp  # 我去掉了负号
    return logits, scores

def get_msp_scores_ad(model, images):

    logits = model(images)
    probs = F.softmax(logits, dim=1)[:,:10]
    msp = probs.max(dim=1).values

    scores = - msp # The larger MSP, the smaller uncertainty
    # scores = msp  # 我去掉了负号
    return logits, scores

def get_ood_score(model,inputs):
    with torch.no_grad():
        outputs = model(inputs)

    # scores = -1.0 * (F.softmax(outputs, dim=1)[:,-1]).float()
    scores = 1.0 * (F.softmax(outputs, dim=1)[:,-1]).float()
    return outputs,scores

def get_sofl_scores(model, images,num_classes):
    with torch.no_grad():
        logits = model(images)
    scores = F.softmax(logits, dim=1)[:, num_classes:].sum(dim=1)  # 不加负号

    return logits, scores
