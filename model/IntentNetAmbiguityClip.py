import os

from comet_ml import Experiment

import sys
sys.path.insert(0,'..')

import cv2
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from opt import *
# from model.IntentNetAmbiguity  import *
from backbone.DPC import DPC_RNN

p_dropout=0.2


import gensim.downloader

import kornia


def load_backbone_state(model):
    state_dict_DataParallel=torch.load(os.path.join(args.exp_path,'EGO4D/SSL/ckpts/model_epoch_100.pth'),map_location='cpu')['model_state_dict']
    from collections import OrderedDict
    state_dict_normal=OrderedDict()
    for k,v in state_dict_DataParallel.items():
        key=k[7:]
        state_dict_normal[key]=v
    model.load_state_dict(state_dict_normal)
    return model

class ConvBlock(nn.Module):
    def __init__(self,in_c,out_c,padding=1):
        super(ConvBlock, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3,padding=padding),
            # nn.BatchNorm2d(out_c),
            nn.Dropout2d(p_dropout),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,kernel_size=3,padding=padding),
            nn.BatchNorm2d(out_c),
        )

    def forward(self,x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,channels=[256,128, 64, 32]):
        super(Decoder, self).__init__()
        self.channels=channels
        self.up_block_list=nn.ModuleList([nn.ConvTranspose2d(256,128,kernel_size=(4,4),stride=(4,4),padding=(0,1)),
                                          nn.ConvTranspose2d(128,64,kernel_size=(4,4),stride=(4,4),padding=(0,2)),
                                          nn.ConvTranspose2d(64,32,kernel_size=(2,4),stride=(2,4),padding=(0,4))
                                          ])
        self.conv_block_list=nn.ModuleList([
            ConvBlock(self.channels[i],self.channels[i]) for i in range(len(channels))
        ])
        self.dropout=nn.Dropout2d(p_dropout)

    def forward(self,x):
        for i in range(len(self.channels)-1):
            x = self.conv_block_list[i](x)
            x = self.dropout(x)
            x=  self.up_block_list[i](x)
        x=self.conv_block_list[-1](x)
        return x




class IntentNetClipWord2VecSoftmax(nn.Module):
    def __init__(self):
        super(IntentNetClipWord2VecSoftmax, self).__init__()
        self.backbone=DPC_RNN(sample_size=128,num_seq=5,seq_len=5,pred_step=0,network='resnet18')
        self.backbone=load_backbone_state(self.backbone)
        self.decoder=Decoder(channels=[256,128,64,32])
        self.compute_contribution=nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32),
        )
        self.head=nn.Conv2d(32,2,kernel_size=1)
        self._initialize_weights(self.decoder)



    def forward(self,x):
        clip_context=self.backbone(x)
        clip_context_pooled=F.adaptive_avg_pool2d(clip_context,output_size=(1,1)).squeeze(-1).squeeze(-1)
        contribution=self.compute_contribution(clip_context_pooled) # B,C
        decoder_feature=self.decoder(clip_context)
        binary_feature=self.head(decoder_feature)
        p=torch.softmax(binary_feature,dim=1)
        return p[:,0].squeeze(1)


    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and param.ndimension()>1:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

class IntentNetClipWord2Vec(nn.Module):
    def __init__(self):
        super(IntentNetClipWord2Vec, self).__init__()
        self.backbone=DPC_RNN(sample_size=128,num_seq=5,seq_len=5,pred_step=0,network='resnet18')
        self.backbone=load_backbone_state(self.backbone)
        self.decoder=Decoder(channels=[256,128,64,32])
        self.compute_contribution=nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,32),
        )
        self._initialize_weights(self.decoder)



    def forward(self,x):
        clip_context=self.backbone(x)
        clip_context_pooled=F.adaptive_avg_pool2d(clip_context,output_size=(1,1)).squeeze(-1).squeeze(-1)
        contribution=self.compute_contribution(clip_context_pooled) # B,C
        decoder_feature=self.decoder(clip_context)
        output = (decoder_feature*contribution.view(-1,32,1,1)).sum(1)
        return F.sigmoid(output)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and param.ndimension()>1:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself


glove_vectors = gensim.downloader.load('glove-twitter-25')

class AttentionLossWord2Vec(nn.Module):
    def __init__(self):
        super(AttentionLossWord2Vec, self).__init__()
        self.BCE=nn.BCELoss(reduction='none')

    def forward(self,outputs,targets,labels,bboxes):

        B,H,W=outputs.shape
        weight_masks=torch.ones(B,H,W,requires_grad=False)*0.8
        weight_masks=weight_masks.to(outputs.device)
        with torch.no_grad():
            for i in range(B):
                nao_label=labels[i][0]
                ro_bbox = bboxes[i][1:]
                for j,box in enumerate(ro_bbox):
                    # print(box)
                    ro_label=labels[i][j]
                    similarity=compute_similarity(nao_label,ro_label)
                    # print(cos_similarity)
                    # weight_masks[i,box[1]:box[3], box[0]:box[2]] += torch.tensor(similarity)
                    weight_masks[i, box[1]:box[3], box[0]:box[2]] += torch.tensor(similarity) / 3.

        pixel_wise_loss=self.BCE(outputs,targets)*weight_masks
        return pixel_wise_loss.mean()

def calibrate_label(label):
    if label=='indument':
        return 'rag'
    return label

def compute_similarity(nao_label,ro_label):
    # print(f'orginal_nao: {nao_label}, calibrated: {calibrate_label(nao_label)}')
    # print(f'orginal_ro: {ro_label}, calibrated: {calibrate_label(ro_label)}')
    return glove_vectors.similarity(calibrate_label(nao_label),calibrate_label(ro_label))





def main():
    from data.dataset_ambiguity_clip import NAODatasetClip

    SEED = 3080
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    from PIL import Image
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.data import DataLoader
    from data.dataset_ambiguity_clip import my_collate
    model = IntentNetClipWord2Vec()
    model = nn.DataParallel(model)
    model=model.to(device)
    model_size=sum(p.numel() for p in model.parameters())
    print(f'model size: {model_size}')
    train_dataset = NAODatasetClip(mode='train', dataset_name=args.dataset)
    indices = torch.randperm(len(train_dataset))[:32].tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices)
    num_workers=4
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True,
                                  # drop_last=True if torch.cuda.device_count() >=4 else False,
                                  collate_fn=my_collate)
    # item=next(iter(train_dataloader))


    # input=input.to(device)
    loss_fn = AttentionLossWord2Vec()
    # loss_fn=nn.BCELoss()
    # cv2.imshow('GT', target_np)
    # target=target.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=3e-3,
                                  betas=(0.9, 0.99))
    epoch = 500

    #mimic one batch
    # input=torch.stack([input,input]).to(device)
    # all_bboxes=[all_bboxes,all_bboxes]
    # target=torch.stack([target,target]).to(device)

    for i in range(epoch):
        for item in train_dataloader:
            input, all_bboxes, current_path, target, labels = item
            input=input.to(device)
            target=target.to(device)
            output= model(input)


            # weight_numpy=weight.numpy()
            # cv2.imshow('weight',weight_numpy)
            # cv2.waitKey(0)
            loss = loss_fn(output, target,labels,all_bboxes)
            print('='*50)
            print(f'epoch: {i}/{epoch}, loss: {loss.item()}')
            experiment.log_metrics({"train_loss": loss.item()}, step=i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output_numpy = output[0].cpu().detach().numpy()
            im_path=os.path.join(args.exp_path,'draft',f'output_{i}.jpg')
            cv2.imwrite(im_path, output_numpy * 255)
        # cv2.waitKey(0)
    # cv2.imshow('output', output_numpy)
    # cv2.waitKey(0)

    # print(input)
    # input=input.permute(1,2,0)
    # print(input)
    # plt.imshow(input)
    # cv2.imshow('test',input)
    # cv2.waitKey(0)



if __name__=='__main__':
    if args.ait:
        experiment = Experiment(
            api_key="wU5pp8GwSDAcedNSr68JtvCpk",
            project_name="intentnetego4d_clip",
            workspace="thesisproject",
        )
    main()
