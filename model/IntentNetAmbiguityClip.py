import os

from comet_ml import Experiment

import sys
sys.path.insert(0,'..')

import cv2
import torch
import torchvision.transforms.transforms
from torch import nn
from torchvision import models
import torch.nn.functional as F
from opt import *
from model.IntentNetAmbiguity  import *
from data.dataset_ambiguity import generate_mask
p_dropout=0.2


import gensim.downloader

import kornia
from backbone.DPC import DPC_RNN



def load_backbone_state(model):
    state_dict_DataParallel=torch.load(os.path.join(args.exp_path,'EGO4D/SSL/ckpts/model_epoch_100.pth'),map_location='cpu')['model_state_dict']
    from collections import OrderedDict
    state_dict_normal=OrderedDict()
    for k,v in state_dict_DataParallel.items():
        key=k[7:]
        state_dict_normal[key]=v
    model.load_state_dict(state_dict_normal)
    return model

class IntentNetClipWord2Vec(nn.Module):
    def __init__(self):
        super(IntentNetClipWord2Vec, self).__init__()
        self.encoder=Encoder(in_channel=3,out_channels=[64,128,256,512])
        self.decoder=Decoder(in_channel=512,out_channels=[256,128,64])
        self.head=nn.Sequential(
            nn.Sigmoid(),
            # nn.Tanh(),
        )
        map=torch.ones(1,64,256,456)*0.001
        for i in range(8):
            for j in range(8):
                map[0,i+j*8,i*32:(i+1)*32,j*57:(j+1)*57]=50
        map=kornia.filters.gaussian_blur2d(map,(171,171),(100.5,100.5))
        for i in range(64):
            cv2.imwrite(f'{args.exp_path}/draft/0_map/{i}.jpg',map[0,i].numpy()*255)
        # self.map=map.to(device)
        self.map=map

        self.contribution = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # self.contribution_old = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(1),
        #     nn.Linear(512,64),
        #     # nn.Dropout(p_dropout),
        #     nn.ReLU(),
        #     nn.Linear(64,64),
        #     nn.ReLU(),
        # )

        self.backbone=DPC_RNN(sample_size=128,num_seq=5,seq_len=5,pred_step=0,network='resnet18')
        self.backbone=load_backbone_state(self.backbone)
        # self.backbone=nn.DataParallel(self.backbone).to(device)
        # for param in self.backbone.module.parameters():
        #     param.requires_grad=False

    def forward(self,x,past_frames):
        encoder_features=self.encoder(x)
        feature_last_layer=encoder_features[-1]
        decoder_feature=self.decoder(feature_last_layer,encoder_features[:-1][::-1])
        # decoder_feature=F.softmax(decoder_feature.view(-1,64,116736),dim=2).view_as(decoder_feature)
        decoder_feature=F.layer_norm(decoder_feature,[256,456])
        decoder_feature=decoder_feature*self.map.to(decoder_feature.device)
        # decoder_feature=F.sigmoid(decoder_feature)
        context=self.backbone(past_frames)
        contributions=self.contribution(context)

        # contributions = self.contribution_old(feature_last_layer)

        # contributions_normalized=torch.softmax(contributions,dim=(1))
        output = self.head((decoder_feature*contributions.view(-1,64,1,1)).sum(1))
        return output,decoder_feature,contributions


glove_vectors = gensim.downloader.load('glove-twitter-25')
import numpy as np
class AttentionLossWord2Vec(nn.Module):
    def __init__(self):
        super(AttentionLossWord2Vec, self).__init__()
        self.BCE=nn.BCELoss(reduction='none')

    def forward(self,outputs,targets,labels,bboxes):

        B,H,W=outputs.shape
        weight_masks=torch.zeros(B,H,W,requires_grad=False)*0.8
        weight_masks=weight_masks.to(outputs.device)
        with torch.no_grad():
            for i in range(B):
                nao_label=labels[i][0]
                ro_bbox = bboxes[i][:]
                for j,box in enumerate(ro_bbox):
                    # print(box)
                    ro_label=labels[i][j]
                    similarity=compute_similarity(nao_label,ro_label)
                    # print(cos_similarity)
                    # weight_masks[i,box[1]:box[3], box[0]:box[2]] += torch.tensor(similarity)/3

                    if j==0:
                        weight_masks[i, box[1]:box[3], box[0]:box[2]] =1
                    else:

                        weight_masks[i, box[1]:box[3], box[0]:box[2]] = torch.max(
                            weight_masks[i, box[1]:box[3], box[0]:box[2]],
                            similarity * torch.ones_like(weight_masks[i, box[1]:box[3], box[0]:box[2]]))
            weight_masks[weight_masks==0]=0.8

        pixel_wise_loss=self.BCE(outputs,targets)*weight_masks
        # weight_masks=weight_masks[0]*255
        # weight_masks = weight_masks.cpu().detach().numpy().astype(np.uint8)
        # weight_masks=cv2.applyColorMap(weight_masks,cv2.COLORMAP_JET)
        # cv2.imwrite('/data/luohwu/experiments/EGO4D/clip/weight.png',weight_masks)
        # targets=targets[0].cpu().detach().numpy()*255
        #
        # cv2.imwrite('/data/luohwu/experiments/EGO4D/clip/targets.png',targets)
        return pixel_wise_loss.mean()

def calibrate_label(label):
    if label=='indument':
        return 'rag'
    return label

def compute_similarity(nao_label,ro_label):
    # print(f'orginal_nao: {nao_label}, calibrated: {calibrate_label(nao_label)}')
    # print(f'orginal_ro: {ro_label}, calibrated: {calibrate_label(ro_label)}')
    return glove_vectors.similarity(calibrate_label(nao_label),calibrate_label(ro_label))



def main_simple():
    model=IntentNetClipWord2Vec()

    model=nn.DataParallel(model).to(device)
    # for param in model.module.parameters():
    #     if param in model.module.backbone.parameters():
    #         print('yes')
    #     else:
    #         print('no')
    past_frames=torch.randn(4,5,3,5,128,128).to(device)
    current_frame=torch.randn(4,3,256,456).to(past_frames.device)
    output,decoder_feature,contributions=model(x=current_frame,past_frames=past_frames)
    print(output.shape)



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
            past_frames,current_frame, all_bboxes, current_path, target, labels = item
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
    main_simple()
