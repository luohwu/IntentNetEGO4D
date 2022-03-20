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
p_dropout=0.5

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import gensim.downloader

import kornia





class IntentNetBaseWord2Vec(nn.Module):
    def __init__(self):
        super(IntentNetBaseWord2Vec, self).__init__()
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
        self.map=map.to(device)

        self.contribution = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(512,64),
            nn.Dropout(p_dropout),
            nn.ReLU()
        )

    def forward(self,x):
        encoder_features=self.encoder(x)
        feature_last_layer=encoder_features[-1]
        decoder_feature=self.decoder(feature_last_layer,encoder_features[:-1][::-1])
        # decoder_feature=F.softmax(decoder_feature.view(-1,64,116736),dim=2).view_as(decoder_feature)
        decoder_feature=F.layer_norm(decoder_feature,[256,456])
        decoder_feature=decoder_feature*self.map
        # decoder_feature=F.sigmoid(decoder_feature)
        contributions=self.contribution(feature_last_layer)
        # contributions_normalized=torch.softmax(contributions,dim=(1))
        output = self.head((decoder_feature*contributions.view(-1,64,1,1)).sum(1))
        return output,decoder_feature,contributions

glove_vectors = gensim.downloader.load('glove-twitter-25')

class AttentionLossWord2Vec(nn.Module):
    def __init__(self):
        super(AttentionLossWord2Vec, self).__init__()
        self.BCE=nn.BCELoss(reduction='none')

    def forward(self,outputs,targets,labels,bboxes):

        B,H,W=outputs.shape
        weight_masks=torch.ones(B,H,W,requires_grad=False)*0.8
        weight_masks=weight_masks.to(device)
        with torch.no_grad():
            for i in range(B):
                nao_label=labels[i][0]
                ro_bbox = bboxes[i][1:]
                for j,box in enumerate(ro_bbox):
                    # print(box)
                    ro_label=labels[i][j]
                    similarity=compute_similarity(nao_label,ro_label)
                    # print(cos_similarity)
                    weight_masks[i,box[1]:box[3], box[0]:box[2]] += torch.tensor(similarity)/3

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
    from PIL import Image
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    # model = IntentNetBaseAdaptive()
    # model = IntentNetBase()
    model = IntentNetBaseWord2Vec()
    model=model.to(device)
    model_size=sum(p.numel() for p in model.parameters())
    print(f'model size: {model_size}')


    transform = torchvision.transforms.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    if args.euler:
        img_file=os.path.join(args.frames_path,'5f527d09-15e5-46c1-8ca4-949a2ba3d5ff','frame_0000002880.jpg')

    elif args.ait:
        img_file = img_file=os.path.join(args.frames_path,'5f527d09-15e5-46c1-8ca4-949a2ba3d5ff','frame_0000002880.jpg')
    else:
        img_file = '/media/luohwu/T7/dataset/EGO4D/rgb_frames_resized/5f527d09-15e5-46c1-8ca4-949a2ba3d5ff/frame_0000002880.jpg'

    all_bboxes=[
            [165, 5, 210, 84],
            [168, 5, 221, 79],
            [105, 3, 177, 61],
             # [319.6934509277344, 139.83990478515625, 407.7671813964844, 217.42263793945312]
                ]
    labels=[['book', 'book', 'book'],['book', 'book', 'book']]
    # all_bboxes=torch.tensor(all_bboxes,dtype=torch.int)
    input = Image.open(img_file)
    # input=cv2.imread(img_file)
    input = transform(input)
    input=input.to(device)
    loss_fn = AttentionLossWord2Vec()
    # loss_fn=nn.BCELoss()
    target_np = generate_mask(256,456,all_bboxes)
    # cv2.imshow('GT', target_np)
    target = torch.tensor(target_np).float()
    target=target.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=3e-4,
                                  betas=(0.9, 0.99))
    epoch = 500

    #mimic one batch
    input=torch.stack([input,input]).to(device)
    all_bboxes=[all_bboxes,all_bboxes]
    target=torch.stack([target,target]).to(device)

    for i in range(epoch):
        output, decoder_feautre,contributions = model(input)

        for j in range(64):
            original_feature=decoder_feautre[0, j, :, :]
            feature = torch.tanh(original_feature) * 255
            feature_numpy = feature.cpu().detach().numpy()
            if not os.path.exists(f'{args.exp_path}/draft/{i}'):
                os.mkdir(f'{args.exp_path}/draft/{i}')

            saved_path = f'{args.exp_path}/draft/{i}/{j}.jpg'
            cv2.imwrite(saved_path, feature_numpy)
        # weight_numpy=weight.numpy()
        # cv2.imshow('weight',weight_numpy)
        # cv2.waitKey(0)
        loss = loss_fn(output, target,labels,all_bboxes)
        print('='*50)
        print(f'epoch: {i}/1000, loss: {loss.item()}')
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
    if args.euler:
        experiment = Experiment(
            api_key="wU5pp8GwSDAcedNSr68JtvCpk",
            project_name="intentnetego4d",
            workspace="thesisproject",
        )
    main()
