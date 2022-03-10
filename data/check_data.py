import os.path

from comet_ml import Experiment

import sys
sys.path.insert(0,'..')

import time
from ast import literal_eval

import pandas as pd
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from opt import *

import numpy as np
import pickle


# read a list from a file line by line
def read_clips(file_path):
    file = open(file_path, 'r')
    file_lines=file.read()
    list=file_lines.split('\n')[:-1]
    return list








def make_sequence_dataset(mode='train',dataset_name='ADL'):
    print(f'dataset name: {dataset_name}')
    #val is the same as test
    if mode=='all':
        clip_list=all_clip_ids
    elif mode=='train':
        clip_list = train_clip_ids
    else:
        clip_list=val_clip_ids


    print(f'start load {mode} data, #videos: {len(clip_list)}')
    df_items = pd.DataFrame()
    for clip_id in sorted(clip_list):
        anno_name =  clip_id + '.csv'
        anno_path = os.path.join(args.annos_path, anno_name)
        if os.path.exists(anno_path):
            # start = time.process_time()
            img_path = os.path.join(args.frames_path,clip_id)

            annos = pd.read_csv(anno_path,
                                # converters={"nao_bbox": literal_eval,
                                #             "ro_bbox":literal_eval,
                                #             # "nao_bbox_resized": literal_eval,
                                #             "previous_frames":literal_eval,
                                #             # "cls":literal_eval
                                #             }
                                )
            annos['img_path']=img_path

            if not annos.empty:
                annos_subset = annos[['img_path','clip_frame','nao_bbox','class']]
                df_items = df_items.append(annos_subset)



    # df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')
    return df_items





class NAODatasetBase(Dataset):
    def __init__(self, mode='train',dataset_name='ADL'):
        self.mode=mode
        self.transform_label = transforms.ToTensor()

        self.data = make_sequence_dataset(mode,dataset_name)
        # self.data = data
        # self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([  # [h, w]
            # transforms.Resize(args.img_resize),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])  # ImageNet
            # , AddGaussianNoise(0., 0.5)
        ])
        self.transform_test = transforms.Compose([  # [h, w]
            # transforms.Resize(args.img_resize),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])  # ImageNet
        ])

        self.transform_previous_frames = transforms.Compose([  # [h, w]
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])  # ImageNet
            # , AddGaussianNoise(0., 0.5)
        ])
        self.transform_previous_frames_test = transforms.Compose([  # [h, w]
            # transforms.Resize((112,112)) if args.C3D else transforms.Resize((224,224))  ,
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])  # ImageNet
        ])

    def __getitem__(self, item):
        # rand_num=torch.rand(1) if self.mode=='train' else 0
        # rand_num=0
        df_item = self.data.iloc[item, :]
        nao_bbox = literal_eval(df_item.nao_bbox)
        img_path = df_item['img_path']
        frame=df_item['clip_frame']
        img_path=os.path.join(img_path,f'frame_{str(frame).zfill(10)}.jpg')
        categories=df_item['class']


        return img_path,nao_bbox,categories

    def __len__(self):
        return self.data.shape[0]

def generate_mask(height,width,bboxes):
    mask=np.zeros((height,width))
    for box in bboxes:
        box=[int(item) for item in box]
        mask[box[1]:box[3],box[0]:box[2]]=1
    return torch.tensor(mask).float()


def my_collate(batch):
    frames_list=[]
    bbox_list=[]
    categories_list=[]
    for item in batch:
        frames_list.append(item[0])
        bbox_list.append(item[1])
        categories_list.append(item[2])
    return frames_list,bbox_list,categories_list

def main_base():
    # train_dataset,val_dataset=ini_datasets(dataset_name='ADL',original_split=False)
    train_dataset = NAODatasetBase(mode='train',dataset_name='EGO4D')
    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/EPIC/test.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                  num_workers=8, shuffle=True,pin_memory=True,
                                  collate_fn=my_collate
                                  )
    print(f'start traversing the dataloader')
    start = time.time()
    for epoch in range(1):
        for data in train_dataloader:
            img_path,nao_bbox,categories=data
            img_path=img_path[0]
            nao_bbox=nao_bbox[0]
            categories=literal_eval(categories[0])
            image=cv2.imread(img_path)
            image_resized=cv2.resize(image,(456,256))
            height, width, channels = image.shape
            for i,bbox in enumerate(nao_bbox):
                bbox=[bbox[0]*width,bbox[1]*height,bbox[2]*width,bbox[3]*height]
                bbox=[int(a) for a in bbox]
                cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color=(255,0,0),thickness=2)
                cv2.putText(image,categories[i],(bbox[0],bbox[1]+20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(0,0,255),
                            thickness=2)
            cv2.imshow('original_image',image)

            height, width, channels = image_resized.shape
            for i,bbox in enumerate(nao_bbox):
                bbox=[bbox[0]*width,bbox[1]*height,bbox[2]*width,bbox[3]*height]
                bbox=[int(a) for a in bbox]
                cv2.rectangle(image_resized,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color=(255,0,0),thickness=2)
                cv2.putText(image_resized,categories[i],(bbox[0],bbox[1]+20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(0,0,255),
                            thickness=2)
            cv2.imshow('resized_image',image_resized)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                cv2.destroyAllWindows()
            elif key == ord('q'):
                cv2.destroyAllWindows()
                break
            else:
                cv2.destroyAllWindows()
                continue


    end = time.time()
    print(f'used time: {end-start}')



def check_acc():
    # train_dataset,val_dataset=ini_datasets(dataset_name='ADL',original_split=False)
    train_dataset = NAODatasetBase(mode='test',dataset_name=args.dataset)
    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/EPIC/test.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                                  num_workers=8, shuffle=True,pin_memory=True,collate_fn=my_collate)
    acc=0
    for data in train_dataloader:
        frames, bboxes, current_frame_path,mask = data
        bboxes=bboxes[0]
        if len(bboxes)>1:
            nao_bbox = bboxes[0]
            ro_bboxes = bboxes[1:]
            for ro_bbox in ro_bboxes:
                iou = compute_iou(nao_bbox, ro_bbox)
                if iou > 0.5:
                    acc = acc + 1
                    break
    print(f'acc: {acc}/{len(train_dataset)}')


def compute_iou(bbox1,bbox2):
    area1=(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2=(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    inter_l=max(bbox1[0],bbox2[0])
    inter_r=min(bbox1[2],bbox2[2])
    inter_t=max(bbox1[1],bbox2[1])
    inter_b=min(bbox1[3],bbox2[3])
    inter_area = max((inter_r - inter_l),0) * max((inter_b - inter_t),0)
    return inter_area/(area1+area2-inter_area)




if __name__ == '__main__':
    # experiment = Experiment(
    #     api_key="wU5pp8GwSDAcedNSr68JtvCpk",
    #     project_name="intentnetambiguity",
    #     workspace="thesisproject",
    # )
    main_base()
    # check_acc()