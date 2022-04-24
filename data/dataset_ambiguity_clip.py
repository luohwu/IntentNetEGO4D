from comet_ml import Experiment

import sys
sys.path.insert(0,'..')

import time
from ast import literal_eval
from utils.augmentation import *
import pandas as pd
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from opt import *

import numpy as np
import pickle




def construct_cls(row):

    labels=row['cls'].replace("'", '').strip('][').split(',')
    labels.insert(0,row['class'])
    return labels

def make_sequence_dataset(mode='train',dataset_name='EGO4D'):
    print(f'dataset name: {dataset_name}')
    #val is the same as test
    if mode=='all':
        clip_ids=args.all_clip_ids
    elif mode=='train':
        clip_ids = args.train_clip_ids
    else:
        clip_ids=args.val_clip_ids


    print(f'start load {mode} data, #videos: {len(clip_ids)}')
    df_items = pd.DataFrame()
    for video_id in sorted(clip_ids):
        anno_name = video_id + '.csv'
        anno_path = os.path.join(args.annos_path, anno_name)
        if os.path.exists(anno_path):

            img_path = os.path.join( args.frames_path, video_id)


            annos = pd.read_csv(anno_path
                                ,converters={"nao_bbox": literal_eval,
                                            "objects": literal_eval,
                                            "previous_frames":literal_eval,
                                             "cls":literal_eval,
                                             "ro_bbox":literal_eval,
                                             "noun":literal_eval,
                                             "all_frame":literal_eval}
                                )
            annos['img_path']=img_path
            annos['cls']=annos.apply(lambda row:[args.noun_categories[item] for item in row['cls']],axis=1)

            if not annos.empty:
                annos_subset = annos[['img_path', 'nao_bbox','noun','objects','clip_frame', 'all_frame','ro_bbox','cls']]
                df_items = df_items.append(annos_subset)


    # df_items=df_items.rename(columns={'class':'category'})
    # df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')
    return df_items

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')



class NAODatasetClip(Dataset):
    def __init__(self, mode='train',dataset_name='EGO4D'):
        self.block_len=5 # how many frams in 1 block
        self.num_blocks=5 #5 pre blocks
        self.mode=mode
        self.transform_label = transforms.ToTensor()

        self.data = make_sequence_dataset(mode,dataset_name)
        self.transform_past_frames=transforms.Compose([
            # RandomSizedCrop(size=128, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            Scale(size=(128, 128)),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
            ToTensor(),
            Normalize()
        ])
        self.transform_current_frame = transforms.Compose([  # [h, w]
            # transforms.Resize(args.img_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
            # , AddGaussianNoise(0., 0.5)
        ])

    def __getitem__(self, item):
        # rand_num=torch.rand(1) if self.mode=='train' else 0
        # rand_num=0
        df_item = self.data.iloc[item, :]
        cls=[df_item.noun[0]]+df_item.cls
        nao_bbox = df_item.nao_bbox
        nao_bbox=[456*nao_bbox[0][0],256*nao_bbox[0][1],456*nao_bbox[0][2],256*nao_bbox[0][3]]
        nao_bbox=[round(item) for item in nao_bbox]
        ro_bbox=df_item.ro_bbox
        all_bboxes=[nao_bbox]+ro_bbox
        # print(f'all boxes: {all_bboxes}')
        # print(f'original bbox: {nao_bbox}')

        # path where images are stored
        img_dir = df_item.img_path

        current_frame_path=os.path.join(img_dir,f'frame_{str(df_item.clip_frame).zfill(10)}.jpg')
        frame_indices = df_item.all_frame
        pil_images_list = [pil_loader(os.path.join(df_item.img_path, f'frame_{str(idx).zfill(10)}.jpg')) for idx in
                           frame_indices]
        tensor_images_list = self.transform_past_frames(pil_images_list)
        (C, H, W) = tensor_images_list[0].size()
        past_frames = torch.stack(tensor_images_list, 0)
        del tensor_images_list
        past_frames = past_frames.view(self.num_blocks, self.block_len, C, H, W).transpose(1, 2)

        current_frame_path = os.path.join(img_dir, f'frame_{str(df_item.clip_frame).zfill(10)}.jpg')
        current_frame = Image.open(current_frame_path)
        current_frame= self.transform_current_frame(current_frame)

        # if rand_num>0.5:
        #     current_frame = ImageOps.mirror(current_frame)
        #     temp=nao_bbox[0]
        #     nao_bbox[0]=455-nao_bbox[2]
        #     nao_bbox[2] = 455 - temp

        # print(f'new bbox: {nao_bbox}')

        # print(f'shape of current frame: {current_frame_tensor.shape}')
        mask=generate_mask(256,456,all_bboxes,cls)
        # with open(current_frame_path.replace('.jpg','.npy'),'wb') as f:
        #     np.save(f,mask)
        # with open(current_frame_path.replace('.jpg','.npy'),'rb') as f:
        #     mask=np.load(f)




        return past_frames, current_frame, all_bboxes,current_frame_path,mask,cls

    def __len__(self):
        return self.data.shape[0]

def generate_mask(height,width,bboxes,cls):
    gt=cls[0]
    mask=np.zeros((height,width))
    for i,box in enumerate(bboxes):
        box=[int(item) for item in box]
        area=(box[2]-box[0])*(box[3]-box[1])/(256*456.)
        if area<0.5 or i==0:
            mask[box[1]:box[3],box[0]:box[2]]=1

        # else:
        #     mask[box[1]:box[3], box[0]:box[2]] = 1

    return torch.tensor(mask).float()



def resize_bbox(bbox,height,width,new_height,new_width):
    new_bbox= [bbox[0]/width*new_width,bbox[1]/height*new_height,bbox[2]/width*new_width,bbox[3]/height*new_width]

    new_bbox= [round(coord) for coord in new_bbox]
    new_bbox[0] = new_width if new_bbox[0] > new_width else new_bbox[0]
    new_bbox[2] = new_width if new_bbox[2] > new_width else new_bbox[2]
    new_bbox[1] = new_height if new_bbox[1] > new_height else new_bbox[1]
    new_bbox[3] = new_height if new_bbox[3] > new_height else new_bbox[3]
    return new_bbox

def my_collate(batch):
    past_frames_list=[]
    current_frame_list=[]
    mask_list=[]
    frame_path_list=[]
    bbox_list=[]
    cls_list=[]
    for item in batch:
        past_frames_list.append(item[0])
        current_frame_list.append(item[1])
        bbox_list.append(item[2])
        frame_path_list.append(item[3])
        mask_list.append(item[4])
        cls_list.append(item[5])
    return torch.stack(past_frames_list),torch.stack(current_frame_list),bbox_list,(frame_path_list),torch.stack(mask_list),cls_list

def main_base():
    # train_dataset,val_dataset=ini_datasets(dataset_name='ADL',original_split=False)
    train_dataset = NAODatasetClip(mode='train',dataset_name='EGO4D')
    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/EPIC/test.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                  num_workers=8, shuffle=True,pin_memory=True,
                                  collate_fn=my_collate)
    print(f'start traversing the dataloader')
    start = time.time()
    for epoch in range(1):
        for data in train_dataloader:
            past_frames,current_frame, bboxes, current_frame_path,mask,cls = data
            if 'indument' not in cls[0]:
                continue
            mask=mask[0].numpy()
            # print(f'previous frames shape: {previous_frames.shape}')
            # print(f'current_frame shape: {current_frame.shape}')
            # print(f'sample frame path: {current_frame_path[0]}, nao_bbox: {nao_bbox[0]}')
            print(f'current frame path: {current_frame_path[0]}')
            window_name = current_frame_path[0][-25:]


            """"
            test data and annotations
            need to undo-resize first !!!
            """

            cv2.imwrite('test.jpg', mask)
            cv2_image = cv2.imread('test.jpg')

            bboxes_example = bboxes[0]
            colors=[(255,0,0),(0,255,0),(0,0,255)]
            cnt=0
            for box in bboxes_example:
                print(box)
                cnt=cnt+1
                cv2.rectangle(cv2_image, (box[0], box[1]),
                              (box[2], box[3]), colors[cnt%3], 1 if cnt>1 else 3)
            winname=f'{current_frame_path[0][-10:-4]}'
            cv2.imshow(winname, cv2_image)
            cv2.moveWindow(winname, 700, 100)  # Move it to (40,30)

            cv2.imshow('mask',mask)
            print(cls)



            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                save_path = os.path.join('/media/luohwu/T7/experiments/visualization', window_name.replace('/', '_'))
                cv2.imwrite(
                    filename=save_path,
                    img=cv2_image
                )
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
    train_dataset = NAODatasetClip(mode='test',dataset_name=args.dataset)
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