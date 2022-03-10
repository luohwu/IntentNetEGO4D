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
                                             "noun":literal_eval}
                                )
            annos['img_path']=img_path
            annos['cls']=annos.apply(lambda row:[args.noun_categories[item] for item in row['cls']],axis=1)

            if not annos.empty:
                annos_subset = annos[['img_path', 'nao_bbox','noun','objects', 'clip_frame','ro_bbox','cls']]
                df_items = df_items.append(annos_subset)


    # df_items=df_items.rename(columns={'class':'category'})
    # df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')
    return df_items





class NAODatasetBase(Dataset):
    def __init__(self, mode='train',dataset_name='EGO4D'):
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
        current_frame=Image.open(current_frame_path)
        # if rand_num>0.5:
        #     current_frame = ImageOps.mirror(current_frame)
        #     temp=nao_bbox[0]
        #     nao_bbox[0]=455-nao_bbox[2]
        #     nao_bbox[2] = 455 - temp

        # print(f'new bbox: {nao_bbox}')

        current_frame_tensor=self.transform(current_frame)
        # print(f'shape of current frame: {current_frame_tensor.shape}')
        mask=generate_mask(256,456,all_bboxes)
        # with open(current_frame_path.replace('.jpg','.npy'),'wb') as f:
        #     np.save(f,mask)
        # with open(current_frame_path.replace('.jpg','.npy'),'rb') as f:
        #     mask=np.load(f)




        return current_frame_tensor, all_bboxes,current_frame_path,mask,cls

    def __len__(self):
        return self.data.shape[0]

def generate_mask(height,width,bboxes):
    mask=np.zeros((height,width))
    for box in bboxes:
        box=[int(item) for item in box]
        mask[box[1]:box[3],box[0]:box[2]]=1
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
    frames_list=[]
    mask_list=[]
    frame_path_list=[]
    bbox_list=[]
    cls_list=[]
    for item in batch:
        frames_list.append(item[0])
        bbox_list.append(item[1])    
        frame_path_list.append(item[2])
        mask_list.append(item[3])
        cls_list.append(item[4])
    return torch.stack(frames_list),bbox_list,(frame_path_list),torch.stack(mask_list),cls_list

def main_base():
    # train_dataset,val_dataset=ini_datasets(dataset_name='ADL',original_split=False)
    train_dataset = NAODatasetBase(mode='test',dataset_name='EGO4D')
    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/EPIC/test.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                  num_workers=8, shuffle=True,pin_memory=True,
                                  collate_fn=my_collate)
    print(f'start traversing the dataloader')
    start = time.time()
    for epoch in range(1):
        for data in train_dataloader:
            frames, bboxes, current_frame_path,mask,cls = data
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
            current_frame_example = frames[0].permute(1, 2, 0).numpy()
            current_frame_example *= 255
            cv2.imwrite('test.jpg', current_frame_example)
            cv2_image = cv2.imread('test.jpg')
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

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


def main():
    # train_dataset,val_dataset=ini_datasets(dataset_name='ADL',original_split=False)
    train_dataset = NAODatasetBase(mode='train',dataset_name=args.dataset)
    print(train_dataset.data.head())
    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/EPIC/test.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                  num_workers=8, shuffle=True,pin_memory=True)
    print(f'start traversing the dataloader')
    start = time.time()
    for epoch in range(5):

        for data in train_dataloader:
            frames, nao_bbox, current_frame_path = data
            # print(f'previous frames shape: {previous_frames.shape}')
            # print(f'current_frame shape: {current_frame.shape}')
            # print(f'sample frame path: {current_frame_path[0]}, nao_bbox: {nao_bbox[0]}')
            print(f'current frame path: {current_frame_path[0]}')
            window_name = current_frame_path[0][-25:]

            nao_bbox_shape = nao_bbox.shape

            """"
            test data and annotations
            need to undo-resize first !!!
            """
            current_frame_example = frames[0, -1].permute(1, 2, 0).numpy()
            current_frame_example *= 255
            cv2.imwrite('test.jpg', current_frame_example)
            cv2_image = cv2.imread('test.jpg')
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            nao_bbox_example = nao_bbox[0].numpy()

            # image_224 = cv2.resize(cv2_image,(224,224))
            # image_224 = cv2.imread(current_frame_path.replace('rgb_frames','rgb_frames_resized'))
            # nao_bbox_resized=resize_bbox(nao_bbox_example,256,456,224,224)
            # cv2.rectangle(image_224, (nao_bbox_resized[0], nao_bbox_resized[1]),
            #               (nao_bbox_resized[2], nao_bbox_resized[3]), (255, 0, 0), 3)

            cv2.rectangle(cv2_image, (nao_bbox_example[0], nao_bbox_example[1]),
                          (nao_bbox_example[2], nao_bbox_example[3]), (255, 0, 0), 3)
            winname=f'{current_frame_path[0][-10:-4]}'
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname, 700, 100)  # Move it to (40,30)
            cv2.imshow(winname, cv2_image)
            # cv2.imshow(f'{current_frame_path[0][-10:-4]}_2', image_224)
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
        # print(f'images size: {images.shape}, nao_bbox: {nao_bbox.shape}')
    # print(len(train_dataloader.dataset),train_dataset.__len__())
    # # for data in train_dataloader:
    # it=iter(train_dataloader)
    # nao_bbox_list=[]
    # img,nao_bbox,hand_hm=next(it)
    # nao_bbox_list.append(nao_bbox)
    # nao_bbox_list.append(nao_bbox)
    # print(nao_bbox.shape)
    # nao_bbox_total=torch.cat(nao_bbox_list,0)
    # print(nao_bbox_total.shape)


    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/ADL/test.csv',index=False)
    # for i in range(100):
    #     img, mask, hand_hm = train_dataset.__getitem__(i)
    #     hand_hm=hand_hm.squeeze(0)
    #     img_numpy=img.numpy().transpose(1,2,0)
    #     cv2.imshow('image',img_numpy)
    #     cv2.imshow('image_mask', mask.numpy())
    #     cv2.imshow('hand_mask', hand_hm.numpy())
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     print(img.shape)
    #     print(mask.shape)
    #     print(hand_hm.shape)
    # train_dataset.generate_img_mask_pair()
    # train_dataset.generate_hm()
    # train_dataset = EpicSequenceDataset(args)
    # train_dataloader = DataLoader(train_dataset)
    # train_dataloader = DataLoader(train_dataset, batch_size=4,
    #                               num_workers=3, shuffle=False)
    # # sequence_lens = []
    # for i, data in enumerate(train_dataloader):
    #     img, mask, hand_hm = data
    #     # sequence_lens.append(img.shape[0])
    #     # show(img, mask)
    #     # print(img.shape)

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