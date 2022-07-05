import os.path

from comet_ml import Experiment
from datetime import datetime
import matplotlib.pyplot as plt
import torch.cuda
import time
from opt import *
import tarfile
from torch import  nn
import pandas as pd
import cv2
import numpy as np
from tools.Schedulers import *
from data.dataset_ambiguity_clip import NAODatasetClip,my_collate
from model.IntentNetAmbiguityClip import *
from ast import literal_eval
from PIL import Image
from torchvision import transforms
from utils.augmentation import *
if args.ait:
    experiment = Experiment(
        api_key="wU5pp8GwSDAcedNSr68JtvCpk",
        project_name="intentnetego4d_clip",
        workspace="thesisproject",
    )
    experiment.log_parameters(args.__dict__)
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

exp_name = args.exp_name

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


multi_gpu = True if torch.cuda.device_count() > 1 else False
print(f'using {torch.cuda.device_count()} GPUs')
print('current graphics card is:')
os.system('lspci | grep VGA')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_current_frame=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet
])
transform_past_frames = transforms.Compose([
    # RandomSizedCrop(size=128, consistent=True, p=1.0),
    RandomHorizontalFlip(consistent=True),
    Scale(size=(128, 128)),
    RandomGray(consistent=False, p=0.5),
    ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.5),
    ToTensor(),
    Normalize()
])
def add_interaction_scores(row,clip_id):
    img_path = os.path.join(args.frames_path, clip_id)
    frame_indices = row['all_frame']
    pil_images_list = [pil_loader(os.path.join(img_path, f'frame_{str(idx).zfill(10)}.jpg')) for idx in
                       frame_indices]
    past_images_list = transform_past_frames(pil_images_list)
    (C, H, W) = past_images_list[0].size()
    past_frames = torch.stack(past_images_list, 0)
    past_frames = past_frames.view(5, 5, C, H, W).transpose(1, 2)
    del past_images_list

    current_frame_path = os.path.join(img_path, f"frame_{str(row['clip_frame']).zfill(10)}.jpg")
    current_frame = Image.open(current_frame_path)
    current_frame = transform_current_frame(current_frame)
    past_frames=past_frames.unsqueeze(0).to(device)
    current_frame=current_frame.unsqueeze(0).to(device)
    # print(f'shape of current frame: {current_frame.shape}, shape of past frames: {past_frames.shape}')
    output, decoder_feautre, contribution = model(current_frame,past_frames)
    interaction_scores=compute_interaction_scores(output,row['ro_bbox'])
    # print(interaction_scores)
    # time.sleep(0.1)
    return interaction_scores



def main():
    global model
    model=IntentNetClipWord2Vec()
    model=nn.DataParallel(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'model size: {total_params}')
    model_path=args.model_path
    print(model_path)
    model.load_state_dict(
        torch.load(model_path, map_location='cpu')[
            'model_state_dict'], strict=True)

    model = model.to(device)
    model.eval()

    print(f'dataset name: EGO4D')
    # val is the same as test
    mode='val'
    if mode == 'all':
        clip_ids = args.all_clip_ids
    elif mode == 'train':
        clip_ids = args.train_clip_ids
    else:
        clip_ids = args.val_clip_ids

    print(f'start load {mode} data, #videos: {len(clip_ids)}')
    for idx,clip_id in enumerate(sorted(clip_ids)):
        print(f'{idx}/{len(clip_ids)} working on clip: {clip_id}')
        anno_name = clip_id + '.csv'
        anno_path = os.path.join(args.annos_path, anno_name)
        if os.path.exists(anno_path):
            # print(anno_path)

            annos = pd.read_csv(anno_path
                                , converters={"nao_bbox": literal_eval,
                                              "objects": literal_eval,
                                              "previous_frames": literal_eval,
                                              "cls": literal_eval,
                                              "ro_bbox": literal_eval,
                                              "noun": literal_eval,
                                              "all_frame":literal_eval}
                                )
            annos['interaction_scores']=[[] for r in range(len(annos))]
            # for i in range(len(annos)):
            #     annos.at[i,'interaction_scores']=add_interaction_scores(row=annos.iloc[i,:],clip_id=clip_id)
            #     annos.at[i, 'interaction_scores'] = [123]
            annos['interaction_scores']=annos.apply(add_interaction_scores,clip_id=clip_id,axis=1)
            annos.to_csv(anno_path,index=False)

    # df_items=df_items.rename(columns={'class':'category'})
    # df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')


def compute_interaction_scores(output,bboxes):
    attention_list=[]
    ro_bbox = bboxes
    for i,box in enumerate(ro_bbox):
        # print(box)
        area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        attention = output[0,box[1]:box[3], box[0]:box[2]].sum()/area
        attention_list.append(attention.item())
    return attention_list



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

    main()

