import os.path

from comet_ml import Experiment
from datetime import datetime
import matplotlib.pyplot as plt
import torch.cuda

from opt import *
import tarfile
from torch import  nn
import pandas as pd
import cv2
import numpy as np
from tools.Schedulers import *
from data.dataset_ambiguity import NAODatasetBase,my_collate
from model.IntentNetAmbiguity import *
from model.IntentNetAmbiguityWord2Vec import *
import  model.IntentNetAmbiguityWord2Vec
from ast import literal_eval
from PIL import Image
from torchvision import transforms
if args.euler:
    experiment = Experiment(
        api_key="wU5pp8GwSDAcedNSr68JtvCpk",
        project_name="intentnetego4d",
        workspace="thesisproject",
    )
    experiment.log_parameters(args.__dict__)
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

exp_name = args.exp_name

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


multi_gpu = True if torch.cuda.device_count() > 1 else False
print(f'using {torch.cuda.device_count()} GPUs')
print('current graphics card is:')
os.system('lspci | grep VGA')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([
    transforms.ToTensor()
])
def add_interaction_scores(row,clip_id,model):
    img_path = os.path.join(args.frames_path, clip_id,f"frame_{str(row['clip_frame']).zfill(10)}.jpg")
    image=Image.open(img_path)
    input=transform(image)
    input=input.to(device)
    input=input.unsqueeze(0)
    output, decoder_feautre, contribution = model(input)
    interaction_scores=compute_interaction_scores(output,row['ro_bbox'])
    return interaction_scores



def main():
    model=IntentNetBaseWord2Vec()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'model size: {total_params}')
    model.load_state_dict(
        torch.load(f'{args.exp_path}/EGO4D/base/ckpts/model_epoch_40.pth', map_location='cpu')[
            'model_state_dict'], strict=False)

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
                                              "noun": literal_eval}
                                )
            annos['interaction_scores']=annos.apply(add_interaction_scores,model=model,clip_id=clip_id,axis=1)
            annos.to_csv(f'/cluster/home/luohwu/dataset/EGO4D/nao_annotations/{clip_id}.csv',index=False)

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
        attention = output[box[1]:box[3], box[0]:box[2]].sum() / (area)
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

