from comet_ml import Experiment
import time
from ast import literal_eval

import pandas as pd
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from opt import *
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from opt import *
from tools.Schedulers import DecayCosinWarmRestars
class Trainer(DefaultTrainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        return torch.optim.AdamW(model.parameters(),
                                 lr=cfg.SOLVER.BASE_LR,
                                 betas=(0.9, 0.99),
                                 weight_decay=0)
    # @classmethod
    # def build_lr_scheduler(cls, cfg, optimizer):
    #     return DecayCosinWarmRestars(optimizer,T_0=100,T_mult=2,eta_min=4e-5,decay_rate=0.5,verbose=True)



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
                                            "previous_frames":literal_eval}
                                )
            annos['img_path']=img_path

            if not annos.empty:
                annos_subset = annos[['img_path', 'nao_bbox','noun','objects', 'clip_frame']]
                df_items = df_items.append(annos_subset)


    # df_items=df_items.rename(columns={'class':'category'})
    # df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')
    return df_items

def get_nao_dicts(data):
    print(data.shape)


    dataset_dicts = []
    for idx,item in data.iterrows():
        # print(item['class'])
        # print(item)
        record = {}
        #
        filename = os.path.join(item['img_path'], f"frame_{str(item['clip_frame']).zfill(10)}.jpg")
        # print(filename)
        height, width = cv2.imread(filename).shape[:2]
        #

        record["file_name"] = filename
        record["image_id"] = filename[-25:]
        record["height"] = height
        record["width"] = width
        #
        # annos = v["regions"]
        objs = []
        for i,element in enumerate(item['objects']):

            # un-resized version
            #bbox=element['box']
            bbox_normalized=item['nao_bbox'][i] # [0~1]
            bbox=[bbox_normalized[0]*width,bbox_normalized[1]*height,bbox_normalized[2]*width,bbox_normalized[3]*height]
            obj={
                "bbox":bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id":element['noun_category_id']
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts





if __name__ == '__main__':
    experiment = Experiment(
        api_key="wU5pp8GwSDAcedNSr68JtvCpk",
        project_name="training-faster-rcnn",
        workspace="thesisproject",
    )


    num_cls=len(args.noun_categories)
    # classes = data['class'].unique()
    for d in ["train", "val"]:
        DatasetCatalog.register("nao_" + d, lambda d=d: get_nao_dicts(make_sequence_dataset(d,args.dataset)))
        MetadataCatalog.get("nao_" + d).set(thing_classes=args.noun_categories)
    nao_train_metadata = MetadataCatalog.get("nao_train")
    print(nao_train_metadata)



    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Get the basic model configuration from the model zoo
    # Passing the Train and Validation sets
    cfg.DATASETS.TRAIN = ("nao_train")
    cfg.DATASETS.TEST = ()
    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = args.bs
    cfg.SOLVER.BASE_LR = 0.00125  # pick a good LearningRate
    cfg.SOLVER.MAX_ITER = 49000*3  # No. of iterations
    # cfg.SOLVER.STEPS= (15000,48999*3)
    cfg.SOLVER.STEPS = (48998*3, 48999*3)
    print(f'MAX ITERS:{cfg.SOLVER.MAX_ITER}')
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_cls  # No. of classes = [HINDI, ENGLISH, OTHER]
    cfg.TEST.EVAL_PERIOD = 1000  # No. of iterations after which the Validation Set is evaluated.
    # cfg.MODEL.DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.euler=='False' and args.ait==False:
        cfg.MODEL.DEVICE = 'cpu'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = Trainer(cfg)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()