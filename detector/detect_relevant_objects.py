import sys
sys.path.insert(0,'..')
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
from detectron2.modeling import build_model
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
from data.dataset_ambiguity import make_sequence_dataset
from train_detectron import get_nao_dicts


def compute_iou(bbox1,bbox2):
    area1=(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2=(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    inter_l=max(bbox1[0],bbox2[0])
    inter_r=min(bbox1[2],bbox2[2])
    inter_t=max(bbox1[1],bbox2[1])
    inter_b=min(bbox1[3],bbox2[3])
    inter_area = max((inter_r - inter_l),0) * max((inter_b - inter_t),0)
    return inter_area/(area1+area2-inter_area)

def none_maximum_suppression(ro_bboxes):
    length=len(ro_bboxes)
    idx_to_remove = []
    if length<2:
        return idx_to_remove
    for i in range(length):
        for j in range(i+1,length):
            iou=compute_iou(ro_bboxes[i],ro_bboxes[j])
            # print(iou)
            if iou>0.6:
                idx_to_remove.append(i)
    # if len(idx_to_remove)>0:
    #     print(f'idx: {idx_to_remove}')
    #     print(f'bboxes: {ro_bboxes}')
    #     print(f'bboxes_nms: {result}')
    return idx_to_remove




def detect_relevant_objects(row,img_folder):
    img_path = os.path.join(img_folder, f"frame_{str(row['clip_frame']).zfill(10)}.jpg")
    img=cv2.imread(img_path)
    # cv2.imshow(img_path[-30:],img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    outputs = predictor(img)
    pred_boxes = outputs['instances'].pred_boxes
    relevant_objs = []
    if len(pred_boxes) > 0:
        for box in pred_boxes:
            relevant_objs.append(box.int().cpu().numpy().tolist())
    # print(relevant_objs)
    idx_to_remove=none_maximum_suppression(relevant_objs)
    # print(idx_to_remove)
    ro_bbox = [relevant_objs[i] for i in range(len(relevant_objs)) if i not in idx_to_remove]

    cls=outputs['instances'].pred_classes.cpu().detach().numpy().tolist()
    scores = outputs['instances'].scores.cpu().detach().numpy().tolist()
    # labels= nao_train_metadata.thing_classes[cls]
    labels=cls
    cls = [labels[i] for i in range(len(labels)) if i not in idx_to_remove]
    scores = [scores[i] for i in range(len(scores)) if i not in idx_to_remove]

    return ro_bbox,cls,scores

def detect_relevant_objects2(row,img_folder):
    img_path = os.path.join(img_folder, f"frame_{str(row['frame']).zfill(10)}.jpg")
    img=cv2.imread(img_path)
    # cv2.imshow(img_path[-30:],img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    outputs = predictor(img)

    pred_boxes = outputs['instances'].pred_boxes
    relevant_objs = []
    if len(pred_boxes) > 0:
        for box in pred_boxes:
            relevant_objs.append(box.int().cpu().numpy().tolist())

    idx_to_remove = none_maximum_suppression(relevant_objs)

    cls=outputs['instances'].pred_classes.cpu().detach().numpy().tolist()
    labels= nao_train_metadata.thing_classes[cls]

    result = [labels[i] for i in range(len(labels)) if i not in idx_to_remove]
    return result




if __name__ == '__main__':
    experiment = Experiment(
        api_key="wU5pp8GwSDAcedNSr68JtvCpk",
        project_name="eval-faster-rcnn",
        workspace="thesisproject",
    )

    classes = args.noun_categories
    num_cls=len(classes)
    # classes = data['class'].unique()
    for d in ["train", "test"]:
        DatasetCatalog.register("nao_" + d, lambda d=d: get_nao_dicts(make_sequence_dataset(d,args.dataset)))
        MetadataCatalog.get("nao_" + d).set(thing_classes=classes)
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
    cfg.MODEL.WEIGHTS = os.path.join( f"../output_all/model_0024999.pth")   # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_cls  # No. of classes = [HINDI, ENGLISH, OTHER]
    cfg.TEST.EVAL_PERIOD = 5  # No. of iterations after which the Validation Set is evaluated.
    # cfg.MODEL.DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.euler and not args.ait:
            cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  #0.1~ 0.27 , 0.35~0.35
    predictor = DefaultPredictor(cfg)

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
    #                                  f"model_final_{args.dataset}.pth")  # Let training initialize from model zoo
    # predictor_resized = DefaultPredictor(cfg)

    clip_id_list = sorted(args.all_clip_ids)
    for i,clip_id in enumerate(sorted(clip_id_list)):
        img_folder=os.path.join(args.frames_path, clip_id)
        anno_file_path = os.path.join(args.annos_path, f'{clip_id}.csv')
        if os.path.exists(anno_file_path):
            print(f'{i}/{len(clip_id_list)}current video id: {clip_id}')
            annotations = pd.read_csv(anno_file_path, converters={"nao_bbox": literal_eval})
            annotations[['ro_bbox','cls','scores']]=annotations.apply(detect_relevant_objects,args=[img_folder],axis=1,result_type='expand')
            annotations.to_csv(anno_file_path, index=False)

