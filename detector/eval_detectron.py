import sys

sys.path.insert(0, '..')
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
experiment = Experiment(
    api_key="wU5pp8GwSDAcedNSr68JtvCpk",
    project_name="eval-faster-rcnn",
    workspace="thesisproject",
)




def detect_relevant_objects(row,img_folder):
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
            relevant_objs.append(box.cpu().numpy().tolist())
    print(relevant_objs)
    return relevant_objs

if __name__ == '__main__':
    # if args.euler:
    #     import tarfile
    #     scratch_path = os.environ['TMPDIR']
    #     tar_path = f'/cluster/home/luohwu/{args.dataset_file}'
    #     assert os.path.exists(tar_path), f'file not exist: {tar_path}'
    #     print('extracting dataset from tar file')
    #     tar = tarfile.open(tar_path)
    #     tar.extractall(os.environ['TMPDIR'])
    #     tar.close()
    #     print('finished')
    data=make_sequence_dataset('all', 'EGO4D')
    print(type(data))
    print(data)
    classes = args.noun_categories
    num_cls=len(classes)
    # classes = data['class'].unique()
    for d in ["train", "val"]:
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
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = os.path.join( f"../output/model_0024999.pth")   # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_cls  # No. of classes = [HINDI, ENGLISH, OTHER]
    cfg.TEST.EVAL_PERIOD = 5  # No. of iterations after which the Validation Set is evaluated.
    # cfg.MODEL.DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.euler:
            cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # set threshold for this model
    predictor = DefaultPredictor(cfg)

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
    #                                  f"model_final_{args.dataset}.pth")  # Let training initialize from model zoo
    # predictor_resized = DefaultPredictor(cfg)

    #overall evaluation
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluator = COCOEvaluator("nao_val", output_dir="../output")
    val_loader = build_detection_test_loader(cfg, "nao_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

