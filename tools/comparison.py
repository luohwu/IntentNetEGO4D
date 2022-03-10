import cv2
from opt import *
import numpy as np
height = args.img_size[0]
width = args.img_size[1]
def generate_comparison(img_path, nao_bbox, nao_bbox_gt):

    nao_bbox = nao_bbox.cpu().numpy().astype(np.int)
    nao_bbox_gt = nao_bbox_gt.cpu().numpy()
    print(nao_bbox,nao_bbox_gt)
    # print(f'nao_bbox: {nao_bbox}, ground truth: {nao_bbox_gt}')
    img = cv2.imread(img_path)
    cv2.rectangle(img, (nao_bbox_gt[0], nao_bbox_gt[1]), (nao_bbox_gt[2], nao_bbox_gt[3]), (0, 0, 255), 5)
    cv2.rectangle(img, (nao_bbox[0], nao_bbox[1]), (nao_bbox[2], nao_bbox[3]), (255, 255, 0), 5)
    # mask = np.zeros([height, width])
    # mask[nao_bbox[1]:nao_bbox[3], nao_bbox[0]:nao_bbox[2]] = 255
    # mask = mask.astype(np.uint8)
    # mask_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # result = cv2.addWeighted(img, 0.7, mask_img, 0.3, 0)
    # return result
    return img