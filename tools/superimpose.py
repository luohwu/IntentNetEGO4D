import cv2
import numpy as np
from comet_ml import Experiment
import torch

height=256
width=456
img_path='/home/luohwu/euler/dataset/ADL/rgb_frames/P_15/frame_0000032178.jpg'
nao_bbox=[0,  0, 281, 123]
img=cv2.imread(img_path)
cv2.rectangle(img, (nao_bbox[0], nao_bbox[1]), (nao_bbox[2],nao_bbox[3]),(0,0,255),5)


mask=torch.rand([height,width]).numpy()*255
# mask[nao_bbox[1]:nao_bbox[3],nao_bbox[0]:nao_bbox[2]]=255
mask=mask.astype(np.uint8)
mask_img=cv2.applyColorMap(mask,cv2.COLORMAP_JET)

result=cv2.addWeighted(img,0.7, mask_img,0.3,0)

cv2.imshow('sample result',result)
cv2.waitKey(0)
