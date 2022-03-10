import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import *
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
from models.IntentNet import *
import requests
import torchvision
from torchvision import transforms
from PIL import Image
import os
from tools.CIOU import cal_ciou
from data.dataset import *
from opt import *
from models.IntentNetAttention import *
from data.statistics_TTC import NAODatasetTTC
def draw_boxes(boxes,image):
    for box in boxes:
        box=box.detach().numpy()
        cv2.rectangle(image,
                      (box[0],box[1]),
                      (box[2],box[3]),
                      color=(255,0,0),thickness=2)
    return image


class IntentNetTarget:
    def __init__(self, bounding_boxes, iou_threshold=0.5):
        self.bounding_boxes = bounding_boxes.unsqueeze(0)
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        model_outputs=model_outputs.unsqueeze(0)
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs) == 0:
            return output
        for box in self.bounding_boxes:
            if torch.cuda.is_available():
                box = box.cuda()

            iou=torchvision.ops.box_iou(box,model_outputs)
            output = output + iou
            # ciou=cal_ciou(box,model_outputs)
            # output=output+ciou

        return output


transform = transforms.Compose([  # [h, w]
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
            # , AddGaussianNoise(0., 0.5)
        ])


model_base = IntentNetBase()
model = IntentNetDataAttention()
# model = IntentNetFuseAttentionVector()
model_base.load_state_dict(
    torch.load(
        '/mnt/euler/experiments/ADL/temporal_bbox_baseline/ckpts/model_epoch_250.pth',
        map_location=device
    )['model_state_dict']
)
model.load_state_dict(
    torch.load(
        '/mnt/euler/experiments/ADL/temporal_bbox_attention/ckpts/model_epoch_100.pth',
        map_location=device
    )['model_state_dict']
)
model_base.eval().to(device)
model.eval().to(device)
dataset=NAODataset(mode='test',dataset_name='ADL')
for data in dataset:
    frames, gt_bbox, image_path = data
    image=Image.open(image_path)
    image_float_np=np.array(image,dtype=np.float32)/255
    img_tensor=transform(image)
    input_tensor=frames
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)


    #
    # Run the model and display the detections
    output_base=model_base(input_tensor)
    output,atten = model(input_tensor)
    # print(output)


    target_layers_base = [model_base.visual_feature[-1][0]]
    target_layers = [model.visual_feature[-1][0]]
    # print(target_layers)

    targets=[IntentNetTarget(bounding_boxes=output)]
    # targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    with GradCAM(model_base,target_layers_base,use_cuda=torch.cuda.is_available()) as cam_base, GradCAM(model,target_layers,use_cuda=torch.cuda.is_available()) as cam:

        grayscale_cam_base = cam_base(input_tensor, targets=targets)
        # grayscale_cam = cam(input_tensor, targets=targets)

        grayscale_cam_base=scale_cam_image(grayscale_cam_base,(456,256))
        # grayscale_cam = scale_cam_image(grayscale_cam, (456, 256))
        # # Take the first image in the batch:
        grayscale_cam_base = grayscale_cam_base[0, :]
        # grayscale_cam = grayscale_cam[0, :]

        cam_image_base = show_cam_on_image(image_float_np, grayscale_cam_base, use_rgb=True)
        # cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)

        cam_image_base=draw_boxes(output_base,cam_image_base)
        cam_image = draw_boxes(output, np.array(image))
        image = draw_boxes(gt_bbox.unsqueeze(0), np.array(image))


        cam_image_concat = np.concatenate((cam_image,cam_image_base, (image)), axis=1)

        window_name=image_path[-25:]
        image_display=cv2.cvtColor(cam_image_concat,cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name,image_display)
        atten=atten.squeeze(0).sum(dim=0)
        frame_contribution=torch.nn.functional.softmax(atten,dim=0).detach().numpy()
        fig=plt.figure(figsize=(13.6,5.12))

        plt.plot(frame_contribution)
        plt.ylabel('frame contribution')
        plt.xlabel('frame index')
        plt.show()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                            sep='')
        graph_image = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        graph_image=cv2.resize(graph_image,(456*3,256*2))
        # print(graph_image.shape)
        image_display = np.concatenate((cam_image_concat,graph_image), axis=0)
        image_display=cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name,image_display)
        key=cv2.waitKey(0) & 0xFF
        if key==ord('s'):
            save_path=os.path.join('/media/luohwu/T7/experiments/grad_cam',window_name.replace('/','_'))
            cv2.imwrite(
                filename=save_path,
                img=image_display
            )
            cv2.destroyAllWindows()
            plt.close()
        elif key==ord('q'):
            cv2.destroyAllWindows()
            plt.close()
            break
        else:
            cv2.destroyAllWindows()
            plt.close()
            continue

