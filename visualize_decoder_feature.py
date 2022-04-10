import os.path

from comet_ml import Experiment
from datetime import datetime
import matplotlib.pyplot as plt
import torch.cuda
from torch import optim
from torch.utils.data import DataLoader
from data.dataset_ambiguity import *
from opt import *
import tarfile
from torch import  nn
import pandas as pd
import cv2
import numpy as np
from tools.Schedulers import *
from data.dataset_ambiguity import NAODatasetBase,my_collate
from model.IntentNetAmbiguity import *
from  model.IntentNetAmbiguityWord2Vec import *
import  model.IntentNetAmbiguityWord2Vec
import model.IntentNetAmbiguity

SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

exp_name = args.exp_name

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

torch.set_printoptions(edgeitems=150)
multi_gpu = True if torch.cuda.device_count() > 1 else False
print(f'using {torch.cuda.device_count()} GPUs')
print('current graphics card is:')
os.system('lspci | grep VGA')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def main():
    # model=IntentNetBaseWord2VecNormalized()
    model = IntentNetBaseWord2Vec()
    model=nn.DataParallel(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'model size: {total_params}')
    model_path=os.path.join(args.exp_path,'EGO4D','base','ckpts/model_epoch_40.pth')
    # model_path='/data/luohwu/experiments/EGO4D/base/ckpts/model_epoch_80.pth'
    model.load_state_dict(
        torch.load(model_path, map_location='cpu')[
            'model_state_dict'], strict=True)

    model = model.to(device)

    train_dataset = NAODatasetBase(mode='train', dataset_name=args.dataset)
    test_dataset = NAODatasetBase(mode='test', dataset_name=args.dataset)


    # train_dataset, test_dataset = ini_datasets(dataset_name=args.dataset, original_split=args.original_split)



    print(f'train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}')

    test_dataloader = DataLoader(train_dataset,
                                batch_size=1,
                                shuffle=True, num_workers=2,
                                pin_memory=True,
                                drop_last=True if torch.cuda.device_count() >= 4 else False,
                                 collate_fn=my_collate)






    criterion = AttentionLoss()
    # criterion=nn.MSELoss()



    for epoch in range(1):

        test_loss = eval(test_dataloader, model, criterion, epoch, illustration=True)




def eval(test_dataloader, model, criterion, epoch, illustration):
    model.eval()
    total_test_loss = 0
    top_1_all=0
    top_3_all=0
    len_dataset = len(test_dataloader.dataset)
    np.set_printoptions(suppress=True)
    with torch.no_grad():
        for i_iter, data in enumerate(test_dataloader):
            frame,all_bboxes, img_path,mask,labels = data
            if (len(labels[0]) > 3):
                frame=frame.to(device)
                mask=mask.to(device)
                print(f'sum of mask: {mask.sum()}')

                output, decoder_feature,contributions = model(frame)
                print(f'sum of mask: {mask.sum()}, sum of outputs: {output.sum()}')
                print(contributions.detach().cpu().numpy())
                similarity_list, attention_list = compute_similarity_and_interaction_scores_w2v(
                    labels[0],
                    output=output[0], bboxes=all_bboxes[0])
                print(similarity_list)
                print(attention_list)
                for j in range(64):
                    feature=torch.tanh(decoder_feature[0,j,:,:])*255*3
                    # feature = torch.softmax(decoder_feature[0, i, :, :].view(1,1,-1),dim=2).view(256,456) * 255*100
                    feature_numpy=feature.cpu().detach().numpy()
                    if not os.path.exists(f'{args.exp_path}/{args.dataset}/ambiguity/features/{i_iter}'):
                        os.system(f'mkdir -p {args.exp_path}/{args.dataset}/ambiguity/features/{i_iter}')
                        # os.mkdir(f'{args.exp_path}/{args.dataset}/ambiguity/features/{i_iter}')
                    saved_path = f'{args.exp_path}/{args.dataset}/ambiguity/features/{i_iter}/{j}.jpg'
                    cv2.imwrite(saved_path,feature_numpy)

                del frame
                illustration=True
                if illustration:
                    for i in range(mask.shape[0]):
                        top_1,top_3=compute_acc(output[i],all_bboxes[i])
                        top_1_all+=top_1
                        top_3_all+=top_3
                        mask=np.ones((256,456),dtype=np.uint8)*1
                        img_path_item=img_path[i]
                        original_image=cv2.imread(img_path_item)
                        original_image_bbox = cv2.imread(img_path_item)
                        original_image_bbox=cv2.rectangle(original_image_bbox,(all_bboxes[i][0][0],all_bboxes[i][0][1]),
                                                     (all_bboxes[i][0][2],all_bboxes[i][0][3]),(255,0,0),2)
                        mask[all_bboxes[i][0][1]:all_bboxes[i][0][3],all_bboxes[i][0][0]:all_bboxes[i][0][2]]=1
                        ro_bboxes=all_bboxes[i][1:]
                        for j,box in enumerate(ro_bboxes):
                            original_image_bbox = cv2.rectangle(original_image_bbox, (box[0], box[1]),
                                                           (box[2], box[3]), (0, 255, 0), 2)
                            mask[box[1]:box[3],box[0]:box[2]]=1
                            font_size=0.5
                            cv2.putText(original_image_bbox, f'i:{attention_list[j]:.2f}', (box[0]+5,box[1]+25),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (36, 12, 255), 1)
                            cv2.putText(original_image_bbox, f's:{similarity_list[j]:.2f}', (box[0]+5, box[1] +10),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (36, 12, 255), 1)

                        output_item=output[i]*255
                        output_item=output_item.cpu().detach().numpy().astype(np.uint8)
                        output_item=output_item*mask
                        # output_item=heatmap_to_bbox(output_item)
                        output_item=cv2.applyColorMap(output_item,cv2.COLORMAP_JET)
                        masked_img = output_item
                        # masked_img = cv2.addWeighted(original_image, 0, output_item, 1.0, 0)

                        saved_img=np.concatenate((original_image_bbox,masked_img),axis=1)
                        saved_path = f'{args.exp_path}/{args.dataset}/ambiguity/features/{i_iter}/original.jpg'
                        # print(saved_path)
                        saved_img=cv2.resize(saved_img,(456*4,256*2))
                        cv2.imwrite(saved_path,saved_img)
                        with open(f'{args.exp_path}/{args.dataset}/ambiguity/features/{i_iter}/img_path.txt','w') as f:
                            f.write(f'end of one sample: {img_path[0]}')
                            f.close()
                        print(f'end of one sample: {img_path[0]}')

        test_loss_avg = total_test_loss / len_dataset
    print(f'top-1: {top_1_all/len_dataset}, top-3: {top_3_all/len_dataset}')

    model.train()

    return test_loss_avg



def compute_similarity_and_interaction_scores(decoder_feature,output,bboxes):
    similarity_list=[]
    attention_list=[]
    nao_bbox = bboxes[0]
    feature_nao_bbox = F.adaptive_max_pool2d(
        decoder_feature[  :,nao_bbox[1]:nao_bbox[3], nao_bbox[0]:nao_bbox[2]].unsqueeze(0), output_size=17)
    feature_nao_bbox = torch.flatten(feature_nao_bbox, 0).unsqueeze(0)
    ro_bbox = bboxes[1:]
    for box in ro_bbox:
        # print(box)
        feature_ro_bbox = F.adaptive_max_pool2d(decoder_feature[:, box[1]:box[3], box[0]:box[2]].unsqueeze(0),
                                                output_size=17)
        feature_ro_bbox = torch.flatten(feature_ro_bbox, 0).unsqueeze(0)
        cos_similarity = torch.nn.functional.cosine_similarity(feature_ro_bbox, feature_nao_bbox)
        similarity_list.append(cos_similarity.item())

        area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        attention = output[box[1]:box[3], box[0]:box[2]].sum() / (area)
        attention_list.append(attention.item())
    return similarity_list,attention_list

def compute_similarity_and_interaction_scores_w2v(labels,output,bboxes):
    similarity_list=[]
    attention_list=[]
    nao_bbox = bboxes[0]
    nao_label=labels[0]
    ro_bbox = bboxes[1:]
    ro_labels=labels[1:]
    for i,box in enumerate(ro_bbox):
        # print(box)
        label_ro=ro_labels[i]
        similarity=model.IntentNetAmbiguityWord2Vec.compute_similarity(nao_label,label_ro)
        similarity_list.append(similarity.item())

        area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        attention = output[box[1]:box[3], box[0]:box[2]].sum() / (area)
        attention_list.append(attention.item())
    return similarity_list,attention_list


def compute_iou(bbox1,bbox2):
    area1=(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2=(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    inter_l=max(bbox1[0],bbox2[0])
    inter_r=min(bbox1[2],bbox2[2])
    inter_t=max(bbox1[1],bbox2[1])
    inter_b=min(bbox1[3],bbox2[3])
    inter_area = max((inter_r - inter_l),0) * max((inter_b - inter_t),0)
    return inter_area/(area1+area2-inter_area)

def compute_acc(output,bboxes):
    num_bboxes=len(bboxes)-1
    if num_bboxes==0:
        return 0,0
    output=output.cpu().detach().numpy()
    attention_vector=np.zeros(num_bboxes)
    iou_vector=np.zeros(num_bboxes)
    gt=bboxes[0]
    for i,box in enumerate(bboxes[1:]):
        area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
        attention=output[box[1]:box[3],box[0]:box[2]].sum()/(area)
        attention_vector[i]=attention
        iou_vector[i]=compute_iou(gt,box)
    desceending_index=np.argsort(attention_vector)[::-1]
    top_1=1 if iou_vector[desceending_index[0]]>0.5 else 0
    if num_bboxes>3:
        top_3= 1 if np.any(iou_vector[desceending_index][:3]>0.5) else 0
    else:
        top_3=1 if np.any(iou_vector>0.5) else 0
    return top_1,top_3


if __name__ == '__main__':
    if args.ait:
        experiment = Experiment(
            api_key="wU5pp8GwSDAcedNSr68JtvCpk",
            project_name="intentnetego4d",
            workspace="thesisproject",
        )
    main()

