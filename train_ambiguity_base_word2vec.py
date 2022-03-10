import os.path

from comet_ml import Experiment
from datetime import datetime
import matplotlib.pyplot as plt
import torch.cuda
from torch import optim
from torch.utils.data import DataLoader
from opt import *
import tarfile
from torch import  nn
import pandas as pd
import cv2
import numpy as np
from tools.Schedulers import *
from data.dataset_ambiguity import NAODatasetBase,my_collate
from model.IntentNetAmbiguityWord2Vec import *

experiment = Experiment(
    api_key="wU5pp8GwSDAcedNSr68JtvCpk",
    project_name="intentnetego4d",
    workspace="thesisproject",
)
experiment.log_code('data/dataset_ambiguity.py')
experiment.log_code('model/IntentNetAmbiguity.py')
experiment.log_code('model/IntentNetAmbiguityWord2Vec.py')
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



def main():

    model = IntentNetBaseWord2Vec()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'model size: {total_params}')
    # model.load_state_dict(
    #     torch.load('/cluster/home/luohwu/experiments/ADL/ambiguity/ckpts/model_epoch_50.pth', map_location='cpu')[
    #         'model_state_dict'])

    model = model.to(device)

    train_dataset = NAODatasetBase(mode='train', dataset_name=args.dataset)
    val_dataset = NAODatasetBase(mode='val', dataset_name=args.dataset)


    # train_dataset, val_dataset = ini_datasets(dataset_name=args.dataset, original_split=args.original_split)



    print(f'train dataset size: {len(train_dataset)}, test dataset size: {len(val_dataset)}')
    num_workers=4
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs,
                                  shuffle=True, num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True if torch.cuda.device_count() >=4 else False,
                                  collate_fn=my_collate)
    test_dataloader = DataLoader(val_dataset,
                                batch_size=args.bs,
                                shuffle=True, num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True if torch.cuda.device_count() >= 4 else False,
                                 collate_fn=my_collate)

    if args.SGD:
        print('using SGD')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:

        print('using AdamW')
        optimizer = optim.AdamW(filter(lambda  p: p.requires_grad,model.parameters()),
                                lr=args.lr,
                                betas=(0.9, 0.99),
                                weight_decay=0
                                # ,weight_decay=args.weight_decay
                                )




    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                  mode='min',
    #                                                  factor=0.8,
    #                                                  patience=3,
    #                                                  verbose=True,
    #                                                  min_lr=0.000001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=4e-5,verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25,eta_min=1e-5,verbose=True)
    # scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.98,verbose=False)
    # scheduler=CosExpoScheduler(optimizer,switch_step=100,eta_min=4e-5,gamma=0.995,min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100,T_mult=2, eta_min=4e-5, verbose=True)
    scheduler=DecayCosinWarmRestars(optimizer,T_0=100,T_mult=2,eta_min=4e-5,decay_rate=0.5,verbose=True)
    """"
    Heatmap version
    """

    criterion = AttentionLossWord2Vec()
    # criterion=nn.MSELoss()

    ckpt_path = os.path.join(args.exp_path,args.dataset,
                                           exp_name, 'ckpts/')
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)


    train_loss_list = []
    test_loss_list = []
    current_epoch = 0
    epoch_save = 20
    for epoch in range(current_epoch + 1, args.epochs + 1):
        print(f"==================epoch :{epoch}/{args.epochs}===============================================")

        train_loss = train(train_dataloader, model, criterion, optimizer, epoch=epoch)
        test_loss = eval(test_dataloader, model, criterion, epoch, illustration=False)

        # scheduler.step(test_loss)
        scheduler.step()
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        if epoch % epoch_save == 0:
            checkpoint_path = os.path.join(ckpt_path, f'model_epoch_{epoch}.pth')
            print(checkpoint_path)

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()
                        },
                       checkpoint_path)
            eval(test_dataloader, model, criterion, epoch, illustration=True)

        experiment.log_metrics({"test_loss": test_loss, "train_loss": train_loss}, step=epoch)
        print(f'train loss: {train_loss:.8f} | test loss:{test_loss:.8f}')


def train(train_dataloader, model, criterion, optimizer,epoch):
    train_losses = 0.
    total_acc=0
    total_f1=0

    len_dataset = len(train_dataloader.dataset)
    for i, data in enumerate(train_dataloader, start=1):
        frame, all_bboxes, img_path,mask,labels = data
        # print(f'file path: {img_path}')
        # print(f'previous_frames:{previous_frames.shape}, cur_frame: {current_frame.shape}')
        frame=frame.to(device)
        mask=mask.to(device)

        #forward
        output, decoder_feautre,contribution = model(frame)
        del frame

        # loss and acc
        loss = criterion(output, mask,labels,all_bboxes)
        # loss = criterion(outputs, nao_bbox)
        # acc, f1, conf_matrix = cal_acc_f1(outputs, nao_bbox)


        del output, decoder_feautre

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += loss.item()
    return train_losses / len_dataset
    # return train_losses

def eval(test_dataloader, model, criterion, epoch, illustration):
    model.eval()
    total_test_loss = 0
    len_dataset = len(test_dataloader.dataset)
    top_1_all=0
    top_3_all=0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            frame,all_bboxes, img_path,mask,labels = data
            frame=frame.to(device)
            mask=mask.to(device)

            output, decoder_feautre,contributions = model(frame)
            del frame


            loss = criterion(output, mask,labels,all_bboxes)


            total_test_loss += loss.item()
            if illustration:
                for i in range(mask.shape[0]):
                    top_1,top_3=compute_acc(output[i],all_bboxes[i])
                    top_1_all+=top_1
                    top_3_all+=top_3
                    mask=np.ones((256,456),dtype=np.uint8)
                    img_path_item=img_path[i]
                    original_image=cv2.imread(img_path_item)
                    original_image=cv2.rectangle(original_image,(all_bboxes[i][0][0],all_bboxes[i][0][1]),
                                                 (all_bboxes[i][0][2],all_bboxes[i][0][3]),(255,0,0),2)
                    # mask[all_bboxes[i][0][1]:all_bboxes[i][0][3],all_bboxes[i][0][0]:all_bboxes[i][0][2]]=1
                    ro_bboxes=all_bboxes[i][1:]
                    for box in ro_bboxes:
                        original_image = cv2.rectangle(original_image, (box[0], box[1]),
                                                       (box[2], box[3]), (0, 255, 0), 2)
                        mask[box[1]:box[3],box[0]:box[2]]=1

                    output_item=output[i]*255
                    output_item=output_item.cpu().detach().numpy().astype(np.uint8)
                    output_item=output_item*mask
                    output_item=cv2.applyColorMap(output_item,cv2.COLORMAP_JET)
                    masked_img=cv2.addWeighted(original_image,0.7,output_item,0.3,0)
                    saved_img=np.concatenate((original_image,masked_img),axis=1)
                    if top_1 == 1:
                        saved_path = os.path.join(f'/cluster/home/luohwu/experiments/{args.dataset}/ambiguity/top_1_correct/',
                                                  img_path_item[-25:].replace('/', f'_{epoch}'))
                    else:
                        saved_path = os.path.join(f'/cluster/home/luohwu/experiments/{args.dataset}/ambiguity/top_1_wrong/',
                                                  img_path_item[-25:].replace('/', f'_{epoch}')) # print(saved_path)
                    cv2.imwrite(saved_path,saved_img)


        test_loss_avg = total_test_loss / len_dataset
    print(f'[epoch {epoch}], [test loss {test_loss_avg:5f}] ')
    if illustration==True:
        print(f'top-1: {top_1_all / len_dataset}, top-3: {top_3_all / len_dataset}')

    model.train()

    return test_loss_avg


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

    # if args.euler:
    #     scratch_path = os.environ['TMPDIR']
    #     tar_path = f'/cluster/home/luohwu/{args.dataset_file}'
    #     assert os.path.exists(tar_path), f'file not exist: {tar_path}'
    #     print('extracting dataset from tar file')
    #     tar = tarfile.open(tar_path)
    #     tar.extractall(os.environ['TMPDIR'])
    #     tar.close()
    #     print('finished')

    main()

