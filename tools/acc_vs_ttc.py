from comet_ml import Experiment
from datetime import datetime
import matplotlib.pyplot as plt
import torch.cuda
from torch import optim
from torch.utils.data import DataLoader
from data.dataset import *
from opt import *
import tarfile
from tools.CIOU import CIOU_LOSS,CIOU_LOSS2,cal_acc_f1
from models.IntentNet import *
from torch import  nn
import pandas as pd
import cv2
from tools.comparison import generate_comparison
import numpy as np
from tools.Schedulers import *
from models.IntentNetAttention import *
from data.statistics_TTC import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
experiment = Experiment(
    api_key="wU5pp8GwSDAcedNSr68JtvCpk",
    project_name="intent-net-with-temporal-context",
    workspace="thesisproject",
    auto_metric_logging=False
)
experiment.log_code(file_name="models/IntentNet.py")
experiment.log_code(file_name="models/IntentNetAttention.py")
experiment.log_code(file_name="data/dataset.py")
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

    model=IntentNetDataAttentionR()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'model size: {total_params}')
    if multi_gpu == True:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load('/cluster/home/luohwu/experiments/EPIC/temporal_bbox/model_DR.pth',map_location='cpu')['model_state_dict'])
    model = model.to(device)

    if args.original_split:
        train_dataset = NAODatasetTTC(mode='train', dataset_name=args.dataset)
        test_dataset = NAODatasetTTC(mode='test', dataset_name=args.dataset)
    else:
        all_data = NAODatasetTTC(mode='all', dataset_name=args.dataset)
        train_size=int(0.8*len(all_data))
        test_size=len(all_data)-train_size
        train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, test_size],
                                                                    generator=torch.Generator().manual_seed(args.seed))

    # train_dataset, test_dataset = ini_datasets(dataset_name=args.dataset, original_split=args.original_split)



    print(f'train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}')
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs,
                                  shuffle=True, num_workers=2,
                                  pin_memory=True,
                                  drop_last=True if torch.cuda.device_count() >=4 else False)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=args.bs,
                                shuffle=True, num_workers=2,
                                pin_memory=True,
                                drop_last=True if torch.cuda.device_count() >= 4 else False)








    criterion = CIOU_LOSS()
    # criterion=nn.MSELoss()

    train_args['ckpt_path'] = os.path.join(train_args['exp_path'],args.dataset,
                                           exp_name, 'ckpts/')
    if not os.path.exists(train_args['ckpt_path']):
        os.mkdir(train_args['ckpt_path'])


    test_loss,acc_table = test(test_dataloader, model, criterion, 1, illustration=False)
    np.set_printoptions(precision=3)
    print(acc_table)
    fig1=plt.figure(1)
    plt.plot(range(1,21),acc_table[1:21,2],'o')
    plt.xticks(range(1,21,1))
    plt.xlabel('Time to contact /s')
    plt.ylabel('Accuracy (IoU>0.5)')
    experiment.log_figure(figure_name=None, figure=fig1, overwrite=False, step=None)
    plt.savefig('/cluster/home/luohwu/workspace/TemporalContextNAO/acc.svg')
    fig2=plt.figure(2,figsize=(8,8))
    plt.pie(acc_table[1:21,0],labels=range(1,21))
    experiment.log_figure(figure_name=None, figure=fig2, overwrite=False, step=None)
    plt.savefig('/cluster/home/luohwu/workspace/TemporalContextNAO/pie.svg')








def test(test_dataloader, model, criterion, epoch, illustration):
    model.eval()
    total_test_loss = 0
    total_acc=0
    total_f1=0
    total_conf_matrix=np.zeros([2,2])
    global_min_iou=999
    global_max_iou=-999
    len_dataset = len(test_dataloader.dataset)
    acc_table=np.zeros((21,3))
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            frames,nao_bbox, img_path,TTC_level = data
            TTC_level=TTC_level.detach().numpy()

            frames=frames.to(device)
            nao_bbox=nao_bbox.to(device)

            outputs= model(frames)
            del frames


            loss, iou = criterion(outputs, nao_bbox)
            acc=iou>0.5
            acc_table[TTC_level, 0] = acc_table[TTC_level, 0] + 1
            acc_table[TTC_level, 1] = acc_table[TTC_level, 1] + [int(item) for item in (iou.cpu().detach().numpy()>0.5)]


            total_test_loss += loss.item()
            total_acc += acc.sum().item()

            if illustration:
                min_acc,min_index=iou.min(0) ###########################################
                max_acc,max_index=iou.max(0) ##############################################
                if global_min_acc > min_acc:
                    global_min_acc = min_acc
                    global_min_image = img_path[min_index]
                    global_min_nao_bbox_gt = nao_bbox[min_index]
                    global_min_nao_bbox = outputs[min_index]
                if global_max_acc < max_acc:
                    global_max_acc = max_acc
                    global_max_image = img_path[max_index]
                    global_max_nao_bbox_gt = nao_bbox[max_index]
                    global_max_nao_bbox = outputs[max_index]
            del outputs, nao_bbox

        if illustration:
            if global_min_acc<999:
                illustration_worst = generate_comparison(global_min_image, global_min_nao_bbox, global_min_nao_bbox_gt)
                experiment.log_image(illustration_worst, name=f'worst_{epoch}',step=epoch)
                experiment.log_text(f'worst_{epoch}: {global_min_image}',step=epoch)
            if global_max_acc>-999:
                illustration_best=generate_comparison(global_max_image, global_max_nao_bbox, global_max_nao_bbox_gt)
                experiment.log_image(illustration_best, name=f'best_{epoch}',step=epoch)
                experiment.log_text(f'best_{epoch}: {global_max_image}', step=epoch)

        test_loss_avg = total_test_loss / len_dataset
        acc_avg = total_acc / len_dataset
    print(f'[epoch {epoch}], [test loss {test_loss_avg:5f}], [acc avg {acc_avg:5f}]')
    experiment.log_metric("test_acc_avg", acc_avg, step=epoch)

    model.train()
    acc_table[1:,2]=acc_table[1:,1]/acc_table[1:,0]

    return test_loss_avg,acc_table



if __name__ == '__main__':

    if args.euler:
        scratch_path = os.environ['TMPDIR']
        tar_path = f'/cluster/home/luohwu/{args.dataset_file}'
        assert os.path.exists(tar_path), f'file not exist: {tar_path}'
        print('extracting dataset from tar file')
        tar = tarfile.open(tar_path)
        tar.extractall(os.environ['TMPDIR'])
        tar.close()
        print('finished')
    # train_data = EpicDatasetV2('train')

    main()

