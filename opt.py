import torch

import argparse
import os

import pandas as pd
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_clips(file_path):
    file = open(file_path, 'r')
    file_lines=file.read()
    list=file_lines.split('\n')[:-1]
    return list

def get_noun_categories():
    noun_categories_path=os.path.join(args.data_path,'noun_categories.csv')
    noun_categories=pd.read_csv(noun_categories_path)['name'].tolist()
    return  noun_categories

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--dataset', type=str, default='EGO4D',
                    help='EPIC , ADL, EGO4D')


parser.add_argument('--euler', default=False,action="store_true",
                    help='runing on euler or local computer')

parser.add_argument('--ait', default=False,action="store_true",
                    help='runing on AIT-server or local computer')



parser.add_argument('--MSE',default=False,action="store_true",
                    help="using MSE as loss function or not")
parser.add_argument('--C3D',default=False,action='store_true',
                    help="using C3D or not")


parser.add_argument('--exp_name', default='exp_name', type=str,
                    help='experiment path (place to store models and logs)')

parser.add_argument('--img_size', default=[256, 456],
                    help='image size: [H, W]')  #
parser.add_argument('--img_resize', default=[224, 224],
                    help='image resize: [H, W]')  #
parser.add_argument('--normalize', default=True, help='subtract mean value')
parser.add_argument('--crop', default=False, help='')

parser.add_argument('--debug', default=False,action="store_true", help='debug')

parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--seed', default=3090, type=int, help='random seed')
parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.05, help='weight decay')
parser.add_argument('--SGD', default=False,action="store_true",
                    help="using SGD or Adam")
parser.add_argument('--data_path', default='/data/luohwu/dataset/EGO4D',type=str,
                    help="path to datasset")
parser.add_argument('--exp_path', default='/data/luohwu/experiments',type=str,
                    help="path to save experiment results")
parser.add_argument('--model_path', default='./pre-train_models/model_epoch_10.pth',type=str,
                    help="path to save experiment results")


args = parser.parse_args()
if args.euler:
    # running on euler
    import tarfile
    scratch_path = os.environ['TMPDIR']
    tar_path = f'/cluster/home/luohwu/{args.dataset_file}'
    assert os.path.exists(tar_path), f'file not exist: {tar_path}'
    print('extracting dataset from tar file')
    tar = tarfile.open(tar_path)
    tar.extractall(os.environ['TMPDIR'])
    tar.close()
    print('finished')
    args.data_path=os.path.join(os.environ['TMPDIR'],'dataset','EGO4D')
    args.annos_path = os.path.join('/cluster/work/hilliges/luohwu/nobackup/training/dataset/EGO4D', 'nao_annotations')
    args.frames_path = os.path.join(args.data_path, 'rgb_frames_resized')
    args.exp_path = '/cluster/home/luohwu/experiments'
elif args.ait:
    # running on ait-server
    args.data_path='/data/luohwu/dataset/EGO4D'
    args.annos_path = '/data/luohwu/dataset/EGO4D/nao_annotations'
    args.frames_path = '/data/luohwu/dataset/EGO4D/rgb_frames'
    args.exp_path = '/data/luohwu/experiments'
else:
    # running on local computer
    # args.data_path = '/home/luohwu/ait-data/dataset/EGO4D/'
    args.annos_path = os.path.join(args.data_path,'nao_annotations')
    args.frames_path = os.path.join(args.data_path,'rgb_frames')
    # args.exp_path = '/home/luohwu/ait-data/experiments'



# args.data_path='/media/luohwu/T7/dataset/EGO4D/' if args.euler==False else os.path.join(os.environ['TMPDIR'],'dataset','EGO4D')
# # args.data_path='/home/luohwu/nobackup/training/dataset/EGO4D' if args.euler==False else '/cluster/work/hilliges/luohwu/nobackup/training/dataset/EGO4D'
# args.exp_path='/home/luohwu/euler/experiments' if args.euler==False else '/cluster/home/luohwu/experiments'
#
# # args.data_path='/media/luohwu/T7/dataset/EGO4D/' if args.euler==False else '/cluster/work/hilliges/luohwu/nobackup/training/dataset/EGO4D'

args.train_clip_ids = read_clips(os.path.join(args.data_path,'train_clips.txt'))
args.val_clip_ids = read_clips(os.path.join(args.data_path,'val_clips.txt'))
if args.debug:
    # train_clip_id = {'P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_06'}
    args.train_clip_ids = args.train_clip_ids[:]
    args.val_clip_ids = args.val_clip_ids[0:2]


args.all_clip_ids = args.train_clip_ids + args.val_clip_ids
args.noun_categories=get_noun_categories()

# annos_path = 'nao_annotations'
# frames_path = 'rgb_frames_resized'  #
# args.annos_path=os.path.join(args.data_path,annos_path)
# if args.euler:
#     args.annos_path=os.path.join('/cluster/work/hilliges/luohwu/nobackup/training/dataset/EGO4D',annos_path)
# else:
#     args.annos_path=os.path.join('/home/luohwu/nobackup/training/dataset/EGO4D',annos_path)
# args.frames_path=os.path.join(args.data_path,frames_path)







if __name__=='__main__':
    print(f'original split? {args.original_split}')
    if args.euler:
        print(f'using euler')
    else:
        print(f'using local ')