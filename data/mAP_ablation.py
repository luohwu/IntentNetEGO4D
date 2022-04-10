import sys
sys.path.insert(0,'..')
from opt import *
from dataset_ambiguity import NAODatasetBase
from ast import literal_eval
import  numpy as np
def compute_iou(bbox1,bbox2):
    area1=(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2=(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    inter_l=max(bbox1[0],bbox2[0])
    inter_r=min(bbox1[2],bbox2[2])
    inter_t=max(bbox1[1],bbox2[1])
    inter_b=min(bbox1[3],bbox2[3])
    inter_area = max((inter_r - inter_l),0) * max((inter_b - inter_t),0)
    return inter_area/(area1+area2-inter_area)

def unnormalize_bbox(row):
    result=[]
    nao_bboxes=row['nao_bbox']
    for bbox in nao_bboxes:
        bbox = [bbox[0] * 456, bbox[1] * 256, bbox[2] * 456, bbox[3] * 256]
        bbox = [round(coor) for coor in bbox]
        result.append(bbox)
    return result


def make_sequence_dataset_for_eval(mode='train',dataset_name='EGO4D'):
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
                                            "previous_frames":literal_eval,
                                             "cls":literal_eval,
                                             "ro_bbox":literal_eval,
                                             "noun":literal_eval,
                                             "scores":literal_eval,
                                             "interaction_scores":literal_eval}
                                )
            annos['img_path']=img_path
            annos['cls']=annos.apply(lambda row:[args.noun_categories[item] for item in row['cls']],axis=1)
            annos['nao_bbox']=annos.apply(unnormalize_bbox,axis=1)

            if not annos.empty:
                annos_subset = annos[['uid', 'nao_bbox','noun','objects', 'clip_frame','ro_bbox','cls','scores',"interaction_scores"]]
                df_items = df_items.append(annos_subset)


    # df_items=df_items.rename(columns={'class':'category'})
    # df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')
    return df_items

def check_no_overlap(new_bbox,bbox_list):
    for bbox in bbox_list:
        iou=compute_iou(new_bbox,bbox)
        if iou>0.5:
            return False
    return True


def compute_topK_result(item,k=5):
    num_gt=len(item['noun'])
    num_pred=len(item['cls'])
    quota=(k)*num_gt
    idx=sorted(range(num_pred),key=lambda k:-item['scores'][k])
    item['ro_bbox']=[item['ro_bbox'][i] for i in idx]
    item['scores'] = [item['scores'][i] for i in idx]
    item['cls'] = [item['cls'][i] for i in idx]
    result_ro_bbox = []
    result_scores = []
    result_cls = []

    if num_gt==1:
        gt_label=item['noun'][0]
        gt_bbox=item['nao_bbox'][0]
        for i,label in enumerate(item['cls']):
            if i<quota:
                if label == gt_label:
                    iou=compute_iou(gt_bbox,item['ro_bbox'][i])
                    if iou>0.5 and len(result_cls)<1:
                        result_ro_bbox.append(item['ro_bbox'][i])
                        result_cls.append(item['cls'][i])
                        result_scores.append(item['scores'][i])
            else:
                result_ro_bbox.append(item['ro_bbox'][i])
                result_cls.append(item['cls'][i])
                result_scores.append(item['scores'][i])
    else:
        # return item
        gt_label=item['noun']
        gt_bbox=item['nao_bbox']
        for i,label in enumerate(item['cls']):
            if i<quota:
                if label in gt_label:
                    index=gt_label.index(label)
                    iou=compute_iou(item['ro_bbox'][i],gt_bbox[index])
                    if iou>0.5 and check_no_overlap(item['ro_bbox'][i],result_ro_bbox):
                        result_ro_bbox.append(item['ro_bbox'][i])
                        result_cls.append(item['cls'][i])
                        result_scores.append(item['scores'][i])
            else:
                result_ro_bbox.append(item['ro_bbox'][i])
                result_cls.append(item['cls'][i])
                result_scores.append(item['scores'][i])

    item['ro_bbox'] = result_ro_bbox
    item['cls'] = result_cls
    item['scores'] = result_scores

    return item



if __name__=='__main__':
    data=make_sequence_dataset_for_eval('val','EGO4D')
    length=data.shape[0]
    for i in range(length):
        print(f'{i}/{length}')
        item=data.iloc[i,:].copy()
        # print(item)
        item=compute_topK_result(item,5)
        # print(item)
        #write gt
        gt_file=os.path.join(args.data_path,'map_input','ground-truth',f"{item['uid']}.txt")
        pred_file=os.path.join(args.data_path,'map_input','detection-results',f"{item['uid']}.txt")
        num_gt=len(item['nao_bbox'])
        num_pred=len(item['ro_bbox'])
        num_random_pick=5*num_gt
        if num_pred>0:
            idx_random_pick=np.random.randint(low=0,high=num_pred,size=num_random_pick).tolist()
        else:
            idx_random_pick =[]
        with open(gt_file,'w') as f:
            for idx_bbox,bbox in enumerate(item['nao_bbox']):
                f.write(f"{item['noun'][idx_bbox]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            f.close()
        # write prediction
        with open(pred_file,'w') as f:
            # picked_ro_bbox=[item['ro_bbox'][idx] for idx in idx_random_pick]
            for idx in idx_random_pick:
                bbox=item['ro_bbox'][idx]
                f.write(f"{item['cls'][idx]} {item['scores'][idx]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            f.close()