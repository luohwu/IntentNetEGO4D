import json
import os.path

import pandas as pd
import argparse



def calibrate_noun(row):
  noun=row['name']
  if '_' in noun:
    index=noun.index('_')
    return noun[:index]
  else:
    return noun

def extract_video_metadata():

  splits=['train','val']
  video_metadata_all={}
  for split in splits:
    with open(f'{data_folder_path}/v1/annotations/fho_sta_{split}.json', 'r') as f:
      data = json.load(f)
      video_metadata=data['info']['video_metadata']
      video_metadata_all.update(video_metadata)
    noun_categories=pd.DataFrame(data['noun_categories'])
    noun_categories['name']=noun_categories.apply(calibrate_noun,axis=1)
    noun_categories.to_csv(os.path.join(out_path,'noun_categories.csv'),index=False)

  with open(f'{data_folder_path}/v1/annotations/video_metadata.json','w') as f:
    f.write(json.dumps(video_metadata_all))

def extract_nao_annotations():
  splits=['train','val']
  video_metadata_all={}
  for split in splits:
    with open(f'{data_folder_path}/v1/annotations/fho_sta_{split}.json', 'r') as f:
      data = json.load(f)
      video_metadata=data['info']['video_metadata']
      annotations=data['annotations']
      noun_categoreis = data['noun_categories']
      verb_categoreis = data['verb_categories']
      annotations_df=pd.DataFrame(annotations)
      groups=tuple(annotations_df.groupby('clip_uid'))
      clip_ids=[]
      for clip_id,df in groups:
        if clip_id in ['e964bb42-f596-4dca-96de-0940b52f0c75', 'ded39483-8be8-4f8a-a3de-c86b86fd1e7c']:
          continue
        clip_ids.append(clip_id)
        video_id=df.iloc[0]['video_id']
        df[['noun','verb']]=df.apply(determined_noun,noun_map=noun_categoreis,verb_map=verb_categoreis,axis=1,result_type='expand')
        df[['nao_bbox','contact_frame']]=df.apply(determined_nao_bbox,video_info=video_metadata[video_id],axis=1,result_type='expand')
        df['contact_frame']=df.apply(lambda row: row['contact_frame']+row['clip_frame'],axis=1)
        helper=[*range(-72,3,3)]
        df['all_frame']=df.apply(lambda row: [max(1,row['clip_frame']+entry) for entry in helper],axis=1)
        df.to_csv(f"{out_path}/nao_annotations/{clip_id}.csv",index=False)

      #record each clip's split
      if not args.euler and not args.ait:
        textfile = open(f"/media/luohwu/T7/dataset/EGO4D/{split}_clips.txt", "w")
      else:
        textfile = open(f"/data/luohwu/dataset/EGO4D/{split}_clips.txt", "w")
      for element in clip_ids:
        textfile.write(element + "\n")
      textfile.close()



def determined_noun(row,noun_map:list,verb_map:list):
  nouns=[]
  verbs=[]
  objects=row['objects']
  for object in objects:
    noun_category_id=object['noun_category_id']
    noun_category=noun_map[noun_category_id]['name']
    if '_' in noun_category:
      index=noun_category.index('_')
      noun_category=noun_category[:index]
    nouns.append(noun_category)

    verb_category_id=object['verb_category_id']
    verb_category=verb_map[verb_category_id]['name']
    if '_' in verb_category:
      index=verb_category.index('_')
      verb_category=verb_category[:index]
    verbs.append(verb_category)
  return nouns,verbs

def determined_nao_bbox(row,video_info):
  video_width = video_info['frame_width']
  video_height = video_info['frame_height']
  fps=video_info['fps']
  nao_bbox_list=[]
  objects=row['objects']
  for object in objects:
    box=object['box']
    box_normalized=[box[0]/video_width,box[1]/video_height,box[2]/video_width,box[3]/video_height]
    nao_bbox_list.append(box_normalized)
    contact_frame=round(object['time_to_contact']*fps)
  return nao_bbox_list,contact_frame





if __name__=='__main__':
  parser = argparse.ArgumentParser(description='extract frames from clips')
  parser.add_argument('--euler', default=False, action='store_true',
                      help='use euler cluster or not')
  parser.add_argument('--ait', default=False, action='store_true',
                      help='use euler cluster or not')
  args = parser.parse_args()
  if args.ait == False:
    data_folder_path = '/media/luohwu/T7/ego4d'
    out_path = '/media/luohwu/T7/dataset/EGO4D'
  else:
    data_folder_path = '/data/luohwu/ego4d_data/'
    out_path = '/data/luohwu/dataset/EGO4D'


  extract_video_metadata()
  extract_nao_annotations()


