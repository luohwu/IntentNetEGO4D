
import  os

import pandas as pd
import cv2
import argparse

from ast import literal_eval

def extract_frames_from_video(annotation_path,output_path,video_path,base_only):
    clip_ids=[f[:-4] for f in os.listdir(annotation_path) if os.path.isfile(os.path.join(annotation_path,f))]
    unavailable_clips=[]
    for i,clip_id in enumerate(clip_ids):

        print(f'{i+1}/{len(clip_ids)} working on clip {clip_id}')

        # mkdir output folder if not existed
        output_folder=os.path.join(output_path,clip_id)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # read annotations and prepare needed frames
        annotations=pd.read_csv(os.path.join(annotation_path,f'{clip_id}.csv'),converters={'all_frame':literal_eval})
        frame=annotations['all_frame'].tolist()
        frame=[item for block in frame for item in block]
        contract_frame=annotations['contact_frame'].tolist()
        frames_to_extract=list(set(frame+contract_frame))
        # in-place sort
        frames_to_extract.sort()

        # read videos and extract frames from clip
        video_file=os.path.join(video_path, f'{clip_id}.mp4')
        if not os.path.exists(video_file):
            print(f'clip not found {video_file}')
            unavailable_clips.append(clip_id)
            continue
        # else:
        #     continue

        vidcap=cv2.VideoCapture(video_file)
        assert vidcap.isOpened(),f"failed opening clip {video_file}"
        success=True
        count=0
        while success:
            success,image=vidcap.read()
            count+=1
            assert success, f"failed reading frame: {count} of clip {clip_id}"
            if count in frames_to_extract:
                image=cv2.resize(image,(456,256))
                cv2.imwrite(os.path.join(output_folder,f'frame_{str(count).zfill(10)}.jpg'),image)
                if count==frames_to_extract[-1]:
                    vidcap.release()
                    break

    print(f'unavailable clips:')
    print(unavailable_clips)


    if not args.ait:
        textfile = open("/media/luohwu/T7/ego4d/v1/missing_clips.txt", "w")
    else:
        textfile = open("/data/luohwu/dataset/EGO4D/missing_clips.txt", "w")
    for element in unavailable_clips:
        textfile.write(element + "\n")
    textfile.close()




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='extract frames from clips')
    parser.add_argument('--ait',default=False,action='store_true',
                        help='use euler cluster or not')
    args=parser.parse_args()
    if not args.ait:
        extract_frames_from_video(annotation_path='/home/luohwu/ait-data/dataset/EGO4D/nao_annotations',
                                  output_path='/home/luohwu/ait-data/dataset/EGO4D/rgb_frames',
                                  video_path='/media/luohwu/T7/ego4d/v1/clips',
                                  base_only=True)
    else:
        extract_frames_from_video(annotation_path='/data/luohwu/dataset/EGO4D/nao_annotations',
                                  output_path='/data/luohwu/dataset/EGO4D/rgb_frames',
                                  video_path='/data/luohwu/ego4d_data/v1/clips',
                                  base_only=True)






