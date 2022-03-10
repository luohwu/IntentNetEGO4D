
import os

import cv2



def resize_images(input_folder,output_folder):
    clip_ids = [f for f in os.listdir(input_folder) if not os.path.isfile(os.path.join(input_folder, f))]
    for i,clip_id in enumerate(clip_ids):
        print(f'{i+1}/{len(clip_ids)} working on clip: {clip_id}')
        original_clip_folder=os.path.join(input_folder,clip_id)
        target_clip_folder=os.path.join(output_folder,clip_id)
        if not os.path.exists(target_clip_folder):
            os.mkdir(target_clip_folder)
        images=[image for image in os.listdir(original_clip_folder) ]
        for image in images:
            original_image=cv2.imread(os.path.join(original_clip_folder,image))
            resized_image=cv2.resize(original_image,(456,256))
            cv2.imwrite(os.path.join(target_clip_folder,image),resized_image)

if __name__=='__main__':
    import sys
    sys.path.insert(0, '..')
    from opt import *
    if args.euler:
        input_folder = '/cluster/work/hilliges/luohwu/nobackup/training/dataset/EGO4D/rgb_frames'
        output_folder = '/cluster/work/hilliges/luohwu/nobackup/training/dataset/EGO4D/rgb_frames_resized'
    else:
        input_folder = '/media/luohwu/T7/dataset/EGO4D/rgb_frames'
        output_folder = '/media/luohwu/T7/dataset/EGO4D/rgb_frames_resized'
    resize_images(input_folder=input_folder,
                  output_folder=output_folder)
