import os
from H5DataGenerator import *

# output dirs
OUT_ROOT_DIR = '../../../h5_dataset/TN06'
if not os.path.exists( OUT_ROOT_DIR ):
    os.mkdir(OUT_ROOT_DIR)

TRAIN_SET_DIR = os.path.join(OUT_ROOT_DIR, 'train')
if not os.path.exists( TRAIN_SET_DIR ):
    os.mkdir(TRAIN_SET_DIR)

# input dirs
IN_ROOT_DIR = '<your path>/Parametric_dataset/training_set/TemplateNum06'
GT_DIR = os.path.join(IN_ROOT_DIR, 'gt')
SEGMENT_DIR = os.path.join(IN_ROOT_DIR, 'segment_images')
DEPTH_DIR = os.path.join(IN_ROOT_DIR, 'depth_images')


if __name__ == "__main__":
    cycle_idx_list = range(0, 30)
    g = H5DataGenerator('./parameter.json')
    for cycle_id in cycle_idx_list:
        out_cycle_dir = os.path.join(TRAIN_SET_DIR, 'cycle_{:0>4}'.format(cycle_id))
        if not os.path.exists(out_cycle_dir):
            os.mkdir(out_cycle_dir)
        scene_path = os.path.join(GT_DIR,'cycle_{:0>4}'.format(cycle_id))
        scene_names = os.listdir(scene_path)
        for scene_id, s_name in enumerate(scene_names):
            # load inputs
            depth_image_path = os.path.join(DEPTH_DIR, 'cycle_{:0>4}'.format(cycle_id), s_name.split('.')[0],'Image0001.png')

            depth_image = cv2.imread(depth_image_path,cv2.IMREAD_UNCHANGED)
            seg_img_path = os.path.join(SEGMENT_DIR, 'cycle_{:0>4}'.format(cycle_id), s_name.split('.')[0],'Image0001.exr')
            segment_image = cv2.imread(seg_img_path,cv2.IMREAD_UNCHANGED)
            gt_file_path = os.path.join(GT_DIR, 'cycle_{:0>4}'.format(cycle_id), s_name)
            output_h5_path = os.path.join(out_cycle_dir,  s_name.split('.')[0]+'.h5')
            obj_num = int(s_name.split('.')[0].split('_')[-1])
            g.process_train_set(obj_num,depth_image, segment_image, gt_file_path, output_h5_path)

