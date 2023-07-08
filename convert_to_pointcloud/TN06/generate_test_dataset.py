import os
from H5DataGenerator import *

# output dirs
OUT_ROOT_DIR = '../../../h5_dataset/TN06'
if not os.path.exists( OUT_ROOT_DIR ):
    os.mkdir(OUT_ROOT_DIR)

TRAIN_SET_DIR = os.path.join(OUT_ROOT_DIR, 'test')
if not os.path.exists( TRAIN_SET_DIR ):
    os.mkdir(TRAIN_SET_DIR)

# input dirs
IN_ROOT_DIR = '<your path>/Parametric_dataset/test_set/TemplateNum06'
DEPTH_DIR = os.path.join(IN_ROOT_DIR, 'depth_images')

if __name__ == "__main__":
    cycle_idx_list = range(0, 1)
    scene_idx_list = range(1, 61)
    obj_idx_list = range(0, 64)
    g = H5DataGenerator('./parameter.json')
    for cycle_id in cycle_idx_list:
        out_cycle_dir = os.path.join(TRAIN_SET_DIR, 'cycle_{:0>4}'.format(cycle_id))
        if not os.path.exists(out_cycle_dir):
            os.mkdir(out_cycle_dir)
        for scene_id in scene_idx_list:
            # load inputs
            for obj_id in obj_idx_list:
            	depth_image_path = os.path.join(DEPTH_DIR, 'cycle_{:0>4}'.format(cycle_id), '{}_{:0>3}'.format(obj_id,scene_id),'Image0001.png')
            	depth_image = cv2.imread(depth_image_path,cv2.IMREAD_UNCHANGED)
            	output_h5_path = os.path.join(out_cycle_dir, '{}_{:0>3}.h5'.format(obj_id,scene_id))
            	g.process_test_set(depth_image, output_h5_path)

