'''
    Evaluate classification performance with optional voting.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import h5py
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import json
import show3d_balls

import sklearn
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
import random
import math
import eulerangles
import dataset_util_keypoint as dataset_util
import poseloss_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='parametricnet', help='Model name. [default: parametricnet]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=16384, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='h5_dataset/TN06/pretrained_model/pretrained.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


"""
---------------load dataset from h5 files---------------
"""
OUTPUT_DIR = './evaluate_results'
if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)
TRAIN_DATA_DIR = '../../../h5_dataset/TN06/train/'
vaild_dataset = dataset_util.load_dataset_by_cycle(TRAIN_DATA_DIR, range(29, 30), range(50,52), range(60, 61), 'test', collect_names=True )
print('vaild_dataset  loaded')

test_data = vaild_dataset['data'][::1,...] * 1000.0 # to mm
test_data_names = vaild_dataset['name']
print('test set',test_data.shape)

def radius_filter(org_points, num_p, rad):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(org_points)
    cl,index = pcd.remove_radius_outlier(nb_points=num_p,radius=rad)
    new_pcd = pcd.select_by_index(index)
    return np.asarray(new_pcd.points)


def quatWAvgMarkley(Q, weights=None):
    '''
    Averaging Quaternions.

    Arguments:
        Q(ndarray): an Mx4 ndarray of quaternions.
        weights(list): an M elements list, a weight for each quaternion.
    '''

    # Form the symmetric accumulator matrix
    A = np.zeros((4, 4))
    M = Q.shape[0]
    wSum = 0
    if weights==None:
        weights = [1.0 for _ in range(M)]

    for i in range(M):
        q = Q[i, :]
        w_i = weights[i]
        A += w_i * (np.outer(q, q)) # rank 1 update
        wSum += w_i

    # scale
    A /= wSum

    # Get the eigenvector corresponding to largest eigen value
    return np.linalg.eigh(A)[1][:, -1]

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''
    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    # print("H: \n", H)
    U, S, Vt = np.linalg.svd(H)
    # print(
    #     "U: \n", U, "\n",
    #     "S: \n", S, "\n",
    #     "V: \n", Vt, "\n",
    # )
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    # homogeneous transformation
    # T = np.identity(m+1)
    # T[:m, :m] = R
    # T[:m, m] = t
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    # R, _ = cv2.Rodrigues(R)
    # print(R.shape, t.shape)
    return  R, t, T # np.concatenate([R, t[:,None]], axis=-1)# T, R, t


def show_points(point_array, color_array=None, radius=3):
    assert isinstance(point_array, list)
    all_color = None
    if color_array is not None:
        assert len(point_array) == len(color_array)
        all_color = [ np.zeros( [ pnts.shape[0] ,3] ) for pnts in point_array]
        for i, c in enumerate(color_array):
            all_color[i][:] = [c[1],c[0],c[2]]
        all_color = np.concatenate(all_color, axis=0)
    all_points = np.concatenate(point_array, axis=0)
    show3d_balls.showpoints(all_points, c_gt=all_color, ballradius=radius)

def read_pcd(file_name, to_mm=True):
    with open(file_name, 'r') as f:
        begin = False
        points = []
        for line in f.readlines():
            if begin:
                xyz = list(map(float, line.strip().split()))
                if to_mm:
                    xyz[:3] = [ 1000*t for t in xyz[:3] ]
                points.append(xyz[:3])
            if line.startswith('DATA'):
                begin = True
    return np.array(points)

def extract_vertexes_from_obj(file_name):
    with open(file_name, 'r') as f:
        vertexes = []
        for line in f.readlines():
            line = line.strip()
            if line.startswith('v'):
                words = line.split()[1:]
                xyz = [float(w) for w in words]
                vertexes.append(xyz)
        ori_model_pc = np.array(vertexes)
        # center = ( np.max(ori_model_pc, axis=0) + np.min(ori_model_pc, axis=0) ) / 2.0
        # ori_model_pc = ori_model_pc - center
    return ori_model_pc

# model_pointcloud *= 1000.0

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, trans_labels_pl, kp_labels_pl, vs_labels_pl, param_labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Get model and loss.
        pred_trans_delta, pred_kp, pred_vs = MODEL.get_model_pointSIFT(pointclouds_pl, is_training_pl)
        pred_trans = pred_trans_delta + pointclouds_pl
        pointclouds_pl_3=tf.tile(pointclouds_pl,[1,1,3])

        pred_kp = pred_kp + pointclouds_pl_3

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
            'trans_labels_pl': trans_labels_pl,
            'is_training_pl': is_training_pl,
            'pred_kp': pred_kp,
            'pred_trans': pred_trans,
            'pred_vs': pred_vs,
            }

    eval_one_epoch(sess, ops, num_votes)

def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    is_training = False
    total_correct_5mm = 0
    total_seen = 0
    loss_sum = 0
    dist_mean_sum = 0

    log_string('----')

    idx = np.arange(NUM_POINT)
    np.random.shuffle(idx)
    current_data = test_data[:,idx,:]
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    obj_index=50
    for batch_idx in range(num_batches):
        MODEL_OBJ_DIR = './object_models/TN06_learning/TN06_'+'{}'.format(obj_index)+'.obj'
        model_pointcloud = extract_vertexes_from_obj(MODEL_OBJ_DIR)
        obj_index += 1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        output_file_name = test_data_names[start_idx]
        input_point = current_data[start_idx, :, :].copy()

        # z_noise = np.zeros(input_point.shape)
        # z_noise[:,2] = np.random.rand(input_point.shape[0]) * 1
        # point_data_with_noise = input_point + z_noise
        # show3d_balls.showpoints(point_data_with_noise, ballradius=5)
        # show3d_balls.showpoints(input_point, ballradius=5)


        time_start = time.time()

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['is_training_pl']: is_training,}
        pred_trans_val, pred_kp_val, pred_vs_val= sess.run([ops['pred_trans'], ops['pred_kp'], ops['pred_vs']],
                                      feed_dict=feed_dict)                           
        print("Forward time:", time.time()-time_start)

        # print('pred_trans_val', pred_trans_val.shape)
        # print('pred_kp_val', pred_kp_val.shape)
        # print('pred_vs_val', pred_vs_val.shape)
        pred_trans_val = pred_trans_val[0]
        pred_kp_val=pred_kp_val[0]
        pred_kp_val_center=pred_kp_val[:,:3]
        pred_kp_val_inr=pred_kp_val[:,3:6]
        pred_kp_val_outr=pred_kp_val[:,6:9]
        pred_kp_val_whole = np.concatenate((pred_kp_val_center,pred_kp_val_inr,pred_kp_val_outr))
        pred_vs_val = pred_vs_val[0,:,0]
        
        ms = MeanShift(bandwidth=10, bin_seeding=True, cluster_all=False, min_bin_freq=40)
        ms.fit(pred_trans_val)
        labels = ms.labels_
        n_clusters = len(set(labels))-(1 if -1 in labels else 0)
        print(n_clusters)

        color_cluster = [np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]) for i in range(n_clusters)]
        color_per_point = np.ones([pred_trans_val.shape[0], pred_trans_val.shape[1]]) * 255
        for idx in range(color_per_point.shape[0]):
            if labels[idx] != -1:
                color_per_point[idx, :] = color_cluster[labels[idx]]
        color_per_point_whole = np.concatenate((color_per_point,color_per_point,color_per_point))

        pred_trans_cluster = [[] for _ in range(n_clusters)]
        pred_center_cluster = [[] for _ in range(n_clusters)]
        pred_inr_cluster = [[] for _ in range(n_clusters)]
        pred_outr_cluster = [[] for _ in range(n_clusters)]
        pred_vs_cluster = [[] for _ in range(n_clusters)]
        for idx in range(NUM_POINT):
            if labels[idx] != -1:
                pred_trans_cluster[labels[idx]].append(np.reshape(pred_trans_val[idx], [1, 3]))
                # if labels_center[idx]!= -1:
                pred_center_cluster[labels[idx]].append(np.reshape(pred_kp_val_center[idx], [1, 3]))
                # if labels_inr[idx]!= -1:
                pred_inr_cluster[labels[idx]].append(np.reshape(pred_kp_val_inr[idx], [1, 3]))
                # if labels_ourt[idx]!= -1:
                pred_outr_cluster[labels[idx]].append(np.reshape(pred_kp_val_outr[idx], [1, 3]))
                pred_vs_cluster[labels[idx]].append(pred_vs_val[idx])

        n_clusters = len(pred_trans_cluster)
        pred_trans_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_trans_cluster]
        pred_center_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_center_cluster]
        pred_outr_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_outr_cluster]
        pred_inr_cluster = [np.concatenate(cluster, axis=0) for cluster in pred_inr_cluster]
        pred_vs_cluster = [ np.mean(l) for l in pred_vs_cluster]


       # Filter objects with vs lower than vs_threshold
        vs_threshold = 0.45
        valid_idxs = [ i for i, vs in enumerate(pred_vs_cluster) if vs > vs_threshold ]
        pred_trans_cluster = [ pred_trans_cluster[i] for i in valid_idxs]
        pred_center_cluster = [ pred_center_cluster[i] for i in valid_idxs]
        pred_inr_cluster = [ pred_inr_cluster[i] for i in valid_idxs]
        pred_outr_cluster = [ pred_outr_cluster[i] for i in valid_idxs]
        pred_vs_cluster = [ pred_vs_cluster[i] for i in valid_idxs]
        n_clusters = len(pred_vs_cluster)

        # Radius filter
        print('before radius:', np.array(pred_trans_cluster).shape)
        for cluster_id in range(n_clusters):
            # show3d_balls.showpoints(np.array(pred_trans_cluster[cluster_id]), ballradius=5) # before radius filter

            symbol = True
            filter_radius = 10
            while symbol:
                try:
                    pred_trans_cluster[cluster_id] = radius_filter(pred_trans_cluster[cluster_id], 20, filter_radius)
                    pred_center_cluster[cluster_id] = radius_filter(pred_center_cluster[cluster_id], 20, filter_radius)
                    pred_outr_cluster[cluster_id] = radius_filter(pred_outr_cluster[cluster_id], 20, filter_radius)
                    pred_inr_cluster[cluster_id] = radius_filter(pred_inr_cluster[cluster_id], 20, filter_radius)
                    symbol = False
                except:
                    filter_radius += 2
                    if filter_radius>20:
                        symbol = False

            # show3d_balls.showpoints(np.array(pred_trans_cluster[cluster_id]), ballradius=5) # after radius filter

        pred_trans_cluster = [ np.mean(cluster, axis=0) for cluster in pred_trans_cluster]
        pred_center_cluster = [ np.mean(cluster, axis=0) for cluster in pred_center_cluster]
        pred_inr_cluster = [ np.mean(cluster, axis=0) for cluster in pred_inr_cluster]
        pred_outr_cluster = [ np.mean(cluster, axis=0) for cluster in pred_outr_cluster]
        pred_kp_val_mean = np.concatenate((pred_center_cluster,pred_inr_cluster,pred_outr_cluster))

        cluster_mat_pred = []

        all_model_point = np.zeros([model_pointcloud.shape[0]*n_clusters, 3])
        all_model_color = np.zeros([model_pointcloud.shape[0]*n_clusters, 3])

        vs_filter = 0.9*np.max(np.array(pred_vs_cluster))
        centroid = np.array(pred_trans_cluster)[np.array(pred_vs_cluster)>vs_filter]
        center = np.array(pred_center_cluster)[np.array(pred_vs_cluster)>vs_filter]
        # pred_center2_cluster = 2*np.array(pred_trans_cluster) - np.array(pred_center_cluster)
        inr = np.array(pred_inr_cluster)[np.array(pred_vs_cluster)>vs_filter]
        # pred_inr2_cluster = 2*np.array(pred_center_cluster) - np.array(pred_inr_cluster)
        outr = np.array(pred_outr_cluster)[np.array(pred_vs_cluster)>vs_filter]
        # pred_outr2_cluster = 2*np.array(pred_center_cluster) - np.array(pred_outr_cluster)
        r = np.mean(np.sqrt(np.sum(np.square(inr-center),axis=1)))
        d = np.mean(np.sqrt(np.sum(np.square(outr-center),axis=1)))
        h_half = np.mean(np.sqrt(np.sum(np.square(centroid-center),axis=1)))
        mdl_pnts = [[0,0,h_half],[d,0,h_half], [r,0,h_half],[0,0,0]]

        mdl_pnts = np.array(mdl_pnts)

        for cluster_idx in range(n_clusters):
            pred_pnts = [pred_center_cluster[cluster_idx],pred_outr_cluster[cluster_idx],pred_inr_cluster[cluster_idx],\
                pred_trans_cluster[cluster_idx]]
            pred_pnts = np.array(pred_pnts)

            pred_R, pred_t, _ = best_fit_transform(mdl_pnts, pred_pnts) 
            cluster_mat_pred.append(pred_R)
            begin_idx = cluster_idx * model_pointcloud.shape[0]
            end_idx = (cluster_idx+1) * model_pointcloud.shape[0]
            all_model_color[begin_idx:end_idx, :] = color_cluster[cluster_idx]
            all_model_point[begin_idx:end_idx, :] = np.dot(model_pointcloud, pred_R.T) + \
                                                    np.tile(np.reshape(pred_trans_cluster[cluster_idx], [1, 3]), [model_pointcloud.shape[0], 1])

        # vs_threshold = 0.45
        pred_results = list(zip(pred_vs_cluster, pred_trans_cluster, cluster_mat_pred))
        # pred_results = [ rst for rst in pred_results if rst[0] > vs_threshold ]
        pred_results.sort(key=lambda x:x[0], reverse=True)

        result_list = []
        for rst in pred_results:
            tmp_dict = {}
            tmp_dict['score'] = 1.0 * rst[0] 
            tmp_dict['t'] = (rst[1] / 1000.0).tolist()
            tmp_dict['R'] = rst[2].tolist()
            result_list.append(tmp_dict)

        with open(os.path.join(OUTPUT_DIR, output_file_name+'.json'),"w") as f:
            json.dump(result_list,f)
            print(output_file_name+' saved')

        # Visualization:
        show3d_balls.showpoints(input_point, ballradius=5)  # Input pointcloud
        show_points([input_point,pred_kp_val_mean],[[255,255,255],[255,0,0]],5)  # Final prediction of 3 keypoints 
        show3d_balls.showpoints(pred_trans_val, c_gt=color_per_point, ballradius=5)  # Centroid clustering results
        show3d_balls.showpoints(input_point, c_gt=color_per_point, ballradius=5)  # Instance segmentation results
        show_points([input_point,pred_kp_val_center[labels!=-1]],[[255,255,255],[255,1,0]],5) # Prediction of kp1
        show_points([input_point,pred_kp_val_inr[labels!=-1],pred_kp_val_outr[labels!=-1]],[[255,255,255],[255,1,0],[255,1,0]],5) # Prediction of kp2
        show_points([input_point,pred_kp_val_outr[labels!=-1]],[[255,255,255],[255,1,0]],5) # Prediction of kp3
        show_points([input_point,all_model_point],[[1,1,1],[1,0,0]],5)  # Pose estimation


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
