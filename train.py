'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import eulerangles
import dataset_util_keypoint as dataset_util
import poseloss_util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='parametricnet', help='Model name [default: parametricnet]')
parser.add_argument('--log_dir', default='logs', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=16384, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) 
os.system('cp train_keypoint.py %s' % (LOG_DIR)) 
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


# ---------------load dataset from h5 files---------------
TRAIN_DATA_DIR = '../../../h5_dataset/TN06/train/'
vaild_dataset = dataset_util.load_dataset_by_cycle(TRAIN_DATA_DIR, range(28, 29), range(0,64), range(1, 61), 'train')
print('vaild_dataset loaded')

validation_data = vaild_dataset['data'] * 1000.0 # to mm
validation_label_trans = vaild_dataset['trans_label'] * 1000.0 # to mm
validation_label_kp = vaild_dataset['kp_label']* 1000.0 # to mm
validation_label_param = vaild_dataset['param_label']* 1000.0 # to mm
validation_label_vs = vaild_dataset['vs_label']


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # Clip the learning rate!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, trans_labels_pl, kp_labels_pl, vs_labels_pl, param_labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred_trans_delta, pred_kp, pred_vs = MODEL.get_model_pointSIFT(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            pred_trans = pred_trans_delta + pointclouds_pl
            pred_trans_reshape = tf.reshape(pred_trans, [-1, 3])

            # Calculate kp_pred
            pointclouds_pl_3=tf.tile(pointclouds_pl,[1,1,3])
            pred_kp = pred_kp + pointclouds_pl_3
            pred_kp_reshape = tf.reshape(pred_kp, [-1, 9])
            pred_vs_reshape = tf.reshape(pred_vs, [-1])
            trans_label_reshape = tf.reshape(trans_labels_pl, [-1, 3])

            # Calculate param_pred r,d,h
            pred_r= tf.sqrt(tf.reduce_sum(tf.square(pred_kp_reshape[:,3:6] - pred_kp_reshape[:,:3]), axis=-1))
            pred_r = tf.reshape(pred_r, [-1,1])
            pred_d = tf.sqrt(tf.reduce_sum(tf.square(pred_kp_reshape[:,6:] - pred_kp_reshape[:,:3]), axis=-1))
            pred_d = tf.reshape(pred_d, [-1,1])
            pred_h = 2*tf.sqrt(tf.reduce_sum(tf.square(pred_trans_reshape - pred_kp_reshape[:,:3]), axis=-1))
            pred_h = tf.reshape(pred_h, [-1,1])
            pred_param = tf.concat([pred_r,pred_h,pred_d],1)
            pred_param_reshape = tf.reshape(pred_param, [-1,3])
            kp_label_reshape = tf.reshape(kp_labels_pl, [-1, 9])
            vs_label_reshape = tf.reshape(vs_labels_pl, [-1])
            param_label_reshape = tf.reshape(param_labels_pl, [-1,3])

            # Calculate the loss
            trans_loss = poseloss_util.get_trans_loss(pred_trans_reshape/1000.0, trans_label_reshape/1000.0) * 200.0
            kp_loss_fixed = poseloss_util.get_trans_loss(pred_kp_reshape/1000.0, kp_label_reshape/1000.0) * 200.0
            kp_loss_rot = tf.constant(0.0)
            kp_loss=kp_loss_fixed + kp_loss_rot
            vs_loss = tf.reduce_mean( tf.abs(vs_label_reshape - pred_vs_reshape) ) * 50
            param_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(pred_param_reshape/1000.0-param_label_reshape/1000.0),axis=-1)) * 200.0
            total_loss = trans_loss + kp_loss + vs_loss + param_loss

            tf.summary.scalar('trans_loss', trans_loss)
            tf.summary.scalar('kp_loss', kp_loss)
            tf.summary.scalar('kp_loss_fixed', kp_loss_fixed)
            tf.summary.scalar('kp_loss_rot', kp_loss_rot)
            tf.summary.scalar('vs_loss', vs_loss)
            tf.summary.scalar('param_loss', param_loss)
            tf.summary.scalar('total_loss', total_loss)

            dist_5mm = tf.cast(tf.norm(pred_trans_reshape-trans_label_reshape, axis=1)<5,dtype=tf.int32)
            num_in_5mm = tf.reduce_sum(dist_5mm)
            dist_mean=trans_loss + kp_loss + param_loss

            tf.summary.scalar('num_in_5mm', num_in_5mm)

            print ("--- Get training operator")
            # Get training operator.
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers.
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        validation_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'validation'), sess.graph)

        # Init variables.
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'trans_labels_pl': trans_labels_pl,
               'kp_labels_pl': kp_labels_pl,
               'param_labels_pl': param_labels_pl,
               'vs_labels_pl': vs_labels_pl,
               'is_training_pl': is_training_pl,
               'pred_kp': pred_kp_reshape,
               'loss': total_loss,
               'kp_loss': kp_loss,
               'param_loss': param_loss,
               'kp_loss_fixed': kp_loss_fixed,
               'kp_loss_rot': kp_loss_rot,
               'trans_loss': trans_loss,
               'vs_loss': vs_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'num_in_5mm': num_in_5mm,
               'dist_mean': dist_mean}

        min_dist = 1e16
        train_scenes =[ [1,5], [5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,61]]
        load_epoch_gap = 2
        for epoch in range(MAX_EPOCH):
            if epoch % load_epoch_gap == 0:
                """
                ---------------load dataset from h5 files---------------
                """
                cid = int(epoch/load_epoch_gap) % len(train_scenes)
                train_dataset = dataset_util.load_dataset_by_cycle(TRAIN_DATA_DIR,range(0, 28), range(0,64), range(train_scenes[cid][0], train_scenes[cid][1]), 'train')
                print('train_dataset  loaded')

                train_data = train_dataset['data'] * 1000.0 # to mm
                train_label_trans = train_dataset['trans_label'] * 1000.0 # to mm
                train_label_kp = train_dataset['kp_label']* 1000.0 # to mm
                train_label_param = train_dataset['param_label']* 1000.0 # to mm
                train_label_vs = train_dataset['vs_label']

                print('train set',train_data.shape)
                print('validation set',validation_data.shape)
                """
                ---------------load dataset from h5 files---------------
                """

            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer, train_data, train_label_trans, train_label_kp, train_label_vs, train_label_param)
            dist = eval_one_epoch(sess, ops, 'validation', validation_writer)

            # Save the variables to disk.
            if dist < min_dist:
                min_dist = dist
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_%d.ckpt"%epoch))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer, train_data, train_label_trans, train_label_kp, train_label_vs, train_label_param):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label_trans, current_label_kp, current_label_vs,current_label_param, _ = provider.shuffle_data(train_data, train_label_trans, train_label_kp, train_label_vs,train_label_param) 
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct_5mm = 0
    total_seen = 0
    loss_sum = 0
    kp_loss_sum = 0
    kp_loss_fixed_sum = 0
    kp_loss_rot_sum = 0
    trans_loss_sum = 0
    vs_loss_sum = 0
    param_loss_sum = 0
    dist_mean_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 200 == 0:
            log_string('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        idx = np.arange(NUM_POINT)
        np.random.shuffle(idx)

        point_batch = current_data[start_idx:end_idx, idx, :]
        xyz_noise = np.random.standard_normal(point_batch.shape)
        point_data_with_noise = point_batch + xyz_noise*0.5
        kp_batch = current_label_kp[start_idx:end_idx, idx, :]
        param_batch = current_label_param[start_idx:end_idx, idx, :]
        pos_batch = current_label_trans[start_idx:end_idx, idx, :]
        vs_batch = current_label_vs[start_idx:end_idx, idx]

        # # Domain randomization
        # z_noise = np.zeros(origin_point_data.shape)
        # z_noise[:,:,2] = np.random.standard_normal(origin_point_data.shape[0:2])
        # all_noise = np.random.standard_normal(point_batch.shape) * 2.0
        # point_batch_with_noise = point_batch + all_noise

        feed_dict = {ops['pointclouds_pl']: point_data_with_noise,
                     ops['trans_labels_pl']: pos_batch,
                     ops['kp_labels_pl']: kp_batch,
                     ops['vs_labels_pl']: vs_batch,
                     ops['param_labels_pl']:param_batch, 
                     ops['is_training_pl']: is_training,
                     }
        summary, step, _, loss_val, num_in_5mm, dist_mean_val, kp_loss_val, trans_loss_val, vs_loss_val, pred_kp_val,kp_loss_fixed_val,kp_loss_rot_val, param_loss_val= sess.run([ops['merged'], ops['step'], ops['train_op'], \
                                        ops['loss'],ops['num_in_5mm'], ops['dist_mean'], ops['kp_loss'], ops['trans_loss'], ops['vs_loss'], ops['pred_kp'],ops['kp_loss_fixed'],ops['kp_loss_rot'],ops['param_loss']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        correct_5mm = float(num_in_5mm)
        total_correct_5mm += correct_5mm
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        kp_loss_sum += kp_loss_val
        vs_loss_sum += vs_loss_val
        trans_loss_sum += trans_loss_val
        kp_loss_fixed_sum += kp_loss_fixed_val
        kp_loss_rot_sum += kp_loss_rot_val
        param_loss_sum += param_loss_val
        dist_mean_sum += dist_mean_val

        if batch_idx % 200 == 0:
            tran_loss_cur = trans_loss_sum/(batch_idx+1)
            kp_loss_cur = kp_loss_sum/(batch_idx+1)
            kp_loss_rot_cur = kp_loss_rot_sum/(batch_idx+1)
            kp_loss_fixed_cur = kp_loss_fixed_sum/(batch_idx+1)
            vs_loss_cur = vs_loss_sum/(batch_idx+1)
            param_loss_cur = param_loss_sum/(batch_idx+1)
            dist_mean_cur = dist_mean_sum/(batch_idx+1)
            log_string('trans_loss: %f\tkp_loss: %f\tkp_loss_fixed: %f\tkp_loss_rot: %f\tparam_loss: %f\tvs_loss: %f\tmean_dist: %f'%(tran_loss_cur,kp_loss_cur,kp_loss_fixed_cur,kp_loss_rot_cur,param_loss_cur, vs_loss_cur,dist_mean_cur))
    
    log_string('mean translation loss: %f' % (trans_loss_sum / float(num_batches)))
    log_string('mean keypoint loss: %f' % (kp_loss_sum / float(num_batches)))
    log_string('eval keypoint loss fixed: %f' % (kp_loss_fixed_sum / float(num_batches))) 
    log_string('eval keypoint loss rot: %f' % (kp_loss_rot_sum / float(num_batches))) 
    log_string('mean vs loss: %f' % (vs_loss_sum / float(num_batches)))
    # log_string('mean total loss: %f' % (loss_sum / float(num_batches)))
    log_string('mean dist: %f' % (dist_mean_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct_5mm / float(total_seen)))

        
def eval_one_epoch(sess, ops, dataset_name, writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct_5mm = 0
    total_seen = 0
    loss_sum = 0
    kp_loss_rot_sum = 0

    trans_loss_sum = 0
    vs_loss_sum = 0
    dist_mean_sum = 0
    kp_loss_fixed_sum = 0
    kp_loss_sum = 0
    param_loss_sum = 0

    log_string('----')
    if dataset_name == 'test':
        current_data = test_data[:,0:NUM_POINT,:]
        current_label_trans = test_label_trans
        current_label_kp = test_label_kp
        current_label_param = test__label_param
        current_label_vs = test_label_vs

    if dataset_name == 'validation':
        current_data = validation_data[:,0:NUM_POINT,:]
        current_label_trans = validation_label_trans
        current_label_kp = validation_label_kp
        current_label_param = validation_label_param
        current_label_vs = validation_label_vs


    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        idx = np.arange(NUM_POINT)
        np.random.shuffle(idx)

        point_batch = current_data[start_idx:end_idx, idx, :]
        xyz_noise = np.random.standard_normal(point_batch.shape)
        point_data_with_noise = point_batch + xyz_noise*0.5
        param_batch = current_label_param[start_idx:end_idx, idx, :]
        kp_batch = current_label_kp[start_idx:end_idx, idx, :]
        pos_batch = current_label_trans[start_idx:end_idx, idx, :]
        vs_batch = current_label_vs[start_idx:end_idx, idx]

        # # Domain randomization
        # z_noise = np.zeros(origin_point_data.shape)
        # z_noise[:,:,2] = np.random.standard_normal(origin_point_data.shape[0:2])
        # all_noise = np.random.standard_normal(point_batch.shape) * 1.0
        # point_batch_with_noise = point_batch + all_noise

        feed_dict = {ops['pointclouds_pl']: point_data_with_noise,
                     ops['trans_labels_pl']: pos_batch,
                     ops['kp_labels_pl']: kp_batch,
                     ops['vs_labels_pl']: vs_batch,
                     ops['param_labels_pl']:param_batch, 
                     ops['is_training_pl']: is_training,}
    
        summary, step, loss_val, num_in_5mm, dist_mean_val, kp_loss_val, trans_loss_val, vs_loss_val ,kp_loss_fixed_val,kp_loss_rot_val,param_loss_val= \
                                      sess.run([ops['merged'], ops['step'], ops['loss'],\
                                      ops['num_in_5mm'], ops['dist_mean'], ops['kp_loss'], ops['trans_loss'], ops['vs_loss'],ops['kp_loss_fixed'],ops['kp_loss_rot'],ops['param_loss']],
                                      feed_dict=feed_dict)
        writer.add_summary(summary, step)
        correct_5mm = float(num_in_5mm)
        total_correct_5mm += correct_5mm
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        kp_loss_sum += kp_loss_val
        kp_loss_fixed_sum += kp_loss_fixed_val
        kp_loss_rot_sum += kp_loss_rot_val
        trans_loss_sum += trans_loss_val
        vs_loss_sum += vs_loss_val
        param_loss_sum += param_loss_val
        dist_mean_sum += dist_mean_val

    log_string('**********'+dataset_name+' set**************')
    log_string('eval translation loss: %f' % (trans_loss_sum / float(num_batches)))
    log_string('eval keypoint loss: %f' % (kp_loss_sum / float(num_batches)))    
    log_string('eval keypoint loss fixed: %f' % (kp_loss_fixed_sum / float(num_batches))) 
    log_string('eval keypoint loss rot: %f' % (kp_loss_rot_sum / float(num_batches))) 
    log_string('eval vs loss: %f' % (vs_loss_sum / float(num_batches)))  
    log_string('eval param loss loss: %f' % (param_loss_sum / float(num_batches)))
    log_string('eval dist: %f' % (dist_mean_sum / float(num_batches)))
    log_string('eval accuracy: %f'% (total_correct_5mm / float(total_seen)))

    return (dist_mean_sum / float(num_batches))

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()

