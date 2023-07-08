import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from pointSIFT_util import pointSIFT_module, pointSIFT_res_module, pointnet_fp_module, pointnet_sa_module

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    trans_labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    kp_labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    vs_label_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    param_labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, trans_labels_pl, kp_labels_pl, vs_label_pl, param_labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz = point_cloud
    l0_points = None

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=4096, radius=30, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=1024, radius=60, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=256, radius=120, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=240, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

    # trans net
    # FC layers
    net_trans = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_trans', bn_decay=bn_decay)
    net_trans = tf_util.conv1d(net_trans, 3, 1, padding='VALID', activation_fn=None, scope='fc2_trans')

    # rot net
    # FC layers
    net_rot = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_rot', bn_decay=bn_decay)
    net_rot = tf_util.conv1d(net_rot, 3, 1, padding='VALID', activation_fn=None, scope='fc2_rot')

    # vs
    net_vs = tf_util.conv1d(l0_points, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_vs', bn_decay=bn_decay)
    net_vs = tf_util.conv1d(net_vs, 1, 1, padding='VALID', activation_fn=tf.nn.sigmoid, scope='fc2_vs')

    return net_trans, net_rot, net_vs

def get_model_pointSIFT(point_cloud, is_training, bn_decay=None):
    """ Semantic segmentation PointNet, input is B x N x 3, output B x num_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    l0_xyz = point_cloud
    l0_points = None

    my_radius = 30

    # c0
    c0_l0_xyz, c0_l0_points, c0_l0_indices = pointSIFT_res_module(l0_xyz, l0_points, radius=my_radius, out_channel=64, is_training=is_training, bn_decay=bn_decay, scope='layer0_c0', merge='concat')
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(c0_l0_xyz, c0_l0_points, npoint=1024, radius=my_radius, nsample=32, mlp=[64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')

    # c1
    c0_l1_xyz, c0_l1_points, c0_l1_indices = pointSIFT_res_module(l1_xyz, l1_points, radius=my_radius*2, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='layer1_c0')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(c0_l1_xyz, c0_l1_points, npoint=256, radius=my_radius*2, nsample=32, mlp=[128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    # c2
    c0_l2_xyz, c0_l2_points, c0_l2_indices = pointSIFT_res_module(l2_xyz, l2_points, radius=my_radius*4, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='layer2_c0')
    c1_l2_xyz, c1_l2_points, c1_l2_indices = pointSIFT_res_module(c0_l2_xyz, c0_l2_points, radius=my_radius*4, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='layer2_c1', same_dim=True)
    l2_cat_points = tf.concat([c0_l2_points, c1_l2_points], axis=-1)
    fc_l2_points = tf_util.conv1d(l2_cat_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='conv_2_fc', bn_decay=bn_decay)

    # c3
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(c1_l2_xyz, fc_l2_points, npoint=64, radius=my_radius*8, nsample=32, mlp=[512,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512,512], is_training, bn_decay, scope='fa_layer2')
    _, l2_points_1, _ = pointSIFT_module(l2_xyz, l2_points, radius=my_radius*4, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c0')
    _, l2_points_2, _ = pointSIFT_module(l2_xyz, l2_points, radius=my_radius*4, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c1')
    _, l2_points_3, _ = pointSIFT_module(l2_xyz, l2_points, radius=my_radius*4, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c2')

    l2_points = tf.concat([l2_points_1, l2_points_2, l2_points_3], axis=-1)
    l2_points = tf_util.conv1d(l2_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='fa_2_fc', bn_decay=bn_decay)

    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,256], is_training, bn_decay, scope='fa_layer3')
    _, l1_points_1, _ = pointSIFT_module(l1_xyz, l1_points, radius=my_radius*2, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3_c0')
    _, l1_points_2, _ = pointSIFT_module(l1_xyz, l1_points_1, radius=my_radius*2, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3_c1')
    l1_points = tf.concat([l1_points_1, l1_points_2], axis=-1)
    l1_points = tf_util.conv1d(l1_points, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fa_1_fc', bn_decay=bn_decay)

    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')
    _, l0_points, _ = pointSIFT_module(l0_xyz, l0_points, radius=my_radius, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='fa_layer4_c0')

    # trans net
    # FC layers
    net_trans = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_trans', bn_decay=bn_decay)

    net_trans = tf_util.conv1d(net_trans, 3, 1, padding='VALID', activation_fn=None, scope='fc2_trans')

    # kypoint net
    # FC layers
    net_kp = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_kp', bn_decay=bn_decay)

    net_kp = tf_util.conv1d(net_kp, 9, 1, padding='VALID', activation_fn=None, scope='fc2_kp')
 
    # vs
    net_vs = tf_util.conv1d(l0_points, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_vs', bn_decay=bn_decay)
    net_vs = tf_util.conv1d(net_vs, 1, 1, padding='VALID', activation_fn=tf.nn.sigmoid, scope='fc2_vs')

    return net_trans, net_kp, net_vs

def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, 
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

def get_trans_loss(pred_trans, trans_label, huber_delta = 0.5):
    # smooth l1 loss
    x = tf.abs(trans_label - pred_trans)
    # print(x)
    # exit(0)
    x = tf.where(x < huber_delta, 0.5 * x ** 2, huber_delta * (x - 0.5 * huber_delta))
    return tf.reduce_mean(x)

def exp_map(axis_angle, scope):
    with tf.variable_scope(scope) as sc:
        zero_tensor = tf.zeros([tf.shape(axis_angle)[0], 1, 1])
        r1 = tf.expand_dims(axis_angle[:, 0], -1)
        r1 = tf.expand_dims(r1, -1)
        r2 = tf.expand_dims(axis_angle[:, 1], -1)
        r2 = tf.expand_dims(r2, -1)
        r3 = tf.expand_dims(axis_angle[:, 2], -1)
        r3 = tf.expand_dims(r3, -1)

        tmp1 = tf.concat([zero_tensor, tf.scalar_mul(-1, r3), r2], axis=2)
        tmp2 = tf.concat([r3, zero_tensor, tf.scalar_mul(-1, r1)], axis=2)
        tmp3 = tf.concat([tf.scalar_mul(-1, r2), r1, zero_tensor], axis=2)
        skew_mat_tensor = tf.concat([tmp1, tmp2, tmp3], axis=1)
        print('skew_mat_tensor', skew_mat_tensor.shape)
        theta = tf.norm(axis_angle, axis=1)
        tf.summary.scalar("theta_0", theta[0])
        # print "theta : \n", theta.eval()
        mask_mat = tf.greater(theta, 0.0001)
        print('mask_mat', mask_mat.shape)
        
        positive_mask = tf.reshape(tf.cast(mask_mat, dtype=tf.float32), [-1,1])
        print('positive_mask', positive_mask.shape)
        negative_mask = tf.reshape(tf.cast(tf.logical_not(mask_mat), dtype=tf.float32), [-1,1])
        # print "mask : \n", positive_mask.eval()
        theta = tf.expand_dims(theta, -1)#Bx1
        print('tf.sin(theta)/theta', (tf.sin(theta)/theta).shape)
        k1 = tf.multiply(positive_mask, tf.sin(theta)/theta) + \
             tf.multiply(negative_mask, 1 - tf.pow(theta, tf.constant(2.0))/tf.constant(6.0) +
                       tf.pow(theta, tf.constant(4.0))/tf.constant(120.0) -
                       tf.pow(theta, tf.constant(6.0))/tf.constant(5040.0))
        
        k2 = tf.multiply(positive_mask, (1 - tf.cos(theta)) / tf.square(theta)) - \
             tf.multiply(negative_mask, 0.5 - tf.pow(theta, tf.constant(2.0))/tf.constant(24.0) +
                       tf.pow(theta, tf.constant(4.0))/tf.constant(720.0) -
                       tf.pow(theta, tf.constant(6.0))/tf.constant(40320.0))
        
        # theta = tf.expand_dims(theta, -1)
        k1 = tf.expand_dims(k1, -1)
        k2 = tf.expand_dims(k2, -1)
        print('k1', k1.shape)
        print('k2', k2.shape)
        # exit(0)
        eye_tensor = tf.eye(3, batch_shape=[tf.shape(axis_angle)[0]])
        r = eye_tensor + \
            k1 * skew_mat_tensor + \
            k2 * tf.matmul(skew_mat_tensor, skew_mat_tensor)
        return r

def euler_to_matrix(pred):
    # pred : B x 3
    rot_x = pred[:, :1]
    rot_y = pred[:, 1:2]
    rot_z = pred[:, 2:3]
    cos_rot_x = tf.cos(rot_x)
    sin_rot_x = tf.sin(rot_x)
    cos_rot_y = tf.cos(rot_y)
    sin_rot_y = tf.sin(rot_y)
    cos_rot_z = tf.cos(rot_z)
    sin_rot_z = tf.sin(rot_z)
    one = tf.ones_like(cos_rot_x, dtype=tf.float32)
    zero = tf.zeros_like(cos_rot_x, dtype=tf.float32)

    rot_x = tf.stack([tf.concat([one, zero, zero], axis=1),
                      tf.concat([zero, cos_rot_x, sin_rot_x], axis=1),
                      tf.concat([zero, -sin_rot_x, cos_rot_x], axis=1)], axis=1)

    rot_y = tf.stack([tf.concat([cos_rot_y, zero, -sin_rot_y], axis=1),
                      tf.concat([zero, one, zero], axis=1),
                      tf.concat([sin_rot_y, zero, cos_rot_y], axis=1)], axis=1)

    rot_z = tf.stack([tf.concat([cos_rot_z, sin_rot_z, zero], axis=1),
                      tf.concat([-sin_rot_z, cos_rot_z, zero], axis=1),
                      tf.concat([zero, zero, one], axis=1)], axis=1)

    # rot_matrix = tf.matmul(rot_z, tf.matmul(rot_y, rot_x))
    rot_matrix = tf.matmul(rot_x, tf.matmul(rot_y, rot_z))

    return rot_matrix

def get_rot_loss(rot_matrix, rot_label):
    loss = tf.reduce_mean(1-tf.sigmoid((tf.trace(tf.matmul(rot_matrix, rot_label, transpose_b=True)) - 1.0) / 2.0))

    return loss

def get_trans_loss_new(pred_trans, trans_label):
    # l1 loss
    x = tf.abs(trans_label - pred_trans)
    return tf.reduce_mean(x)

def get_rot_loss_new(rot_matrix, rot_label, lambda_p, G):
    n = rot_matrix.shape.as_list()[0]
    c = len(G)
    # G = [ g for g in G ] 
    # G = [ tf.expand_dims(g, 0) for g in G ] 
    # print(G[0])
    # tf.tile(G[0], [n,1,1])
    # G = [ tf.tile(g, [n,1,1]) for g in G ] 
    G = [ tf.tile(tf.expand_dims(g, 0), [n,1,1]) for g in G ] # shape: n,3,3

    lambda_p = tf.reshape(lambda_p, [-1,3,3]) # shape: 1,3,3
    lambda_p = tf.tile(lambda_p, [n,1,1]) # shape: n,3,3

    P = tf.matmul(rot_matrix, lambda_p) # shape: n,3,3
    P = tf.expand_dims(P, -1) # shape: n,3,3,1
    P = tf.tile(P, [1,1,1,c]) # shape: n,3,3,c

    L_list = []
    for i in range(c):
        l = tf.matmul(tf.matmul(rot_label, G[i]), lambda_p) # shape: n,3,3
        L_list.append(tf.expand_dims(l, -1)) # shape: n,3,3,1
    L = tf.concat(L_list, axis=-1) # shape: n,3,3,c

    sub = tf.transpose(tf.abs(P - L), [0,3,1,2]) # shape: n,c,3,3
    sub = tf.reshape(sub, [n,c,9]) # shape: n,c,9
    dist = tf.reduce_mean(sub, axis=-1) # shape: n,c
    min_dist = tf.reduce_min(dist, axis=-1) # shape: n

    loss = tf.reduce_mean(min_dist)

    return loss


if __name__=='__main__':
    with tf.Graph().as_default():
        rot_mat_pl = tf.placeholder(tf.float32, [2,3,3])
        rot_label_pl = tf.placeholder(tf.float32, [2,3,3])
        lambda_p = tf.constant([[1,0,0], [0,1,0], [0,0,1]], dtype=tf.float32)
        G = [
            tf.constant([[1,0,0], [0,1,0], [0,0,1]], dtype=tf.float32),
            tf.constant([[-1,0,0], [0,-1,0], [0,0,-1]], dtype=tf.float32)
        ]
        loss = get_rot_loss_new(rot_mat_pl, rot_label_pl, lambda_p, G)
        print(loss)
    
        sess = tf.Session()
        pred = np.ones([2,3,3])
        label = np.ones([2,3,3])
        label[0,...]*=0.9
        label[1,...]*=-0.8
        l = sess.run(loss, feed_dict={rot_mat_pl:pred, rot_label_pl:label})
        print(l)


