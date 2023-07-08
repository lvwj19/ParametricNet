import tensorflow as tf 

def get_trans_loss(pred_trans, trans_label, weight=None):
    '''
        input:
            pred_trans: n,3
            trans_label: n,3
    '''
    # l1 loss
    x = tf.abs(trans_label - pred_trans) # shape: n,3
    x = tf.reduce_sum(x, axis=-1) # shape: n
    per_loss = x
    if weight is not None:
        x = tf.multiply(x, tf.reshape(weight, [-1]))
    loss = tf.reduce_mean(x)
    return loss, per_loss

def get_rot_loss_finite(rot_matrix, rot_label, lambda_p, G, weight=None):
    '''
        input:
            rot_matrix: n,3,3
            rot_label: n,3,3
            lambda_p: 3,3
            G : list of 3,3
    '''
    n = rot_matrix.shape.as_list()[0]
    c = len(G)
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
    dist = tf.reduce_sum(sub, axis=-1) # shape: n,c
    min_dist = tf.reduce_min(dist, axis=-1) # shape: n
    per_loss = min_dist

    if weight is not None:
        min_dist = tf.multiply(min_dist, tf.reshape(weight, [-1]))

    #loss = tf.reduce_mean(min_dist)
    loss=min_dist
    loss = tf.reduce_mean(min_dist)
    return loss, per_loss

def get_rot_loss_revolution(rot_matrix, rot_label, lambda_p, retoreflection=False, weight=None):
    '''
        input:
            rot_matrix: n,3,3
            rot_label: n,3,3
            lambda_p: 1
            retoreflection : bool
    '''
    ez = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32, shape=[1,3,1]) # shape: 1,3,1
    ez = tf.tile(ez, [rot_matrix.shape.as_list()[0],1,1]) # shape: n,3,1
    if retoreflection==False:
        loss = lambda_p * tf.abs(tf.matmul(rot_matrix, ez) - tf.matmul(rot_label, ez)) # shape: n,3,1
        loss = tf.reshape(loss,[-1,3]) # shape: n,3
        loss = tf.reduce_sum(loss, axis=-1) # shape: n
        dist = loss
        if weight is not None:
            loss = tf.multiply(loss, tf.reshape(weight, [-1]))
        loss = tf.reduce_mean(loss) # shape: 1
    else:
        loss1 = lambda_p * tf.abs(tf.matmul(rot_matrix, ez) - tf.matmul(rot_label, ez)) # shape: n,3,1
        loss2 = lambda_p * tf.abs(tf.matmul(rot_matrix, ez) + tf.matmul(rot_label, ez)) # shape: n,3,1
        loss = tf.concat([loss1, loss2], axis=-1) # shape: n,3,2
        loss = tf.transpose(loss, [0,2,1]) # shape: n,2,3
        loss = tf.reduce_sum(loss, axis=-1) # shape: n,2
        loss = tf.reduce_min(loss, axis=-1) # shape: n
        per_loss = loss
        if weight is not None:
            loss = tf.multiply(loss, tf.reshape(weight, [-1]))
        loss = tf.reduce_mean(loss) # shape: 1
    return loss, per_loss


def get_conf_loss(conf_pred, conf_label):
    conf_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits = conf_pred,labels = conf_label)
    conf_loss=tf.reduce_mean(conf_loss)
    return conf_loss

if __name__=='__main__':
    import numpy as np
    with tf.Graph().as_default():
        rot_mat_pl = tf.placeholder(tf.float32, [2,3,3])
        rot_label_pl = tf.placeholder(tf.float32, [2,3,3])
        lambda_p = tf.constant([[1,0,0], [0,1,0], [0,0,1]], dtype=tf.float32)
        G = [
            tf.constant([[1,0,0], [0,1,0], [0,0,1]], dtype=tf.float32),
            tf.constant([[-1,0,0], [0,-1,0], [0,0,-1]], dtype=tf.float32)
        ]
        loss = get_rot_loss_finite(rot_mat_pl, rot_label_pl, lambda_p, G)
        print(loss)
    
        sess = tf.Session()
        pred = np.ones([2,3,3])
        label = np.ones([2,3,3])
        label[0,...]*=0.9
        label[1,...]*=-1.0
        l = sess.run(loss, feed_dict={rot_mat_pl:pred, rot_label_pl:label})
        print(l)
