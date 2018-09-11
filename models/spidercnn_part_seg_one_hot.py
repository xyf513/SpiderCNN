import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from tf_grouping import query_ball_point, group_point, knn_point

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl, cls_labels_pl

NUM_CATEGORIES = 16

def get_model(xyz_withnor, cls_label, is_training, bn_decay=None, num_classes=50):
    batch_size = xyz_withnor.get_shape()[0].value
    num_point = xyz_withnor.get_shape()[1].value

    K_knn = 16
    taylor_channel = 3

    xyz = xyz_withnor[:, :, 0:3]
    
    with tf.variable_scope('delta') as sc:
        _, idx = knn_point(K_knn, xyz, xyz)
        
        grouped_xyz = group_point(xyz, idx)   
        point_cloud_tile = tf.expand_dims(xyz, [2])
        point_cloud_tile = tf.tile(point_cloud_tile, [1, 1, K_knn, 1])
        delta = grouped_xyz - point_cloud_tile

    with tf.variable_scope('SpiderConv1') as sc:
        feat_1 = tf_util.spiderConv(xyz_withnor, idx, delta, 32, taylor_channel = taylor_channel, 
                                        bn=True, is_training=is_training, bn_decay=bn_decay)

    with tf.variable_scope('SpiderConv2') as sc:
        feat_2 = tf_util.spiderConv(feat_1, idx, delta, 64, taylor_channel = taylor_channel, 
                                        bn=True, is_training=is_training, bn_decay=bn_decay)

    with tf.variable_scope('SpiderConv3') as sc:
        feat_3 = tf_util.spiderConv(feat_2, idx, delta, 128, taylor_channel = taylor_channel, 
                                        bn=True, is_training=is_training, bn_decay=bn_decay)

    with tf.variable_scope('SpiderConv4') as sc:
        feat_4 = tf_util.spiderConv(feat_3, idx, delta, 256, taylor_channel = taylor_channel, 
                                        bn=True, is_training=is_training, bn_decay=bn_decay)

    point_feat = tf.concat([feat_1, feat_2, feat_3, feat_4], 2)

    #top-k pooling
    global_fea = tf_util.topk_pool(point_feat, k = 2, scope='topk_pool')

    global_fea = tf.reshape(global_fea, [batch_size, -1])
    global_fea = tf.expand_dims(global_fea, 1)
    global_fea = tf.expand_dims(global_fea, 1)


    cls_label_one_hot = tf.one_hot(cls_label, depth=NUM_CATEGORIES, on_value=1.0, off_value=0.0)
    one_hot_label_expand = tf.reshape(cls_label_one_hot, [batch_size, 1, 1, NUM_CATEGORIES])
    global_fea = tf.concat([global_fea, one_hot_label_expand], 3)
    global_fea_expand = tf.tile(global_fea, [1, num_point, 1, 1])
    point_feat = tf.expand_dims(point_feat, 2)
    global_point_fea = tf.concat([global_fea_expand, point_feat], 3)
    
    net = tf_util.conv2d(global_point_fea, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                        bn=True, is_training=is_training, scope='seg/conv1')
    net = tf_util.dropout(net, keep_prob=0.8, is_training=is_training, scope='seg/dp1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                        bn=True, is_training=is_training, scope='seg/conv2')
    net = tf_util.dropout(net, keep_prob=0.8, is_training=is_training, scope='seg/dp2')
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
                        bn=True, is_training=is_training, scope='seg/conv3')
    net = tf_util.conv2d(net, num_classes, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
                        bn=False, scope='seg/conv4')

    net = tf.reshape(net, [batch_size, num_point, num_classes])

    return net


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss



if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        cls_labels = tf.zeros((32),dtype=tf.int32)
        output, ep = get_model(inputs, cls_labels, tf.constant(True))
        print(output)
