import tensorflow as tf
import numpy as np
from os import listdir
from os.path import join
from tensorflow.python.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from scipy.misc import imread, imresize
import sys
sys.path.append('../input')
from imagenet_classes import class_names

# defining the VGG16 network


def network(x):
    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        conv1_1 = tf.layers.conv2d(x, filters=64, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        conv1_2 = tf.layers.conv2d(conv1_1, filters=64, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)
                                   
    # pool1
    pool1 = tf.layers.max_pooling2d(conv1_2,
                                    pool_size=[2, 2],
                                    strides=2,
                                    padding='SAME',
                                    name='pool1')
    
    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        conv2_1 = tf.layers.conv2d(pool1, filters=128, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
        conv2_2 = tf.layers.conv2d(conv2_1, filters=128, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # pool2
    pool2 = tf.layers.max_pooling2d(conv2_2,
                                    pool_size=[2, 2],
                                    strides=2,
                                    padding='SAME',
                                    name='pool2')
    
    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        conv3_1 = tf.layers.conv2d(pool2, filters=256, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        conv3_2 = tf.layers.conv2d(conv3_1, filters=256, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        conv3_3 = tf.layers.conv2d(conv3_2, filters=256, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # pool3
    pool3 = tf.layers.max_pooling2d(conv3_3,
                                    pool_size=[2, 2],
                                    strides=2,
                                    padding='SAME',
                                    name='pool3')

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        conv4_1 = tf.layers.conv2d(pool3, filters=512, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        conv4_2 = tf.layers.conv2d(conv4_1, filters=512, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        conv4_3 = tf.layers.conv2d(conv4_2, filters=512, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # pool4
    pool4 = tf.layers.max_pooling2d(conv4_3,
                                    pool_size=[2, 2],
                                    strides=2,
                                    padding='SAME',
                                    name='pool4')

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        conv5_1 = tf.layers.conv2d(pool4, filters=512, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        conv5_2 = tf.layers.conv2d(conv5_1, filters=512, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        conv5_3 = tf.layers.conv2d(conv5_2, filters=512, kernel_size = 3,
                                   strides = 1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # pool5
    pool5 = tf.layers.max_pooling2d(conv5_3,
                                    pool_size=[2, 2],
                                    strides=2,
                                    padding='SAME',
                                    name='pool5')

    # fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc1 = tf.layers.dense(pool5_flat, 4096, activation = tf.nn.relu,
                              name = scope)

    # fc2
    with tf.name_scope('fc2') as scope:
        fc2 = tf.layers.dense(fc1, 4096, activation = tf.nn.relu,
                              name = scope)

    # fc3
    with tf.name_scope('fc3') as scope:
        fc3l = tf.layers.dense(fc2, 1000, activation = None,
                               name = scope)

    return fc3l


weights = np.load('../input/vgg16_weights.npz')
keys = sorted(weights.keys())
print(keys)
tf.reset_default_graph()
with tf.Session() as sess:
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = network(imgs)
    parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for i, k in enumerate(keys):
        print(i, k, np.shape(weights[k]))
        sess.run(parameters[i].assign(weights[k]))
    img1 = imread('../input/laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))
    probs = tf.nn.softmax(vgg)
    prob = sess.run(probs, feed_dict={imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])
