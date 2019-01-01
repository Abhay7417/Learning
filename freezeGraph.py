import tensorflow as tf
import numpy as np
from os import listdir
from os.path import join
from tensorflow.python.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from scipy.misc import imread, imresize
import sys, os
from imagenet_classes import class_names

print(os.listdir())
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
    
    with tf.name_scope('predictions') as scope:
        result = tf.nn.softmax(fc3l, name= scope)
    
    
    return result


weights = np.load('vgg16_weights.npz')
keys = sorted(weights.keys())
print(keys)
tf.reset_default_graph()
save_dir = './model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with tf.Session() as sess:
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = network(imgs)
    saver = tf.train.Saver()
    parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for i, k in enumerate(keys):
        print(i, k, np.shape(weights[k]))
        sess.run(parameters[i].assign(weights[k]))
    print([n.name for n in tf.get_default_graph().as_graph_def().node])
    saver.save(sess,save_dir)
    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))
    prob = sess.run(vgg, feed_dict={imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])

    print("loading saved model--")   
    saver.restore(sess=sess, save_path=save_dir)
    img2 = imread('test.png', mode='RGB')
    img2 = imresize(img2, (224, 224))
    prob = sess.run(vgg, feed_dict={imgs: [img2]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print(class_names[p], prob[p])



# freezing graph and creating .pb file

# freeze graph to load saved model and create .pb file

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)
        # We use a built-in TF helper to export variables to constants
        print([n.name for n in tf.get_default_graph().as_graph_def().node])
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


freeze_graph('model', 'predictions')

