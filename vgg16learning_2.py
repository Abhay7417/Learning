# Implemented VGG16 architecture using tensorflow dataset api and tf.layers


import tensorflow as tf
import numpy as np
from os import listdir
from os.path import join
from sklearn.preprocessing import LabelEncoder

epochs = 5
batch_size = 32
display_progress = 1  # to check training progress

# defining the VGG16 network


def network(x):
    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        conv1_1 = tf.layers.conv2d(x, filters=64, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        conv1_2 = tf.layers.conv2d(conv1_1, filters=64, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)
                                   
    # pool1
    pool1 = tf.layers.max_pooling2d(conv1_2,
                                    pool_size=[2, 2],
                                    strides=2,
                                    padding='SAME',
                                    name='pool1')
    
    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        conv2_1 = tf.layers.conv2d(pool1, filters=128, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
        conv2_2 = tf.layers.conv2d(conv2_1, filters=128, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)

    # pool2
    pool2 = tf.layers.max_pooling2d(conv2_2,
                                    pool_size=[2, 2],
                                    strides=2,
                                    padding='SAME',
                                    name='pool2')
    
    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        conv3_1 = tf.layers.conv2d(pool2, filters=256, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        conv3_2 = tf.layers.conv2d(conv3_1, filters=256, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        conv3_3 = tf.layers.conv2d(conv3_2, filters=256, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation = tf.nn.relu, name=scope)

    # pool3
    pool3 = tf.layers.max_pooling2d(conv3_3,
                                    pool_size=[2, 2],
                                    strides=2,
                                    padding='SAME',
                                    name='pool3')

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        conv4_1 = tf.layers.conv2d(pool3, filters=512, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        conv4_2 = tf.layers.conv2d(conv4_1, filters=512, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        conv4_3 = tf.layers.conv2d(conv4_2, filters=512, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)

    # pool4
    pool4 = tf.layers.max_pooling2d(conv4_3,
                                    pool_size=[2, 2],
                                    strides=2,
                                    padding='SAME',
                                    name='pool4')

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        conv5_1 = tf.layers.conv2d(pool4, filters=512, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        conv5_2 = tf.layers.conv2d(conv5_1, filters=512, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        conv5_3 = tf.layers.conv2d(conv5_2, filters=512, kernel_size=3,
                                   strides=1, padding='SAME',
                                   activation=tf.nn.relu, name=scope)

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
        fc1 = tf.layers.dense(pool5_flat, 4096, activation=tf.nn.relu,
                              name=scope)

    # fc2
    with tf.name_scope('fc2') as scope:
        fc2 = tf.layers.dense(fc1, 4096, activation=tf.nn.relu,
                              name=scope)

    # fc3
    with tf.name_scope('fc3') as scope:
        fc3l = tf.layers.dense(fc2, 81, activation=None,
                               name=scope)

    return fc3l


# creating image paths list and labels list for training
train_image_paths = []
train_image_labels = []
mypath = 'Dataset/Training'
for f in listdir(mypath):
    for imagename in listdir(join(mypath, f)):
        # print(join(join(mypath, f),imagename))
        train_image_paths.append(join(join(mypath, f), imagename))
        train_image_labels.append(f)

# creating image paths list and labels list for testing
test_image_paths = []
test_image_labels = []
mypath = 'Dataset/Test'
for f in listdir(mypath):
    for imagename in listdir(join(mypath, f)):
        # print(join(join(mypath, f),imagename))
        test_image_paths.append(join(join(mypath, f), imagename))
        test_image_labels.append(f)

# labels encoder to convert class names to class labels
lb_make = LabelEncoder()
train_image_labels = lb_make.fit_transform(train_image_labels)
test_image_labels = lb_make.fit_transform(test_image_labels)


# Creating datasets using dataset api

def parser(filename, label):
    one_hot = tf.one_hot(label, 81)
    # read the img from file
    img_file = tf.read_file(filename)
    images = tf.image.decode_image(img_file, channels=3)
    images.set_shape([None, None, None])
    images = tf.image.resize_images(images, [224, 224])
    images.set_shape([224, 224, 3])
    d = dict(zip(['vgg16_input'], [images])), one_hot
    return d


train_data = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
train_data = train_data.map(parser)
train_data = train_data.shuffle(1000).repeat().batch(batch_size)

test_data = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))
test_data = test_data.map(parser)
test_data = test_data.shuffle(1000).repeat().batch(batch_size)

# create general iterator
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
next_element = iterator.get_next()

# make datasets that we can initialize separately, but using the same structure via the common iterator
training_init_op = iterator.make_initializer(train_data)
validation_init_op = iterator.make_initializer(test_data)

logits = network(next_element[0]['vgg16_input'])
# add the optimizer and loss
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1], logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(loss)
# get accuracy
prediction = tf.argmax(logits, 1)
equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
init_op = tf.global_variables_initializer()

# run the training
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(training_init_op)
    num_minibatches = int(len(train_image_paths)/batch_size)
    for i in range(epochs):
        epoch_cost = 0
        epoch_accuracy = 0
        for j in range(num_minibatches):
            l, _, acc = sess.run([loss, optimizer, accuracy])
            epoch_cost += l / num_minibatches
            epoch_accuracy += acc/ num_minibatches
        if i % display_progress == 0:
            print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, epoch_cost, epoch_accuracy * 100))
        # setting up the validation run
        valid_iters = int(len(test_image_paths)/batch_size)
        # re-initialize the iterator with validation data
        sess.run(validation_init_op)
        avg_acc = 0
        for k in range(valid_iters):
            acc = sess.run([accuracy])
            avg_acc += acc[0]
        print("Average validation set accuracy over {} iterations is {:.2f}%".format(valid_iters,
                                                                                     (avg_acc / valid_iters) * 100))
