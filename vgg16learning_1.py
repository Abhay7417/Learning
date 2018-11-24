# Implements VGG16 architecture using tensorflow dataset api with tf.nn

import tensorflow as tf
import numpy as np
from os import listdir
from os.path import join
from tensorflow.python.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

epochs = 5
batch_size = 32
display_progress = 10  # to check training progress

# defining the VGG16 network


def network(x):
    parameters = []
    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # pool1
    pool1 = tf.nn.max_pool(conv1_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # pool2
    pool2 = tf.nn.max_pool(conv2_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # pool3
    pool3 = tf.nn.max_pool(conv3_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool3')

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # pool4
    pool4 = tf.nn.max_pool(conv4_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool4')

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
                             dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope)
        parameters += [kernel, biases]

    # pool5
    pool5 = tf.nn.max_pool(conv5_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool4')

    # fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)
        parameters += [fc1w, fc1b]

    # fc2
    with tf.name_scope('fc2') as scope:
        fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)
        parameters += [fc2w, fc2b]

    # fc3
    with tf.name_scope('fc3') as scope:
        fc3w = tf.Variable(tf.truncated_normal([4096, 81],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc3b = tf.Variable(tf.constant(1.0, shape=[5], dtype=tf.float32),
                           trainable=True, name='biases')
        fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
        parameters += [fc3w, fc3b]

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

# function to generate batches


def input_val_fn(filenames, labels=None, shuffle=False,
                 repeat_count=1, batch_size=1):
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

    if labels is None:
        labels = np.zeros((len(filenames), 81))
    labels = np.array(labels)
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    labels = tf.cast(labels, tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parser)
    if shuffle:
        # Randomizes input using a window of 1000 elements
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat(None)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

# creating training batch element


next_batch = input_val_fn(train_image_paths, labels=train_image_labels,
                          shuffle=True, batch_size=32)
# creating test batch element
next_batch_test = input_val_fn(test_image_paths, labels=test_image_labels,
                               shuffle=True, batch_size=32)

# checking the batch generator function
with tf.Session() as sess:
    first_batch = sess.run(next_batch)
x = first_batch[0]['vgg16_input']

# to confirm iterator is working correctly
print(x.shape)
img = image.array_to_img(x[5])
img

# creating placeholders for input image and label
num_classes = 81
input_image_height = 224
input_image_width = 224
n_channels = 3
x = tf.placeholder(tf.float32, [None, input_image_height,
                                input_image_width, n_channels])
y = tf.placeholder(tf.float32, [None, num_classes])

# to check logits and labels are tensor of same shape
predictions = network(x)
print(predictions)

# defining cost, optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                      labels=y, logits=predictions))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# calculating acuracy and accuracy percentage
correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy_pct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

# calling global varibles initializer
initializer_op = tf.global_variables_initializer()

# creating training session along with testing accuracy
with tf.Session() as sess:
    sess.run(initializer_op)

    print("Training for", epochs, "epochs.")

    # loop over epochs:
    for epoch in range(epochs):

        avg_cost = 0.0  # track cost to monitor performance during training
        avg_accuracy_pct = 0.0

        # loop over all batches of the epoch:
        n_batches = int(len(train_image_paths) / batch_size)
        for i in range(n_batches):

            # to reassure you something's happening!
            if i % display_progress == 0:
                print("Step ", i+1, " of ", n_batches, " in epoch ",
                      epoch+1, ".", sep='')

            first_batch = sess.run(next_batch)
            batch_x = first_batch[0]['vgg16_input']
            batch_y = first_batch[1]

            # feed batch data to run optimization and get cost and accuracy:
            _, batch_cost, batch_acc = sess.run([optimizer, cost, accuracy_pct],
                                                feed_dict={x: batch_x, y: batch_y})

            # accumulate mean loss and accuracy over epoch:
            avg_cost += batch_cost / n_batches
            avg_accuracy_pct += batch_acc / n_batches

        # output logs at end of each epoch of training:
        print("Epoch ", '%03d' % (epoch+1),
              ": cost = ", '{:.3f}'.format(avg_cost),
              ", accuracy = ", '{:.2f}'.format(avg_accuracy_pct), "%",
              sep='')
    test_batch = sess.run(next_batch_test)
    print("Training Complete. Testing Model.\n")

    test_cost = cost.eval({x: test_batch[0]['vgg16_input'], y: test_batch[1]})
    test_accuracy_pct = accuracy_pct.eval({x: test_batch[0]['vgg16_input'],
                                          y: test_batch[1]})

    print("Test Cost:", '{:.3f}'.format(test_cost))
    print("Test Accuracy: ", '{:.2f}'.format(test_accuracy_pct), "%", sep='')
