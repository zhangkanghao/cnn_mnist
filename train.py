from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical


def cnn_model(features):
    # Input Layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolution1 Layer #1 (30, 30) -> (32, 28, 28):padding='same'时,output_width=output_height=layer_size/strides
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding='same',
                             activation=tf.nn.relu, )

    # Pooling Layer #1 (32, 28, 28) -> (32, 14, 14)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolution Layer #2 (32, 14, 14) -> (64, 14, 14)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)

    # Pooling Layer #2 (64, 14, 14) -> (64, 7, 7)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flatten = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(pool2_flatten, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    # Logits Layer
    with tf.variable_scope('logits'):
        logits = tf.layers.dense(inputs=dropout, units=10,activation=None)

    return logits

    '''predictions={
        #Generate predictions for PREDICT and EVAL
        'classes': tf.argmax(input=logits,axis=1),
        'probabilities':tf.nn.softmax(logits,name='softmax_tensor')
    }

    #Configure Training Op
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions)

    #Calculate Loss(for train and eval mode)
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    #Configure the training Op
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.GradientDescentOptimizer(0.001)
        train_op=optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)

    #Configure evaluation metrics(for eval mode)
    eval_metrics_ops={
        'accuracy':tf.metrics.accuracy(labels=labels,predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metrics_ops)'''


def main(unused_argv):
    #parameter
    batch_size=64
    epoch=100

    # Load data
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')

    x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
    y = tf.placeholder(tf.int64, shape=[None, 10], name='target')

    pred_labels = cnn_model(x)

    with tf.name_scope('loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred_labels)
        loss = tf.reduce_mean(loss)

    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(pred_labels, axis=1), tf.argmax(y,axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # save
    logpath = 'logdir'
    train_writer = tf.summary.FileWriter(logpath)
    train_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # train
        for i in range(epoch):
            batch = mnist.train.next_batch(batch_size)
            train_data=batch[0]
            train_labels=to_categorical(batch[1],10)
            loss_rate = loss.eval(feed_dict={x: train_data, y: train_labels})
            print('step %d, loss rate %g' % (i, loss_rate))

            accuracy_rate = accuracy.eval(feed_dict={x: train_data, y: train_labels})
            print('step %d,accuracy rate %g' % (i, accuracy_rate))

            # save variables
            save_path = logpath + '/model_%d' % i
            print('%s' % save_path)
            saver.save(sess, save_path)
            train_step.run(feed_dict={x: train_data, y: train_labels})
    '''#Create estimator
    model_classifier=tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='logdir')

    #CReate logging
    tensor_to_log={'probabilities':'softmax_tensor'}
    logging_hook=tf.train.LoggingTensorHook(tensor_to_log,50)

    train_input=tf.estimator.inputs.numpy_input_fn(
        x={'x':train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    model_classifier.train(train_input,hooks=[logging_hook],steps=20000)

    eval_input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x':eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_result=model_classifier.evaluate(eval_input_fn)
    print(eval_result)'''

def predict():
    mnist=tf.contrib.learn.datasets.load_dataset('mnist')
    predict=mnist.test.next_batch(1)
    x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
    model=cnn_model(x)
    print(model)
    new_saver = tf.train.Saver()
    with tf.Session() as sess:
        new_saver.restore(sess,tf.train.latest_checkpoint('logdir'))
        y=sess.run(model,feed_dict={x:predict[0]})
        print('predict:',np.argmax(y[0]))
        print('real:',predict[1])

if __name__ == '__main__':
    # tf.app.run()
    predict()
    # sss()