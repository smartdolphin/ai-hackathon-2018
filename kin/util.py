import os

import tensorflow as tf

PATH = '/tmp/kin/models'

def local_save(sess, epoch, *args):
    os.makedirs(PATH, exist_ok=True)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(PATH, str(epoch)))

    train_writer = tf.summary.FileWriter(os.path.join(PATH, str(epoch)),
            sess.graph)

def local_load(sess):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(PATH)
    if ckpt and ckpt.model_checkpoint_path:
        checkpoint = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(PATH, checkpoint))
    else:
        raise NotImplemented('No checkpoint!')
    print('Model loaded')
