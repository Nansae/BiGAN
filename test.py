from config import Config
from model import BIGAN
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time, datetime
#from data import Data
import utils
import os, csv
import sys, cv2
import numpy as np

#deviceList = ['/gpu:0','/gpu:1']
#deviceIdx = 0

config = Config()
config.display()

model = DCGAN(config.IMG_SIZE, config.IMG_SIZE, config.LATENT_DIM)
t_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(t_vars, print_info=True)

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))


with tf.Session(config=sess_config) as sess:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_writer = tf.summary.FileWriter(config.PATH_TENSORBOARD, sess.graph)

    # Load a previous checkpoint if desired
    model_checkpoint_name = config.PATH_CHECKPOINT + "/latest_model_" + config.PATH_DATA + ".ckpt"
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)