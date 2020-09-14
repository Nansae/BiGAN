import cv2
import sys, os
import utils
import numpy as np
import tensorflow as tf
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time, datetime
from model import BIGAN
from config import Config
from utils import read_images

config = Config()
config.display()

#D = Data(config)
#D.LoadData()

model = BIGAN(config)
t_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(t_vars, print_info=True)

opt_D = tf.train.AdamOptimizer(learning_rate = 0.0001, beta1=0.5)
opt_G = tf.train.AdamOptimizer(learning_rate = 0.0001, beta1=0.5)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')):
    train_D = opt_D.minimize(model.loss_D, var_list=model.D_vars)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator') + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='encoder')):
    train_G = opt_G.minimize(model.loss_G, global_step=model.global_step, var_list=model.G_vars)

sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

with tf.Session(config=sess_config) as sess:
#with sv.managed_session(config=sess_config) as sess:
    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_writer = tf.summary.FileWriter(config.PATH_TENSORBOARD, sess.graph)

    # Load a previous checkpoint if desired
    model_checkpoint_name = config.PATH_CHECKPOINT + "/latest_model_" + config.PATH_DATA + ".ckpt"
    if config.IS_CONTINUE:
        print('Loaded latest model checkpoint')
        saver.restore(sess, model_checkpoint_name)

    images, labels = read_images("data", "folder")

    #num_iters = int(np.floor(len(D.train) / config.BATCH_SIZE))
    num_iters = int(np.floor(len(images) / config.BATCH_SIZE))

    length = 7
    valid_set = utils.generate_latent_points(100, length**2)
    valid_image = []
    for i in range(length**2):
        valid_image.append(images[i])

    for epoch in range(config.EPOCH):
        cnt=0
        st = time.time()
        epoch_st=time.time()

        #id_list = np.random.permutation(len(D.train))

        for i in range(num_iters):
            image_batch = []
            noise_batch = utils.generate_latent_points(100, config.BATCH_SIZE)
            #noise_batch = np.random.uniform(-1., 1., size=[config.BATCH_SIZE, 100])

            # Collect a batch of images
            for j in range(config.BATCH_SIZE):
                index = i*config.BATCH_SIZE+j
                #id = id_list[index]
                image_batch.append(images[index])
                            
            _, loss_D = sess.run([train_D, model.loss_D], feed_dict={model.image:image_batch, model.random_z:noise_batch})
            for i in range(5):
                _, loss_G = sess.run([train_G, model.loss_G], feed_dict={model.image:image_batch, model.random_z:noise_batch})  
            _, loss_G, global_step = sess.run([train_G, model.loss_G, model.global_step], feed_dict={model.image:image_batch, model.random_z:noise_batch})
                    
            cnt = cnt + config.BATCH_SIZE
            if cnt % 20 == 0:
                string_print = "Epoch = %d Count = %d Current_Loss_D = %.4f Current_Loss_G = %.4f Time = %.2f"%(epoch, cnt, loss_D, loss_G, time.time()-st)
                utils.LOG(string_print) 
                st = time.time()

            if global_step%config.SUMMARY_STEP == 0:
                summary_str = sess.run(summary_op, feed_dict={model.image:image_batch, model.random_z:noise_batch})
                train_writer.add_summary(summary_str, global_step)

            if global_step%config.PRINT_STEP == 0 or global_step is 0:
                print("Performing validation")
                results=None
                for idx in range(length):
                    #X = sess.run(model.generated, feed_dict={model.image:valid_set[length*idx:length*idx+length], model.random_z:valid_image[length*idx:length*idx+length]})
                    X = sess.run(model.generated, feed_dict={model.random_z:valid_set[length*idx:length*idx+length]})
                    X = (X+1)/2.0
                    if results is None:
                        results = X
                    else:
                        results = np.vstack((results, X))
                utils.save_plot_generated(results, length, "sample_data/" + str(global_step) + "_" + str(epoch) + "_gene_data.png")

        epoch_time=time.time()-epoch_st
        remain_time=epoch_time*(config.EPOCH-1-epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        if s!=0:
            train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
        else:
            train_time="Remaining training time : Training completed.\n"
        utils.LOG(train_time)
                        

        if epoch % config.CHECKPOINTS_STEP == 0:
            # Create directories if needed
            if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
                os.makedirs("%s/%04d"%("checkpoints",epoch))

            tf.logging.info('Saving model with global step %d ( = %d epochs) to disk' % (global_step, epoch))
            saver.save(sess, "%s/%04d/model.ckpt"%("checkpoints",epoch), global_step=global_step)

        # Save latest checkpoint to same file name
        tf.logging.info('Saving model with %d epochs to disk' % (epoch))
        saver.save(sess, model_checkpoint_name)
        
    tf.logging.info('complete training...')