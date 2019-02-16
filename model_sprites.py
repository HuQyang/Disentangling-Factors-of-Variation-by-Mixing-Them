import os
import time

import tensorflow as tf 

from ops_alex import *
from utils import *
from input_pipeline_rendered_data_sprites import get_pipeline_training_from_dump

import math
import numpy as np
import scipy.io as sio

class DCGAN(object):
    def __init__(self, sess,
                 batch_size=256, sample_size = 64, image_shape=[256, 256, 3],
                 y_dim=None, z_dim=0, gf_dim=128, df_dim=64,
                 gfc_dim=512, dfc_dim=1024, c_dim=3, cg_dim=1, is_train=True):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.model_name = "DCGAN.model"
        self.sess = sess
        self.batch_size = batch_size
        self.sample_size = sample_size

        self.image_shape = image_shape
        self.image_size = image_shape[0]
        
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim
        self.cg_dim = cg_dim


        self.d_bn1 = batch_norm(is_train, name='d_bn1')
        self.d_bn2 = batch_norm(is_train, name='d_bn2')
        self.d_bn3 = batch_norm(is_train, name='d_bn3')
        self.d_bn4 = batch_norm(is_train, name='d_bn4')

        self.c_bn1 = batch_norm(is_train, name='c_bn1')
        self.c_bn2 = batch_norm(is_train, name='c_bn2')
        self.c_bn3 = batch_norm(is_train, name='c_bn3')
        self.c_bn4 = batch_norm(is_train, name='c_bn4')
        self.c_bn5 = batch_norm(is_train, name='c_bn5')

        self.g_s_bn5 = batch_norm(is_train,convolutional=False, name='g_s_bn5')

        self.build_model(is_train)


    def build_model(self, is_train):
        
        self.abstract_size = self.sample_size // 2 ** 4

        _,_,images= get_pipeline_training_from_dump('data_example.tfrecords',
                                                                 self.batch_size*3,
                                                                 1000, image_size=60,resize_size=60,
                                                                 img_channels=self.c_dim)

        _,_,test_images1 = get_pipeline_training_from_dump('data_example.tfrecords',
                                                                 self.batch_size*2,
                                                                 10000000, image_size=60,resize_size=60,
                                                                 img_channels=self.c_dim)

        self.images = images[0:self.batch_size,:,:,:]
        self.imagesR = images[self.batch_size:self.batch_size*2,:,:,:]
        
        self.third_image = images[self.batch_size*2:self.batch_size*3,:,:,:]

        self.test_images1 = test_images1[0:self.batch_size,:,:,:]
        self.test_images2 = test_images1[self.batch_size:self.batch_size*2,:,:,:]


        self.chunk_num = 8
        self.chunk_size = 64
        self.feature_size = self.chunk_size*self.chunk_num
        

        with tf.variable_scope('generator') as scope: 

            self.rep = self.encoder(self.images)
            self.D_I = self.generator(self.rep)

            _ = self.classifier(self.D_I,self.D_I,self.D_I)

            scope.reuse_variables()
            self.repR = self.encoder(self.imagesR)
            self.D_IR = self.generator(self.repR)

            k = tf.random_uniform(shape=[self.chunk_num],minval=0,maxval=2,dtype=tf.int32)
            a_chunk = tf.ones((self.batch_size,self.chunk_size),dtype=tf.int32) 
            a_fea = tf.ones_like(self.rep,dtype=tf.int32)

            i=0
            t1 = self.rep[:,i*self.chunk_size:(i+1)*self.chunk_size]
            e1 = self.repR[:,i*self.chunk_size:(i+1)*self.chunk_size]
            self.fea = tf.where(tf.equal(k[0]*a_chunk,0),t1,e1)
            self.fea_mix = self.fea

            self.feaR = tf.where(tf.equal(k[0]*a_chunk,1),t1,e1)
            self.fea_mixR = self.feaR
    
            # mix the feature
            for i in xrange(1,self.chunk_num):
                t1 = self.rep[:,i*self.chunk_size:(i+1)*self.chunk_size]
                e1 = self.repR[:,i*self.chunk_size:(i+1)*self.chunk_size]
                self.fea = tf.where(tf.equal(k[i]*a_chunk,0),t1,e1)
                self.fea_mix = tf.concat(axis=1,values=[self.fea_mix,self.fea])

                self.feaR = tf.where(tf.equal(k[i]*a_chunk,1),t1,e1)
                self.fea_mixR = tf.concat(axis=1,values=[self.fea_mixR,self.feaR])


            self.k = k
            self.k0 = k[0]

            self.D_mix = self.generator(self.fea_mix)

            self.cf = self.classifier(self.images,self.imagesR,self.D_mix)

            self.kfc = tf.cast(tf.ones((self.batch_size,self.chunk_num),dtype=tf.int32)*k,tf.float32)

            self.D_mixR = self.generator(self.fea_mixR)

            self.rep_mix = self.encoder(self.D_mix)

            i = 0 
            tt = self.rep_mix[:,i*self.chunk_size:(i+1)*self.chunk_size]
            ee = self.rep[:,i*self.chunk_size:(i+1)*self.chunk_size]
            eeR = self.repR[:,i*self.chunk_size:(i+1)*self.chunk_size]
            self.rep_re = tf.where(tf.equal(k[i]*a_chunk,0),tt,ee)
            self.repR_re = tf.where(tf.equal(k[i]*a_chunk,1),tt,eeR)

            for i in xrange(1,self.chunk_num):
                tt = self.rep_mix[:,i*self.chunk_size:(i+1)*self.chunk_size]
                ee = self.rep[:,i*self.chunk_size:(i+1)*self.chunk_size]
                eeR = self.repR[:,i*self.chunk_size:(i+1)*self.chunk_size]
                self.rep_re = tf.concat(axis=1,values=[self.rep_re,tf.where(tf.equal(k[i]*a_chunk,0),tt,ee)]) 
                self.repR_re = tf.concat(axis=1,values=[self.repR_re,tf.where(tf.equal(k[i]*a_chunk,1),tt,eeR)]) 

            self.D_regenerate1 = self.generator(self.rep_re)

            self.D_regenerate2 = self.generator(self.repR_re)

            scope.reuse_variables()
            self.rep_test1 = self.encoder(self.test_images1)
            self.rep_test2 = self.encoder(self.test_images2)

            i = 0
            self.rep_test = self.rep_test2[:,0*self.chunk_size:1*self.chunk_size]
            for j in xrange(1,self.chunk_num):
                tmp = self.rep_test1[:,j*self.chunk_size:(j+1)*self.chunk_size]
                self.rep_test = tf.concat(axis=1,values=[self.rep_test,tmp])
            self.D_mix_allchunk = self.generator(self.rep_test,reuse=True)
            self.D_mix_allchunk_sup = self.D_mix_allchunk
            
            
            for i in xrange(1,self.chunk_num):
                self.rep_test = self.rep_test1[:,0*self.chunk_size:1*self.chunk_size]
                for j in xrange(1,self.chunk_num):
                    if j==i:
                        tmp = self.rep_test2[:,j*self.chunk_size:(j+1)*self.chunk_size]
                        self.rep_test = tf.concat(axis=1,values=[self.rep_test,tmp])
                    else:
                        tmp = self.rep_test1[:,j*self.chunk_size:(j+1)*self.chunk_size]
                        self.rep_test = tf.concat(axis=1,values=[self.rep_test,tmp])
                tmp_mix = self.generator(self.rep_test)
                self.D_mix_allchunk = tf.concat(axis=0,values=[self.D_mix_allchunk,tmp_mix])

            for i in xrange(1,self.chunk_num):
                self.rep_test = self.rep_test2[:,0*self.chunk_size:1*self.chunk_size]
                for j in xrange(1,self.chunk_num):
                    if j<=i:
                        tmp = self.rep_test2[:,j*self.chunk_size:(j+1)*self.chunk_size]
                        self.rep_test = tf.concat(axis=1,values=[self.rep_test,tmp])
                    else:
                        tmp = self.rep_test1[:,j*self.chunk_size:(j+1)*self.chunk_size]
                        self.rep_test = tf.concat(axis=1,values=[self.rep_test,tmp])
                tmp_mix = self.generator(self.rep_test)
                self.D_mix_allchunk_sup = tf.concat(axis=0,values=[self.D_mix_allchunk_sup,tmp_mix])

        with tf.variable_scope('classifier_loss') as scope:

            self.cf_loss = binary_cross_entropy_with_logits(self.kfc,self.cf)

        with tf.variable_scope('discriminator') as scope:

            self.D = self.discriminator(self.images)  

            self.D_ = self.discriminator(self.D_mix, reuse=True)

        with tf.variable_scope('discriminator_loss') as scope:
            self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
            self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)

            self.d_loss = self.d_loss_real + self.d_loss_fake


        with tf.variable_scope('generator_loss') as scope:

            self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)


        with tf.variable_scope('L2') as scope:

            self.rec_loss = tf.reduce_mean(tf.square(self.D_I - self.images))

            self.recR_loss = tf.reduce_mean(tf.square(self.D_IR - self.imagesR))

            self.rec_mix_loss = tf.reduce_mean(tf.square(self.D_regenerate1 - self.images))

            self.recR_mix_loss = tf.reduce_mean(tf.square(self.D_regenerate2 - self.imagesR))


        self.bn_assigners = tf.group(*batch_norm.assigners)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.g_s_vars = [var for var in t_vars if 'g_s' in var.name]
        self.g_e_vars = [var for var in t_vars if 'g_en' in var.name]
        self.c_vars = [var for var in t_vars if 'c_' in var.name]

        self.saver = tf.train.Saver(self.d_vars + self.g_vars + self.c_vars+
                                    batch_norm.shadow_variables,
                                    max_to_keep=0)


    def train(self, config, run_string="???"):
        """Train DCGAN"""

        if config.continue_from_iteration:
            counter = config.continue_from_iteration
        else:
            counter = 0

        global_step = tf.Variable(counter, name='global_step', trainable=False)

        # Learning rate of generator is gradually decreasing.
        self.g_lr = tf.train.exponential_decay(0.0002, global_step=global_step,
                                               decay_steps=20000, decay_rate=0.9, staircase=True)
        
        self.d_lr = tf.train.exponential_decay(0.0002, global_step=global_step,
                                               decay_steps=20000, decay_rate=0.9, staircase=True) 

        self.c_lr = tf.train.exponential_decay(0.0002, global_step=global_step,
                                               decay_steps=20000, decay_rate=0.9, staircase=True)

        labmda = 0 

        g_loss = lambda*self.rec_loss+10*self.recR_loss +10*self.rec_mix_loss+1*self.g_loss+1*self.cf_loss

        g_optim = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=config.beta1) \
                          .minimize(g_loss, var_list=self.g_vars)

        c_optim = tf.train.AdamOptimizer(learning_rate=self.c_lr, beta1=config.beta1) \
                          .minimize(self.cf_loss, var_list=self.c_vars)

        d_optim = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars, global_step=global_step)
        with tf.control_dependencies([g_optim]): 
            g_optim = tf.group(self.bn_assigners)

        tf.global_variables_initializer().run()
        if config.continue_from:
            checkpoint_dir = os.path.join(os.path.dirname(config.checkpoint_dir), config.continue_from)
            print('Loading variables from ' + checkpoint_dir)
            self.load(checkpoint_dir, config.continue_from_iteration)

        start_time = time.time()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord) 
        self.make_summary_ops()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.summary_dir, graph_def=self.sess.graph_def)
        
        try:
            # Training
            while not coord.should_stop():
                # Update D and G network
                tic = time.time()
                self.sess.run([g_optim])
                self.sess.run([c_optim]) 
                self.sess.run([d_optim]) 
                toc = time.time()
                counter += 1
                print(counter)
                duration = toc - tic
                
                if counter % 200 == 0:
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, counter) 

                if np.mod(counter, 4000) == 2:
                        
                    images,imagesR, D_mix, D_mix_allchunk,test_images1,test_images2, train_img,train_imgR,third_image,\
                    D_mix_allchunk_sup,D_re1,D_re2,D_mix_allchunk_re, D_mix_allchunk_sup_re = self.sess.run([self.images,self.imagesR,\
                        self.D_mix,self.D_mix_allchunk,self.test_images1,self.test_images2,\
                        self.D_I,self.D_IR,self.third_image,self.D_mix_allchunk_sup,self.D_regenerate1,self.D_regenerate2,\
                        self.D_mix_allchunk,self.D_mix_allchunk_sup])

                    grid_size = np.ceil(np.sqrt(self.batch_size))
                    grid = [grid_size, grid_size]
                    grid_celebA = [12, self.chunk_num+2]

                    save_images(images,grid, os.path.join(config.summary_dir, '%s_train_images.png' % counter))
                    save_images(imagesR, grid, os.path.join(config.summary_dir, '%s_train_imageR.png' % counter))
                    save_images(train_img,grid, os.path.join(config.summary_dir, '%s_train.png' % counter))
                    save_images(train_imgR, grid, os.path.join(config.summary_dir, '%s_trainR.png' % counter))
                    save_images(D_mix, grid, os.path.join(config.summary_dir, '%s_train_mix.png' % counter))
                    save_images(D_re1, grid, os.path.join(config.summary_dir, '%s_train_re1.png' % counter))
                    save_images(D_re2, grid, os.path.join(config.summary_dir, '%s_train_re2.png' % counter))

                    save_images_multi(test_images1,test_images2,D_mix_allchunk, grid_celebA,self.batch_size, os.path.join(config.summary_dir, '%s_test1.png' % counter))
                    save_images_multi(test_images1,test_images2,D_mix_allchunk_sup, grid_celebA,self.batch_size, os.path.join(config.summary_dir, '%s_test_sup1.png' % counter))
                    

                if np.mod(counter, 2000) == 0:
                    self.save(config.checkpoint_dir, counter)


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


    def discriminator(self, image,keep_prob=0.5, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(self.d_bn1(conv2d(image, self.df_dim, name='d_1_h0_conv')))
        h1 = lrelu(self.d_bn2(conv2d(h0, self.df_dim*2, name='d_1_h1_conv')))
        h2 = lrelu(self.d_bn3(conv2d(h1, self.df_dim*4, name='d_1_h2_conv')))
        h3 = lrelu(self.d_bn4(conv2d(h2, self.df_dim*8,k_h=1, k_w=1, d_h=1, d_w=1, name='d_1_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_1_h3_lin')

        return tf.nn.sigmoid(h4) 


    def classifier(self, image1,image2,image3,keep_prob=0.8, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        concated = tf.concat(axis=3, values=[image1, image2])
        concated = tf.concat(axis=3, values=[concated,image3])

        conv1 = self.c_bn1(conv(concated,96, 8,8,2,2, padding='VALID', name='c_3_s0_conv'))
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='c_3_mp0')

        conv2 = self.c_bn2(conv(pool1, 256, 5,5,1,1, groups=2, name='c_3_conv2'))
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='c_3_pool2')
        
        conv3 = self.c_bn3(conv(pool2, 384, 3, 3, 1, 1, name='c_3_conv3'))

        conv4 = self.c_bn4(conv(conv3, 384, 3, 3, 1, 1, groups=2, name='c_3_conv4'))

        conv5 = self.c_bn5(conv(conv4, 256, 3, 3, 1, 1, groups=2, name='c_3_conv5'))
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='c_3_pool5')

        fc6 = tf.nn.relu( linear(tf.reshape(pool5, [self.batch_size, -1]), 4096, 'c_3_fc6') )

        fc7 = tf.nn.relu( linear(tf.reshape(fc6, [self.batch_size, -1]), 4096, 'c_3_fc7') )

        self.fc8 = linear(tf.reshape(fc7, [self.batch_size, -1]), self.chunk_num, 'c_3_fc8')

        return tf.nn.sigmoid(self.fc8)


    def encoder(self, sketches_or_abstract_representations,keep_prob=0.9, reuse=False):            
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s0 = lrelu(instance_norm(conv2d((sketches_or_abstract_representations), self.df_dim,k_h=4, k_w=4, name='g_1_conv0')))
        s1 = lrelu(instance_norm(conv2d(s0, self.df_dim * 2,k_h=4, k_w=4, name='g_1_conv1')))
        s2 = lrelu(instance_norm(conv2d(s1, self.df_dim * 4,k_h=4, k_w=4, name='g_1_conv2')))
        s3 = lrelu(instance_norm(conv2d(s2, self.df_dim * 8,k_h=2, k_w=2, name='g_1_conv3')))
        s4 = lrelu(instance_norm(conv2d(s3, self.df_dim * 8,k_h=2, k_w=2, name='g_1_conv4')))
        used_abstract = lrelu((linear(tf.reshape(s4, [self.batch_size, -1]), self.feature_size, 'g_1_fc'))) 
        
        return used_abstract


    def generator(self, representations, reuse=False):
        if reuse:           
            tf.get_variable_scope().reuse_variables()            

        h = deconv2d(tf.reshape(representations,[self.batch_size,1,1,self.feature_size]), [self.batch_size,4,4, self.gf_dim*4],k_h=4, k_w=4, d_h=1, d_w=1,padding = 'VALID',name='g_de_h')
        h = tf.nn.relu((h))

        h1 = deconv2d(h, [self.batch_size, 8, 8, self.gf_dim*4 ], name='g_h1')
        h1 = tf.nn.relu(instance_norm(h1))

        h2 = deconv2d(h1, [self.batch_size, 15, 15, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(instance_norm(h2))

        h3 = deconv2d(h2, [self.batch_size, 30, 30, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(instance_norm(h3))

        h4 = deconv2d(h3, [self.batch_size, 60, 60, self.c_dim], name='g_h4') 

        return tf.nn.tanh(h4)
         
    def make_summary_ops(self):
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('classifier_loss',self.cf_loss)
        tf.summary.scalar('d_loss_fake', self.d_loss_fake)
        tf.summary.scalar('d_loss_real', self.d_loss_real)
        tf.summary.scalar('rec_loss', self.rec_loss)
        tf.summary.scalar('rec_mix_loss', self.rec_mix_loss)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir) 

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir, iteration=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and iteration:
            # Restores dump of given iteration
            ckpt_name = self.model_name + '-' + str(iteration)
        elif ckpt and ckpt.model_checkpoint_path:
            # Restores most recent dump
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)

        ckpt_file = os.path.join(checkpoint_dir, ckpt_name)
        print('Reading variables to be restored from ' + ckpt_file)
        self.saver.restore(self.sess, ckpt_file)
        return ckpt_name
