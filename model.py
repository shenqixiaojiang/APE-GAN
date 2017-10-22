from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=28, input_width=28, crop=False,
         batch_size=64, sample_num = 64, output_height=64, output_width=64, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, dataset_name='mnist',checkpoint_dir=None,sample_dir=None, 
         adversarial_path=None,ground_truth_path=None,test_path=None,save_path=None):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.adversarial_path = adversarial_path
    assert(self.adversarial_path != None)
    self.ground_truth_path = ground_truth_path
    assert(self.ground_truth_path != None)
    
    if test_path != None:
        self.test_path = test_path
    else:
        self.test_path = adversarial_path

    if save_path != None:
	self.save_path = save_path
    else:
	self.save_path = "./data/resAPE-GAN.npy"

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    self.dataset_name = dataset_name
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir

    self.adv = np.load(self.adversarial_path)
    
    if self.dataset_name == "mnist":
	self.c_dim = 1
    else:
	self.c_dim = 3
    
    self.grayscale = (self.c_dim == 1)
    self.gt = np.load(self.ground_truth_path) 
    self.build_model()

  def build_model(self):
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]
    
    self.gtInputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='ground_truth_images')
      
    self.advInputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='adversarial_images')
    
    self.z_sum = histogram_summary("z", self.advInputs)
    
    self.G                  = self.generator(self.advInputs)
    self.D, self.D_logits   = self.discriminator(self.gtInputs, reuse=False)
    self.sampler            = self.sampler(self.advInputs)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)
    
    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    
    self.g_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    self.mse_loss = tf.sqrt(tf.reduce_mean(tf.pow(tf.subtract(self.gtInputs, self.G), 2)))
    
    self.d_loss = self.d_loss_real + self.d_loss_fake
    self.g_loss = 0.02 * self.g_loss_real + 0.9 * self.mse_loss
    
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.g_loss_real_sum = scalar_summary("g_loss_real", self.g_loss_real)
    self.mse_loss_sum    = scalar_summary("mse_loss",    self.mse_loss)
    
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
    
    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    
    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum,self.g_loss_real_sum,self.mse_loss_sum])
      
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_adv = self.adv[0:self.sample_num]
    sample_gt  = self.gt[0:self.sample_num]

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      batch_idxs = min(len(self.adv), config.train_size) // config.batch_size
      for idx in xrange(0, batch_idxs):
        batch_adv = self.adv[idx*config.batch_size:(idx+1)*config.batch_size]
        batch_gt = self.gt[idx*config.batch_size:(idx+1)*config.batch_size]
        # Update D network
        _, summary_str = self.sess.run([d_optim,self.d_sum],
        feed_dict={ 
          self.gtInputs: batch_gt,
          self.advInputs: batch_adv
        })
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim,self.g_sum],
        feed_dict={
          self.gtInputs: batch_gt,
          self.advInputs: batch_adv
        })
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim,self.g_sum],
        feed_dict={ 
          self.gtInputs: batch_gt,
          self.advInputs: batch_adv
        })
        self.writer.add_summary(summary_str, counter)

        errD_fake = self.d_loss_fake.eval({
          self.advInputs: batch_adv, 
        })
        errD_real = self.d_loss_real.eval({
          self.gtInputs: batch_gt,
        })
        errG_real = self.g_loss_real.eval({
          self.advInputs: batch_adv,
        })
        errMSE = self.mse_loss.eval({
          self.gtInputs: batch_gt,
          self.advInputs: batch_adv
        })
        
        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, mse_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG_real + errMSE, errMSE))

        if np.mod(counter, 100) == 1:
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.gtInputs: sample_gt,
                  self.advInputs: sample_adv
              }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                './{}/train_{:02d}_{:04d}_reconstructed.png'.format(config.sample_dir, epoch, idx))
            save_images(sample_adv, image_manifold_size(sample_adv.shape[0]),
                './{}/train_{:02d}_{:04d}_adv.png'.format(config.sample_dir, epoch, idx))
            save_images(sample_gt, image_manifold_size(sample_gt.shape[0]),
                './{}/train_{:02d}_{:04d}_gt.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_h3_lin')
      return tf.nn.sigmoid(h3), h3

  def generator(self, z):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

        gconv1 = lrelu(self.g_bn0(conv2d(z, self.gf_dim, name='g_h0_conv')))
        gconv2 = lrelu(self.g_bn1(conv2d(gconv1, self.gf_dim*2, name='g_h1_conv')))
        
        self.h1, self.h1_w, self.h1_b = deconv2d(
            gconv2, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h2_deconv', with_w=True)
        h1 = tf.nn.relu(self.g_bn2(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3_deconv', with_w=True)

        return tf.nn.tanh(h2)

  def sampler(self, z):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

        gconv1 = lrelu(self.g_bn0(conv2d(z, self.gf_dim, name='g_h0_conv')))
        gconv2 = lrelu(self.g_bn1(conv2d(gconv1, self.gf_dim*2, name='g_h1_conv')))
        
        self.h1, self.h1_w, self.h1_b = deconv2d(
            gconv2, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h2_deconv', with_w=True)
        h1 = tf.nn.relu(self.g_bn2(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3_deconv', with_w=True)

        return tf.nn.tanh(h2)
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
