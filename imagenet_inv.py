from resnet50 import ResNet50
import numpy as np
import sugartensor as tf 
from keras.layers import Input
from keras import backend as K
from imagenet_utils import preprocess_input, decode_predictions
import ipdb

import sys
sys.path.append("/home/xeraph/lonestar_ai")
from x_utils import sampling
from image_utils import save_images
# set session to prevent consuming all GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

lam = 1e-6
z_dim = 2048 
image_shape = [224, 224, 3]
# 32 ~ 9GB
batch_size = 48 
num_cls = 1000

# construct generator
def generator(z):
  with tf.sg_context(name='generator', size=4, stride=2, act='relu', bn=True):
    # generator network
    gen = (z.sg_dense(dim=2048)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128))
           .sg_upconv(dim=48)
           .sg_upconv(dim=24)
           .sg_upconv(dim=12)
           .sg_upconv(dim=6)
           .sg_upconv(dim=3, act='tanh', bn=False))
    return gen

z = tf.placeholder(tf.float32, [None] + [z_dim])
y = tf.placeholder(tf.float32, [None] + [num_cls])
#preds = tf.placeholder(tf.float32, [None] + [num_cls])
#imgs = Input(batch_shape=[None] + image_shape)
#z = tf.random_uniform((batch_size, z_dim))

#model = ResNet50(input_tensor=imgs, weights='imagenet')
# glue the generator with discriminator using input_tensor
gen = generator(z)
model = ResNet50(input_tensor=gen, weights='imagenet')
preds = model.layers[-1].output
correct = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
accu = tf.reduce_mean(tf.cast(correct, tf.float32))
#loss_pred = -tf.reduce_sum(y*tf.log(preds + 1e-10))
loss_pred = -tf.reduce_sum(y*tf.log(tf.clip_by_value(preds, 1e-8, 1.0)))
#loss_pred = -tf.reduce_sum(y*tf.log(preds))
#fd = {K.learning_phase(): 0, imgs: gen.eval({z: z_})}

vars = tf.trainable_variables()
var_gen = [var for var in vars if "generator" in var.name]
var_dis = [var for var in vars if "generator" not in var.name]
reg_gen = [tf.nn.l2_loss(var) for var in var_gen]
loss_reg = sum(reg_gen)
loss = loss_pred + lam * loss_reg

var_before = tf.all_variables()
# learning rate and global_step variable
var_lr_step = list(set(var_before).difference(set(vars)))
train_gen = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=var_gen)
var_after = tf.all_variables()
var_adam = list(set(var_after).difference(var_before))

# ---- get generator into a seperate function module ----
# training
with sess.as_default():
  # only initialize generator variables
  init = tf.initialize_variables(var_gen)
  #tf.sg_init(sess)  # would also initilize discriminator variables
  sess.run(init)
  adam_ini = tf.initialize_variables(var_adam)
  sess.run(adam_ini)
  local_ini = tf.initialize_local_variables()
  sess.run(local_ini)
  lr_step_ini = tf.initialize_variables(var_lr_step)
  sess.run(lr_step_ini)

  for i in range(30000):
    z_, y_ = sampling(sess, z_dim=z_dim, num_category=1000, bs=batch_size, random=True)

    fd = {z: z_, y: y_, K.learning_phase(): 1}
    if i%3000 == 0:
      imgs = gen.eval(feed_dict={z: z_})
      save_images(imgs, [np.ceil(batch_size/8.0), 8], "./imgs/generated_{}.png".format(i))
      #ipdb.set_trace()

    if i%100 == 0:
      train_accuracy = accu.eval(feed_dict=fd)
      loss_val = loss.eval(fd)
      print("step %d, training accuracy %g, loss: %g"%(i, train_accuracy, loss_val))

    train_gen.run(feed_dict={y: y_, z: z_, K.learning_phase(): 1}) 

  ipdb.set_trace()
  saver = tf.train.Saver()
  ckpt_path = "./checkpoint/generator_lr_1e-3.model"
  saver.save(sess, ckpt_path)

  imgs = gen.eval(feed_dict={z: z_})
  save_images(imgs, [np.ceil(batch_size/8.0), 8], "./imgs/generated_{}.png".format(i))
  save_images(imgs[0:2, :], [1, 2], "./imgs/generated_large.png")
  y_vec = decode_predictions(y_)
  y_pres = decode_predictions(preds.eval(fd))
  ipdb.set_trace()
