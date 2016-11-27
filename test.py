from resnet50 import ResNet50
import numpy as np
import sugartensor as tf 
from keras.applications import vgg16
from keras.layers import Input
from keras import backend as K
#from imagenet_utils import preprocess_input, decode_predictions
import ipdb

import sys
sys.path.append("/home/xeraph/lonestar_ai")
from x_utils import sampling
# set session to prevent consuming all GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

z_dim = 2048 
image_shape = [224, 224, 3]
batch_size = 256
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
           .sg_upconv(dim=3, act='sigmoid', bn=False))
    return gen

z = tf.placeholder(tf.float32, [None] + [z_dim])
y = tf.placeholder(tf.float32, [None] + [num_cls])
#preds = tf.placeholder(tf.float32, [None] + [num_cls])
#imgs = tf.placeholder(tf.float32, [None] + image_shape)
imgs = Input(batch_shape=[None] + image_shape)
#z = tf.random_uniform((batch_size, z_dim))
gen = generator(z)

#model = ResNet50(input_tensor=imgs, weights='imagenet')
model = ResNet50(input_tensor=imgs, weights='imagenet')
output = model.layers[-1].output
#model.layers[-1].output.eval(feed_dict={K.learning_phase(): 0, imgs: gen.eval(fd)}

ipdb.set_trace()
vars = tf.trainable_variables()
var_gen = [var for var in vars if "generator" in var.name]

# ---- get generator into a seperate function module ----
# training
with sess.as_default():
  # only initialize generator variables
  init = tf.initialize_variables(var_gen)
  sess.run(init)
  local_ini = tf.initialize_local_variables()
  sess.run(local_ini)

  for i in range(1000):
    z_, y_ = sampling(sess, z_dim=z_dim, num_category=1000, bs=batch_size, random=True)
    ipdb.set_trace()
    preds = model.predict(gen)
    #preds = model.predict(gen.eval(feed_dict={z: z_}))
    loss = -tf.reduce_sum(y*tf.log(preds))
    train_gen = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=var_gen)
    #train_gen.run(feed_dict={preds: preds_, y: y_}) 
    train_gen.run(feed_dict={y: y_}) 
