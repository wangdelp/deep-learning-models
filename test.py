from resnet50 import ResNet50
import numpy as np
import sugartensor as tf
from keras import backend as K
#from imagenet_utils import preprocess_input, decode_predictions
import ipdb

# set session to prevent consuming all GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

z_dim = 100
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
preds = tf.placeholder(tf.float32, [None] + [num_cls])
#imgs = tf.placeholder(tf.float32, [None] + image_shape)
#z = tf.random_uniform((batch_size, z_dim))
gen = generator(z)

model = ResNet50(weights='imagenet')

vars = tf.trainable_variables()
var_gen = [var for var in vars if "generator" in var.name]
loss = -tf.reduce_sum(y*tf.log(preds))
train_gen = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=var_gen)

# only initialize generator variables
init = tf.initialize_variables(varlist=var_gen)
sess.run(init)

# get generator into a seperate function module
# training
for i in range(1000):
  ipdb.set_trace()
  z_, y_ = sampling(sess)
  preds_ = model.predict(gen.eval(feed_dict={z: z_, y: y_}))
  train_gen.run(feed_dict={preds: preds_, y: y_}) 
