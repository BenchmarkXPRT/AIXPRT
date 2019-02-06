# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:31:28 2018

@author: t
"""

import tensorflow as tf
from official.resnet.imagenet_main import ImagenetModel
from tensorflow.contrib.nccl.python.ops import nccl_ops
import sys


#%% Parameters
data_format='channels_last'
output_node_names=["logit"]
output_graph="frozen_graph.pb"

if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]
if len(sys.argv) > 2:
    output_node_names = sys.argv[2].split(',')
if len(sys.argv) > 3:
    output_graph = sys.argv[3]
if len(sys.argv) > 4:
    data_format = sys.argv[4]


#nccl_ops._maybe_load_nccl_ops_so()
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
_ = tf.contrib

height, width = 224, 224
model = ImagenetModel(50,data_format=data_format, resnet_version=1) # This data_format agrees with CPU inference
## Create graph
X = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3])
    
#output_graph = 'resnet_50.pb' # Name of output frozen graph
with tf.Graph().as_default() as tf_graph:
    with tf.Session(config=tf_config) as tf_sess:
    
        
        tf_input = tf.placeholder(tf.float32, [None, height, width, 3], name='input') # Set this according to data_format
        tf_net = model(tf_input, training=False)
        
        tf_output = tf.identity(tf_net, name='softmax_tensor') # 
        tf_saver = tf.train.Saver()
        tf_saver.restore(save_path=checkpoint_path, sess=tf_sess)
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            tf_sess,
            tf_sess.graph_def,
            output_node_names=output_node_names
        )


    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(frozen_graph.SerializeToString())

